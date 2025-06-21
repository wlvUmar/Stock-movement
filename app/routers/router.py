import os
import logging
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.sql import func

from ml.training import predict_next_day
from app.schemas import StockDataResponse, PredictionResponse
from db import AsyncSessionLocal, StockData
from utils import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")



@router.get("/predict/{symbol}", response_model=PredictionResponse)
async def get_prediction(request: Request,
    symbol: str,
    minutes_ahead: Optional[int] = Query(1, description="Number of days to predict ahead", ge=1, le=15)
):
    model_data = request.app.state.model
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "temp.csv")
    
    try:
        async with AsyncSessionLocal() as session:
            query = select(StockData).where(
                StockData.ticker == symbol
            ).order_by(StockData.date.desc()).limit(60)
            
            result = await session.execute(query)
            data = result.scalars().all()
            
            if not data:
                raise HTTPException(status_code=404, detail=f"No historical data found for symbol {symbol}")
            
            df = pd.DataFrame([{
                'date': d.date,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume
            } for d in data])
            
            df.to_csv(temp_file, index=False)
            prediction = predict_next_day(temp_file, settings.MODEL_PATH, minutes_ahead=minutes_ahead)
            
            return {
                "symbol": symbol,
                "predicted_movement": "UP" if prediction > 0.5 else "DOWN",
                "confidence": float(prediction if prediction > 0.5 else 1 - prediction),
                "model_version": settings.VERSION
            }
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)

@router.get("/model/status")
async def get_model_status(request:Request):
    model_data = request.app.state.model
    return {
        "model_loaded": model_data is not None,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(settings.MODEL_PATH)).isoformat() if model_data else None,
        "model_version": settings.VERSION,
        "gpu_enabled": settings.USE_GPU
    }
