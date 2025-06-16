from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from app.schemas.stock import StockDataResponse, PredictionResponse
from app.db import AsyncSessionLocal, StockData
from sqlalchemy import select
from sqlalchemy.sql import func
from ml.training import load_trained_model, predict_next_day
from app.utils import settings
import pandas as pd
import logging
import os
import tempfile
import shutil

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# Load model at startup
model_data = None
if os.path.exists(settings.MODEL_PATH):
    try:
        model_data = load_trained_model(settings.MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
else:
    logger.warning(f"Model file not found at {settings.MODEL_PATH}")

@router.get("/predict/{symbol}", response_model=PredictionResponse)
async def get_prediction(
    symbol: str,
    days_ahead: Optional[int] = Query(1, description="Number of days to predict ahead", ge=1, le=5)
):
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create a temporary directory for our files
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "temp.csv")
    
    try:
        # Get historical data for prediction
        async with AsyncSessionLocal() as session:
            query = select(StockData).where(
                StockData.ticker == symbol
            ).order_by(StockData.date.desc()).limit(60)  # Get last 60 days for prediction
            
            result = await session.execute(query)
            data = result.scalars().all()
            
            if not data:
                raise HTTPException(status_code=404, detail=f"No historical data found for symbol {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': d.date,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume
            } for d in data])
            
            # Save to temp file for prediction
            df.to_csv(temp_file, index=False)
            
            # Get prediction
            prediction = predict_next_day(temp_file, settings.MODEL_PATH)
            
            return {
                "symbol": symbol,
                "prediction_date": datetime.now().date() + timedelta(days=1),
                "predicted_movement": "UP" if prediction > 0.5 else "DOWN",
                "confidence": float(prediction if prediction > 0.5 else 1 - prediction),
                "model_version": settings.VERSION
            }
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

@router.get("/model/status")
async def get_model_status():
    return {
        "model_loaded": model_data is not None,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(settings.MODEL_PATH)).isoformat() if model_data else None,
        "model_version": settings.VERSION,
        "gpu_enabled": settings.USE_GPU
    }
