from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class StockDataResponse(BaseModel):
    id: int
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    symbol: str
    prediction_date: datetime
    predicted_movement: str
    confidence: float
    model_version: str 