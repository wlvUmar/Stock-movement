import os
from fastapi import FastAPI
import logging
from db import engine, Base
from app.routers import router
from utils import settings
from scheduler import setup_scheduler
from ml.training import load_trained_model
from logger_setup import setup_logging

setup_logging("logs/app.log")

logger = logging.getLogger(__name__)
model_data = None


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    if os.path.exists(settings.MODEL_PATH):
        try:
            app.state.model = load_trained_model(settings.MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model file not found at {settings.MODEL_PATH}")
    # scheduler = setup_scheduler()
    # scheduler.start()
