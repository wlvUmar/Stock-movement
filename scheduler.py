import os
import torch
import logging
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from pipeline import StockMovementPipeline
from ml.training import train_stock_model, incremental_train
from utils import settings

logger = logging.getLogger(__name__)

async def run_pipeline():
    try:
        pipeline = StockMovementPipeline()
        await pipeline.run_pipeline()        
        # await run_training()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

async def run_training():
    try:
        device = torch.device('cuda' if settings.USE_GPU and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        TRAINING_CONFIG = settings.TRAINING_CONFIG
        if os.path.exists(settings.MODEL_PATH):
            logger.info("Running incremental training...")
            model = incremental_train(
                csv_path="temp.csv",
                model_path=settings.MODEL_PATH,
                epochs=TRAINING_CONFIG['incremental_epochs'],
                learning_rate=TRAINING_CONFIG['incremental_learning_rate']
            )
        else:
            logger.info("Running initial training...")
            model = train_stock_model(
                csv_path="temp.csv",
                epochs=TRAINING_CONFIG['epochs'],
                batch_size=TRAINING_CONFIG['batch_size'],
                learning_rate=TRAINING_CONFIG['learning_rate'],
                model_save_path=settings.MODEL_PATH
            )
        
        logger.info("Training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def setup_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(
            day_of_week='mon-fri',
            hour=16,
            minute=30
        ),
        id='stock_pipeline',
        name='Stock Data Pipeline',
        replace_existing=True
    )
    
    return scheduler 