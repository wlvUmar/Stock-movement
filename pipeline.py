import aiohttp
import logging
import asyncio
import time
import requests
import holidays
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, List, Dict, Generator
import pandas as pd

from sqlalchemy import insert, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dataclasses import dataclass
from contextlib import asynccontextmanager

from db import AsyncSessionLocal, StockData
from utils import settings
from logger_setup import setup_logging

setup_logging("logs/pipeline.log")

@dataclass
class MarketHours:
    open_time: dt_time = dt_time(9, 30)
    close_time: dt_time = dt_time(16, 0)

        

class RateLimiter:
    def __init__(self, calls_per_minute: int= 5):
        self.calls_per_minute = calls_per_minute 
        self.calls_interval = 60.0 /calls_per_minute
        self.last_call = 0 
    
    async def wait(self):
        time_since_last = time.time() - self.last_call
        if time_since_last < self.calls_interval:  
            sleep_time = self.calls_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self.last_call = time.time()


class StockMovementPipeline:
    def __init__(self, symbol: str = "AAPL", batch_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.API_KEY
        self.base_url = "http://127.0.0.1:8000/stream"
        self.symbol = symbol
        self.batch_size = batch_size
        self.market_hours = MarketHours()
        self.rate_limiter = RateLimiter(calls_per_minute=5)
        self.holidays = holidays.US()
        
    def is_market_day(self, date: datetime) -> bool:
        return (date.weekday() < 5 and 
                date.date() not in self.holidays)
    
    def get_market_hours_for_date(self, date: datetime) -> tuple[datetime, datetime]:
        market_open = datetime.combine(date.date(), self.market_hours.open_time)
        market_close = datetime.combine(date.date(), self.market_hours.close_time)
        return market_open, market_close
    
    async def get_stock_data_async(self, start_date: str, end_date: str, 
                                  retries: int = 3) -> pd.DataFrame:
        params = {"start": start_date, "end": end_date}
        
        for attempt in range(retries):
            try:
                await self.rate_limiter.wait()
                
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        response.raise_for_status()
                        records = await response.json()
                        
                if not isinstance(records, list) or not records:
                    self.logger.warning(f"Empty response for {params}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(records)
                
                if "date" not in df.columns:
                    self.logger.error(f"Missing 'date' column: {df.columns.tolist()}")
                    return pd.DataFrame()
                
                # Clean and process data
                df = self._clean_dataframe(df)
                return df
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed for {params}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        
        if "adj_close" in df.columns:
            df = df.drop(columns=["adj_close"])
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
        if df.empty:
            return df
        
        df = df.sort_values("date").reset_index(drop=True)
        df["ticker"] = self.symbol
        df = df.drop_duplicates(subset=["date"], keep="last")
        return df
    
    def generate_time_chunks(self, start_date: str, 
                           chunk_minutes: int = 60) -> Generator[tuple[str, str], None, None]:
        """Generate time chunks for data fetching"""
        
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        while True:
            if not self.is_market_day(current_date):
                current_date += timedelta(days=1)
                continue
                
            market_open, market_close = self.get_market_hours_for_date(current_date)
            current_time = market_open
            
            while current_time < market_close:
                chunk_end = min(current_time + timedelta(minutes=chunk_minutes), 
                              market_close)
                
                yield (
                    current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    chunk_end.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                current_time = chunk_end
            
            current_date += timedelta(days=1)
    
    async def fetch_data_in_chunks(self, start_date: str, max_records: int = 200_000) -> List[pd.DataFrame]:
        all_dataframes = []
        record_count = 0
        chunk_generator = self.generate_time_chunks(start_date, chunk_minutes=60)
        batch_tasks = []
        
        for start_time, end_time in chunk_generator:
            task = self.get_stock_data_async(start_time, end_time)
            batch_tasks.append(task)
            
            if len(batch_tasks) >= 5:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        all_dataframes.append(result)
                        record_count += len(result)
                        if record_count >= max_records:
                            self.logger.info(f"Reached record limit: {record_count}")
                            return all_dataframes
                    elif isinstance(result, Exception):
                        self.logger.error(f"Batch request failed: {result}")
                
                batch_tasks = []
                self.logger.info(f"Processed {record_count} requests")
        
        if batch_tasks and record_count < max_records:
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_dataframes.append(result)
                    record_count += len(result)
                    if record_count >= max_records:
                        break

        return all_dataframes
    
    async def get_latest_date(self) -> datetime:
        """Get latest date from database"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.max(StockData.date)))
            return result.scalar() or datetime(2006, 1, 3)

    async def load_in_batches(self, records: List[Dict]) -> None:
        async with AsyncSessionLocal() as session:
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                try:
                    async with session.begin():
                        stmt = pg_insert(StockData).values(batch)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['ticker', 'date'],
                            set_={
                                'open': stmt.excluded.open,
                                'high': stmt.excluded.high,
                                'low': stmt.excluded.low,
                                'close': stmt.excluded.close,
                                'volume': stmt.excluded.volume
                            }
                        )
                        await session.execute(stmt)
                        
                    self.logger.info(f"Loaded batch {i // self.batch_size + 1}: {len(batch)} records")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load batch {i // self.batch_size + 1}: {e}")
                    continue
    
    async def extract(self, start_date: Optional[str] = None) -> None:
        try:
            if not start_date:
                date = datetime.now() - timedelta(days=14)
                start_date = date.strftime("%Y-%m-%d")  # Fixed format
            
            self.logger.info(f"Starting extraction from {start_date} for {self.symbol}")
            
            dataframes = await self.fetch_data_in_chunks(start_date)
            
            if not dataframes:
                self.logger.warning(f"No data extracted for {self.symbol}")
                self.raw_data = pd.DataFrame()
                return
            
            self.raw_data = pd.concat(dataframes, ignore_index=True)
            self.raw_data = self.raw_data.drop_duplicates(subset=['date'], keep='last')
            self.raw_data = self.raw_data.sort_values("date").reset_index(drop=True)
            
            self.logger.info(f"Extracted {len(self.raw_data)} rows for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            raise
    
    def transform(self) -> None:
        try:
            if not hasattr(self, "raw_data") or self.raw_data.empty:
                raise ValueError("No data to transform. Run extract() first.")
            
            df = self.raw_data.copy()
            
            required_columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.transformed_data = df[required_columns].copy()
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                self.transformed_data[col] = pd.to_numeric(
                    self.transformed_data[col], errors='coerce'
                )
            
            initial_count = len(self.transformed_data)
            self.transformed_data = self.transformed_data.dropna()
            final_count = len(self.transformed_data)
            
            if initial_count != final_count:
                self.logger.warning(f"Removed {initial_count - final_count} rows with invalid data")
            
            self.logger.info(f"Transformation complete: {final_count} valid rows")
            
        except Exception as e:
            self.logger.error(f"Transformation error: {e}")
            raise
    
    
    async def load(self) -> None:
        try:
            if not hasattr(self, "transformed_data") or self.transformed_data.empty:
                raise ValueError("No transformed data to load. Run transform() first.")
            
            self.transformed_data.to_csv("temp.csv", index=False)
            self.logger.info("Data saved to temp.csv")
            
            records = self.transformed_data.to_dict(orient="records")
            await self.load_in_batches(records)
            
            self.logger.info(f"Successfully loaded {len(records)} records to database")
            
        except Exception as e:
            self.logger.error(f"Load error: {e}")
            raise
    
    async def run_pipeline(self, start_date: Optional[str] = None) -> None:
        try:
            if start_date is None:
                start_date = (await self.get_latest_date()).strftime("%Y-%m-%d")
            
            await self.extract(start_date)
            self.transform()
            await self.load()
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

async def main(start_date=None):
    pipeline = StockMovementPipeline(symbol="AAPL", batch_size=1000)
    await pipeline.run_pipeline(start_date)

# Run with: asyncio.run(main())