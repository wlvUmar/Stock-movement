from pydantic_settings import BaseSettings
from typing import ClassVar
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Movement Prediction API"
    PROJECT_DESCRIPTION: str = """Description goes here"""
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    PG_DB_URL: str = "postgresql+asyncpg://postgres:getout04@localhost:5433/postgres"
    PG_TEST_URL: str = "postgresql+asyncpg://postgres:getout04@localhost:5433/postgres_test"
    ROOT_DIR : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    API_KEY: str = "0de885b71f8146e99ab2ca388d3ee622"
settings = Settings()   
