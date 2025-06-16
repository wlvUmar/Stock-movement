from fastapi import FastAPI
from app.routers import router
from app.utils import settings
from app.db import engine, Base
from app.scheduler import setup_scheduler

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
    
    scheduler = setup_scheduler()
    scheduler.start()
