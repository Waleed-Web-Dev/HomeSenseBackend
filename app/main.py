from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.routers import auth
from app.services.vision_inference import load_plant_model, ml_models, init_species_model, init_disease_model
from app.routers import plants
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.services.notifier import check_and_send_plant_reminders


scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MongoDB + Beanie models on startup
    await init_db()
    init_species_model()
    init_disease_model()
    scheduler.add_job(check_and_send_plant_reminders, "interval", hours=1)
    scheduler.start()
    yield
    ml_models.clear()
    scheduler.shutdown()
    # Any cleanup logic on shutdown goes here

app = FastAPI(
    title="FloraTwin AI Backend",
    description="MongoDB + FastAPI Backend for Plant Identification, Weather Care, and Smart Checkups",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Flutter / React Native frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(plants.router)
@app.get("/")
async def health_check():
    return {"status": "online", "database": "MongoDB", "message": "FloraTwin AI API is running"}