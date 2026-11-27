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



#____________________________________
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
#________________________Error Finding_________________________________________
# 1. Prints 422 Pydantic validation errors (e.g., missing or wrong field types)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n❌ [422 VALIDATION ERROR] {request.method} {request.url}")
    print(f"Details: {exc.errors()}")
    print(f"Body received: {exc.body}\n")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# 2. Prints HTTPExceptions like 404 ("Selected species not found in catalog") or 401
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    print(f"\n⚠️ [HTTP {exc.status_code}] {request.method} {request.url} -> {exc.detail}\n")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# 3. Prints full traceback for 500 Internal Server Errors (e.g., Beanie/MongoDB crash)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"\n🔥 [500 SERVER CRASH] {request.method} {request.url}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})
#_______________________________________________________________________________________

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