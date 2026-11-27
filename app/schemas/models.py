from datetime import datetime, timezone
from typing import Optional, List
from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from beanie import Document
from pydantic import BaseModel, Field
import pymongo

class User(Document):
    full_name: str
    email: Indexed(EmailStr, unique=True)
    hashed_password: str

    # Environmental & Safety Context
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    city_name: Optional[str] = None
    has_pets: bool = False
    has_children: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"  # MongoDB Collection name


class SpeciesCatalog(Document):
    species_id: str
    common_name: str
    scientific_name: str
    sensitivity_level: str          # "Hardy" or "Sensitive"
    is_toxic_to_pets: bool
    ideal_sunlight_hours: str
    light_type: str
    base_water_days: int
    base_nutrition_days: int
    recommended_fertilizer: str

    class Settings:
        name = "species_catalog"
        indexes = [
            pymongo.IndexModel([("species_id", pymongo.ASCENDING)], unique=True)
        ]


class UserPlant(Document):
    user_id: str
    species_id: str
    common_name: str
    scientific_name: str
    nickname: str
    height_cm: float
    pot_size_inches: float
    image_url: str
    sensitivity_level: str
    is_toxic_to_pets: bool
    ideal_sunlight_hours: str
    recommended_fertilizer: str
    health_status: str = "Thriving"
    placement: Dict[str, Any]
    next_water_date: datetime
    next_nutrition_date: datetime
    next_checkup_date: datetime
    history_logs: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "user_plants"


# --- Request / Response Pydantic Schemas for the Router ---
class CandidatePlant(BaseModel):
    species_id: str
    common_name: str
    scientific_name: str
    confidence: float
    required_user_inputs: List[str] = ["height_cm", "pot_size_inches"]

class IdentificationResponse(BaseModel):
    status: str
    temp_image_path: str
    candidates: List[CandidatePlant]

class ConfirmPlantRequest(BaseModel):
    # Remove `user_id: str` since it now comes from `current_user` via JWT!
    selected_species_id: str
    nickname: Optional[str] = None
    height_cm: float
    pot_size_inches: float
    current_room: str
    temp_image_path: str
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

class HistoryLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # e.g., "PLANT_ADDED", "HEALTH_CHECKUP", "WATERED"
    image_url: Optional[str] = None
    condition: Optional[str] = None
    confidence: Optional[float] = None
    canopy_greenness_pct: Optional[float] = None
    health_status_assigned: Optional[str] = None
    treatment_recommended: Optional[str] = None
    notes: Optional[str] = None