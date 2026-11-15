from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# 1. Response returned when the user uploads an image for identification
class CandidatePlant(BaseModel):
    species_id: str
    common_name: str
    scientific_name: str
    confidence: float
    light: str = "3-4 hrs bright indirect light"
    water: str = "Every 7 days"
    humidity: str = "Moderate (40-50%)"
    is_toxic_to_pets: bool = False
    toxic_note: str = ""
    required_user_inputs: List[str] = ["height_cm", "pot_size_inches"]

class IdentificationResponse(BaseModel):
    status: str  # "confirmed" or "ambiguous"
    temp_image_path: str
    candidates: List[CandidatePlant]


# 2. Request sent by Frontend after the user picks the plant & fills out height/pot info
class ConfirmPlantRequest(BaseModel):
    user_id: Optional[str] = None  # No longer required from frontend!
    selected_species_id: str
    nickname: Optional[str] = None
    height_cm: float
    pot_size_inches: float
    current_room: str = "Living Room"
    temp_image_path: str


# 3. Final MongoDB Document Structure saved in `user_plants` collection
class UserPlantDocument(BaseModel):
    user_id: str
    species_id: str
    common_name: str
    scientific_name: str
    nickname: str
    height_cm: float
    pot_size_inches: float
    image_url: str

    # Auto-populated from species_catalog
    sensitivity_level: str  # "Hardy" (monthly checkup) or "Sensitive" (weekly checkup)
    is_toxic_to_pets: bool
    ideal_sunlight_hours: str
    recommended_fertilizer: str

    # Dynamic State & Placement (Ready for 3D coordinates later)
    health_status: str = "Thriving"  # "Thriving", "Needs Attention", or "Recovering"
    placement: Dict[str, Any]  # Stores room name + seasonal advice now; (X,Y) later

    # Smart Scheduling Timestamps
    next_water_date: datetime
    next_nutrition_date: datetime
    next_checkup_date: datetime

    # Embedded History Array
    history_logs: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

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