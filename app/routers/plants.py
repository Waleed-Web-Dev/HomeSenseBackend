import os
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.schemas.models import (
    SpeciesCatalog,
    UserPlant,
    IdentificationResponse,
    CandidatePlant,
    ConfirmPlantRequest,
    HistoryLogEntry
)
from app.services.vision_inference import run_species_inference, CONFIDENCE_THRESHOLD
from app.services.disease_inference import disease_inference
from app.security import get_current_user
from app.schemas.models import User
from fastapi import Depends
from typing import List, Optional
from beanie import PydanticObjectId
router = APIRouter(prefix="/plants", tags=["Plants"])
UPLOAD_DIR = "uploads/plants"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-identify", response_model=IdentificationResponse)
async def upload_and_identify_plant(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    # 1. Run ONNX inference in threadpool
    raw_predictions = await run_in_threadpool(run_species_inference, image_bytes)

    # 2. Query MongoDB via Beanie and enrich with biological & toxicity data
    candidates: List[CandidatePlant] = []
    for pred in raw_predictions:
        catalog_doc = await SpeciesCatalog.find_one(
            SpeciesCatalog.species_id == pred["species_id"]
        )

        if catalog_doc:
            humidity_str = (
                "High (50-70%)"
                if catalog_doc.sensitivity_level == "Sensitive"
                else "Moderate (40-50%)"
            )
            toxic_msg = (
                f"{catalog_doc.common_name} is toxic to pets and children if ingested."
                if catalog_doc.is_toxic_to_pets
                else ""
            )

            candidates.append(
                CandidatePlant(
                    species_id=catalog_doc.species_id,
                    common_name=catalog_doc.common_name,
                    scientific_name=catalog_doc.scientific_name,
                    confidence=round(float(pred["confidence"]), 4),
                    light=f"{catalog_doc.ideal_sunlight_hours} hrs {catalog_doc.light_type}",
                    water=f"Every {catalog_doc.base_water_days} days",
                    humidity=humidity_str,
                    is_toxic_to_pets=catalog_doc.is_toxic_to_pets,
                    toxic_note=toxic_msg,
                    required_user_inputs=["height_cm", "pot_size_inches"],
                )
            )
        else:
            # Fallback if a predicted class isn't in SpeciesCatalog yet
            fallback_name = pred.get("common_name", pred["species_id"].replace("sp_", "").replace("_", " ").title())
            candidates.append(
                CandidatePlant(
                    species_id=pred["species_id"],
                    common_name=fallback_name,
                    scientific_name=pred.get("scientific_name", fallback_name),
                    confidence=round(float(pred["confidence"]), 4),
                    required_user_inputs=["height_cm", "pot_size_inches"],
                )
            )

    if not candidates:
        raise HTTPException(status_code=422, detail="Could not identify any plant from the image.")

    # 3. Check confidence threshold (0.75) for confirmed vs Top-3 ambiguous disambiguation
    if candidates[0].confidence >= CONFIDENCE_THRESHOLD:
        return IdentificationResponse(
            status="confirmed",
            temp_image_path=temp_path,
            candidates=[candidates[0]],
        )

    return IdentificationResponse(
        status="ambiguous",
        temp_image_path=temp_path,
        candidates=candidates,
    )


@router.post("/confirm-add", status_code=201)
async def confirm_and_add_plant(
    payload: ConfirmPlantRequest,
    current_user: User = Depends(get_current_user)  # <-- Added JWT user dependency
):
    # 1. Fetch species from catalog via Beanie
    species_info = await SpeciesCatalog.find_one(
        SpeciesCatalog.species_id == payload.selected_species_id
    )
    if not species_info:
        raise HTTPException(status_code=404, detail="Selected species not found in catalog.")

    now = datetime.utcnow()
    checkup_days = 7 if species_info.sensitivity_level == "Sensitive" else 30
    pot_factor = 0.85 if (payload.height_cm / max(payload.pot_size_inches, 1)) > 8 else 1.0
    water_days = max(2, round(species_info.base_water_days * pot_factor))

    month = now.month
    season = (
        "Winter" if month in (12, 1, 2)
        else "Spring" if month in (3, 4, 5)
        else "Summer" if month in (6, 7, 8)
        else "Autumn"
    )

    placement_advice = (
        f"For {season}, place your {species_info.common_name} in {payload.current_room} "
        f"where it receives roughly {species_info.ideal_sunlight_hours} hours of {species_info.light_type}."
    )

    # 2. Create and insert Beanie UserPlant document linked to the logged-in user
    new_plant = UserPlant(
        user_id=str(current_user.id),  # <-- Changed from payload.user_id to str(current_user.id)
        species_id=payload.selected_species_id,
        common_name=species_info.common_name,
        scientific_name=species_info.scientific_name,
        nickname=payload.nickname or species_info.common_name,
        height_cm=payload.height_cm,
        pot_size_inches=payload.pot_size_inches,
        image_url=payload.temp_image_path,
        sensitivity_level=species_info.sensitivity_level,
        is_toxic_to_pets=species_info.is_toxic_to_pets,
        ideal_sunlight_hours=species_info.ideal_sunlight_hours,
        recommended_fertilizer=species_info.recommended_fertilizer,
        health_status="Thriving",
        placement={
            "room_name": payload.current_room,
            "season": season,
            "seasonal_advice": placement_advice,
            "coordinates": None
        },
        next_water_date=now + timedelta(days=water_days),
        next_nutrition_date=now + timedelta(days=species_info.base_nutrition_days),
        next_checkup_date=now + timedelta(days=checkup_days),
        history_logs=[{
            "timestamp": now,
            "event_type": "PLANT_ADDED",
            "notes": f"Added {species_info.common_name} ({payload.height_cm}cm, {payload.pot_size_inches}\" pot)."
        }]
    )

    await new_plant.insert()

    return {
        "plant_id": str(new_plant.id),
        "common_name": new_plant.common_name,
        "toxicity_warning": "Keep off floor level (Toxic to pets/kids)" if new_plant.is_toxic_to_pets else "Pet-safe",
        "placement_suggestion": placement_advice,
        "schedule": {
            "next_water_date": new_plant.next_water_date,
            "next_nutrition_date": new_plant.next_nutrition_date,
            "next_checkup_date": new_plant.next_checkup_date,
            "fertilizer_guide": new_plant.recommended_fertilizer
        }
    }

@router.get("/my-plants")
async def get_my_plants(current_user: User = Depends(get_current_user)):
    # Query Beanie directly using the authenticated user's ID
    plants = await UserPlant.find(UserPlant.user_id == str(current_user.id)).to_list()

    return {
        "count": len(plants),
        "plants": plants
    }


from beanie import PydanticObjectId


@router.get("/{plant_id}")
async def get_single_plant(
        plant_id: PydanticObjectId,
        current_user: User = Depends(get_current_user)
):
    plant = await UserPlant.get(plant_id)

    if not plant or plant.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Plant not found in your inventory.")

    return plant

from datetime import datetime, timedelta
import os
import uuid
from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# Adjust imports to match your project structure
# from app.models import User, UserPlant
# from app.security import get_current_user
# from app.ml import disease_inference


class DiagnosisResponse(BaseModel):
    condition: str
    confidence: float
    advice: str
    healthy: bool


def evaluate_fsm_transition(
    current_status: str, sensitivity: str, is_healthy: bool, confidence: float
) -> tuple[str, int]:
    """
    Finite State Machine (FSM) for Plant Health Badges & Checkup Intervals:
    - Healthy -> 'Thriving' (7d Sensitive / 30d Hardy)
    - High-confidence disease (>= 0.55) -> 'Recovering' (3-day Recovery Mode)
    - Low-confidence anomaly (< 0.55) -> 'Needs Attention' (5-day checkup)
    """
    if is_healthy:
        new_status = "Thriving"
        next_days = 7 if sensitivity == "Sensitive" else 30
    elif confidence >= 0.55:
        new_status = "Recovering"
        next_days = 3
    else:
        new_status = "Needs Attention"
        next_days = 5

    return new_status, next_days


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_plant_health(
    plant_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    # 1. Validate image file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 2. Verify plant exists and belongs to the logged-in user
    try:
        plant = await UserPlant.get(PydanticObjectId(plant_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid plant ID format.")

    if not plant or plant.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Plant not found in your garden.")

    # 3. Save checkup image
    image_bytes = await file.read()
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    checkup_path = os.path.join(UPLOAD_DIR, f"checkup_{uuid.uuid4()}.{ext}")
    with open(checkup_path, "wb") as f:
        f.write(image_bytes)

    # 4. Run disease/health ONNX inference in threadpool
    prediction = await run_in_threadpool(
        disease_inference, image_bytes, plant.species_id
    )

    is_healthy: bool = prediction["healthy"]
    condition: str = prediction["condition"]
    confidence: float = round(float(prediction["confidence"]), 2)
    base_advice: str = prediction["advice"]
    greenness: float = prediction.get("canopy_greenness_pct", 0.0)

    # 5. Evaluate FSM transition (keeps health_status strictly to the 3 UI badges)
    new_status, next_days = evaluate_fsm_transition(
        current_status=plant.health_status,
        sensitivity=plant.sensitivity_level,
        is_healthy=is_healthy,
        confidence=confidence,
    )

    # 6. Check past history logs to make the agent advice context-aware
    last_checkups = [
        log for log in plant.history_logs if log.get("event_type") == "HEALTH_CHECKUP"
    ]
    history_note = ""
    if last_checkups:
        prev_cond = last_checkups[-1].get("condition")
        prev_healthy = last_checkups[-1].get("healthy", True)
        if not prev_healthy and is_healthy:
            history_note = f"Great progress! Your plant has recovered from {prev_cond}."
        elif prev_cond == condition and not is_healthy:
            history_note = f"Persistent {condition} detected since last checkup. Continue treatment closely."

    full_advice = f"{base_advice} {history_note}".strip()

    # 7. Update plant health status, next checkup date, and history logs in MongoDB
    now = datetime.utcnow()
    plant.health_status = new_status
    plant.next_checkup_date = now + timedelta(days=next_days)
    plant.history_logs.append({
        "timestamp": now,
        "event_type": "HEALTH_CHECKUP",
        "condition": condition,
        "confidence": confidence,
        "canopy_greenness_pct": greenness,
        "health_status_assigned": new_status,
        "healthy": is_healthy,
        "image_url": checkup_path,
        "notes": full_advice,
    })

    await plant.save()

    return DiagnosisResponse(
        condition=condition,
        confidence=confidence,
        advice=full_advice,
        healthy=is_healthy,
    )