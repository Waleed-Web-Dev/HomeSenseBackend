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
    ConfirmPlantRequest
)
from app.services.vision_inference import run_species_inference, CONFIDENCE_THRESHOLD
from app.security import get_current_user
from app.schemas.models import User
from fastapi import Depends
from beanie import PydanticObjectId
router = APIRouter(prefix="/plants", tags=["Plants"])
UPLOAD_DIR = "uploads/plants"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-identify", response_model=IdentificationResponse)
async def upload_and_identify_plant(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    # 1. Run ONNX inference in threadpool
    raw_predictions = await run_in_threadpool(run_species_inference, image_bytes)

    # 2. Query MongoDB via Beanie
    candidates = []
    for pred in raw_predictions:
        catalog_doc = await SpeciesCatalog.find_one(
            SpeciesCatalog.species_id == pred["species_id"]
        )
        if catalog_doc:
            candidates.append(CandidatePlant(
                species_id=catalog_doc.species_id,
                common_name=catalog_doc.common_name,
                scientific_name=catalog_doc.scientific_name,
                confidence=pred["confidence"],
                required_user_inputs=["height_cm", "pot_size_inches"]
            ))

    if candidates and candidates[0].confidence >= CONFIDENCE_THRESHOLD:
        return IdentificationResponse(
            status="confirmed",
            temp_image_path=temp_path,
            candidates=[candidates[0]]
        )

    return IdentificationResponse(
        status="ambiguous",
        temp_image_path=temp_path,
        candidates=candidates
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