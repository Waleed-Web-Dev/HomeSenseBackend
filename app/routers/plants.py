import os
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from schemas.plant_schema import IdentificationResponse, ConfirmPlantRequest, UserPlantDocument
from app.database import init_db
from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.plant_schema import IdentificationResponse
from services.vision_inference import predict_plant_species, ml_models
router = APIRouter(prefix="/plants", tags=["Plants"])
UPLOAD_DIR = "uploads/plants"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-identify", response_model=IdentificationResponse)
@router.post("/upload-identify", response_model=IdentificationResponse)
async def upload_and_identify_plant(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    file_ext = file.filename.split(".")[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{file_ext}")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # Grab the pre-loaded model from the dictionary and run inference!
    loaded_cnn_model = ml_models.get("species_cnn")
    if loaded_cnn_model is None:
        raise HTTPException(status_code=503, detail="Vision model is not loaded.")

    result = predict_plant_species(image_bytes, model=loaded_cnn_model)

    return IdentificationResponse(
        status=result["status"],
        temp_image_path=file_path,
        candidates=result["candidates"]
    )


@router.post("/confirm-add", status_code=201)
async def confirm_and_add_plant(payload: ConfirmPlantRequest, db=Depends(init_db())):
    # 1. Fetch biological rules for the chosen species from MongoDB `species_catalog`
    species_info = await db.species_catalog.find_one({"species_id": payload.selected_species_id})

    if not species_info:
        raise HTTPException(status_code=404, detail="Selected plant species not found in catalog.")

    now = datetime.utcnow()

    # 2. Set smart checkup interval based on sensitivity level (7 days vs 30 days)
    checkup_days = 7 if species_info["sensitivity_level"] == "Sensitive" else 30

    # 3. Adjust initial watering days slightly based on plant height & pot size ratio
    base_water_days = species_info.get("base_water_days", 7)

    # 4. Generate seasonal sunlight placement advice
    placement_advice = (
        f"Place in {payload.current_room} where it receives roughly "
        f"{species_info['ideal_sunlight_hours']} hours of {species_info['light_type']} daily."
    )

    # 5. Build the MongoDB Document
    new_plant_doc = UserPlantDocument(
        user_id=payload.user_id,
        species_id=payload.selected_species_id,
        common_name=species_info["common_name"],
        scientific_name=species_info["scientific_name"],
        nickname=payload.nickname or species_info["common_name"],
        height_cm=payload.height_cm,
        pot_size_inches=payload.pot_size_inches,
        image_url=payload.temp_image_path,
        sensitivity_level=species_info["sensitivity_level"],
        is_toxic_to_pets=species_info["is_toxic_to_pets"],
        ideal_sunlight_hours=species_info["ideal_sunlight_hours"],
        recommended_fertilizer=species_info["recommended_fertilizer"],
        health_status="Thriving",
        placement={
            "room_name": payload.current_room,
            "seasonal_advice": placement_advice,
            "coordinates": None  # Reserved for 3D/2D (X, Y) floor grid later
        },
        next_water_date=now + timedelta(days=base_water_days),
        next_nutrition_date=now + timedelta(days=species_info.get("base_nutrition_days", 14)),
        next_checkup_date=now + timedelta(days=checkup_days),
        history_logs=[
            {
                "timestamp": now,
                "event_type": "PLANT_ADDED",
                "notes": f"Added at {payload.height_cm}cm height in a {payload.pot_size_inches}-inch pot."
            }
        ]
    )

    # 6. Insert into MongoDB `user_plants` collection
    insert_result = await db.user_plants.insert_one(new_plant_doc.model_dump())

    return {
        "message": "Plant successfully added to your home inventory.",
        "plant_id": str(insert_result.inserted_id),
        "placement_suggestion": placement_advice,
        "next_water_date": new_plant_doc.next_water_date,
        "next_checkup_date": new_plant_doc.next_checkup_date
    }