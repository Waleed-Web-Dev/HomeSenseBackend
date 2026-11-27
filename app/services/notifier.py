from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, messaging
from beanie import PydanticObjectId
from app.schemas.models import User, UserPlant

# Initialize once at startup
if not firebase_admin._apps:
    cred = credentials.Certificate("app/aiFirebaseCredential.json")
    firebase_admin.initialize_app(cred)

def send_plant_push(fcm_token: str, title: str, body: str, plant_id: str, action_type: str):
    if not fcm_token:
        return
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={
            "plant_id": plant_id,
            "action_type": action_type,  # "WATER", "NUTRITION", "CHECKUP", "RELOCATE"
        },
        token=fcm_token,
    )
    try:
        messaging.send(message)
    except Exception as e:
        print(f"FCM send failed for plant {plant_id}: {e}")

async def check_and_send_plant_reminders():
    """Hourly background job scanning due plant tasks across all users."""
    now = datetime.utcnow()
    # Find plants with any due timestamp
    due_plants = await UserPlant.find({
        "$or": [
            {"next_water_date": {"$lte": now}},
            {"next_nutrition_date": {"$lte": now}},
            {"next_checkup_date": {"$lte": now}},
        ]
    }).to_list()

    for plant in due_plants:
        user = await User.get(PydanticObjectId(plant.user_id))
        if not user or not user.fcm_token:
            continue

        prefs = user.notification_prefs
        plant_id_str = str(plant.id)
        name = plant.nickname or plant.common_name

        # 1. Watering Notification
        if plant.next_water_date <= now and not prefs.mute_routine_watering:
            send_plant_push(
                user.fcm_token,
                title=f"💧 Water {name}",
                body=f"Your {plant.common_name} in {plant.placement.get('room_name', 'your room')} is ready for watering.",
                plant_id=plant_id_str,
                action_type="WATER"
            )
            # Snooze reminder by 24h until user marks task complete
            plant.next_water_date = now + timedelta(hours=24)

        # 2. Nutrition / Fertilizer Notification
        if plant.next_nutrition_date <= now and prefs.nutrition_reminders:
            send_plant_push(
                user.fcm_token,
                title=f"🌱 Feed {name}",
                body=f"Time for nutrition: Apply {plant.recommended_fertilizer}.",
                plant_id=plant_id_str,
                action_type="NUTRITION"
            )
            plant.next_nutrition_date = now + timedelta(hours=24)

        # 3. Smart Health Checkup / FSM Recovery Prompt
        if plant.next_checkup_date <= now:
            is_recovery = plant.health_status == "Recovering"
            if (is_recovery and prefs.recovery_mode_alerts) or (not is_recovery and prefs.health_checkup_prompts):
                title = f"🚨 Recovery Checkup: {name}" if is_recovery else f"📸 Health Checkup: {name}"
                body = (
                    f"{name} is in Recovery Mode. Snap a quick leaf photo so AI can track healing."
                    if is_recovery else
                    f"Time for {name}'s periodic checkup ({plant.sensitivity_level} schedule)."
                )
                send_plant_push(
                    user.fcm_token,
                    title=title,
                    body=body,
                    plant_id=plant_id_str,
                    action_type="CHECKUP"
                )
                plant.next_checkup_date = now + timedelta(hours=24)

        await plant.save()