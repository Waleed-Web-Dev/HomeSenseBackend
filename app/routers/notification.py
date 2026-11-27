from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas.models import User
from app.schemas.schemas import UserCreate, UserResponse, UserOnboardingUpdate, Token
from app.security import get_password_hash, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication & Onboarding"])

@router.post("/users/fcm-token")
async def save_fcm_token(
    payload: dict,  # {"fcm_token": "..."}
    current_user: User = Depends(get_current_user)
):
    current_user.fcm_token = payload.get("fcm_token")
    await current_user.save()
    return {"status": "token_registered"}