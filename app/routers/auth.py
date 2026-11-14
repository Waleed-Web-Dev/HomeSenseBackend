from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas.models import User
from app.schemas.schemas import UserCreate, UserResponse, UserOnboardingUpdate, Token
from app.security import get_password_hash, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication & Onboarding"])

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    existing_user = await User.find_one(User.email == user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email is already registered")

    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        gps_lat=user_data.gps_lat,
        gps_lon=user_data.gps_lon,
        city_name=user_data.city_name,
        has_pets=user_data.has_pets,
        has_children=user_data.has_children
    )
    await new_user.insert()

    access_token = create_access_token(data={"sub": str(new_user.id)})
    return {"access_token": access_token, "token_type": "bearer", "user": new_user}

@router.post("/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.find_one(User.email == form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@router.put("/onboarding-context", response_model=UserResponse)
async def update_onboarding_context(
    context_data: UserOnboardingUpdate,
    current_user: User = Depends(get_current_user)
):
    current_user.gps_lat = context_data.gps_lat
    current_user.gps_lon = context_data.gps_lon
    current_user.city_name = context_data.city_name
    current_user.has_pets = context_data.has_pets
    current_user.has_children = context_data.has_children

    await current_user.save()
    return current_user

@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    return current_user