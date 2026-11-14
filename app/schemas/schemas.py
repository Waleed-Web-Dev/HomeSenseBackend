from typing import Optional
from pydantic import BaseModel, EmailStr
from beanie import PydanticObjectId

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    city_name: Optional[str] = None
    has_pets: bool = False
    has_children: bool = False

class UserOnboardingUpdate(BaseModel):
    gps_lat: float
    gps_lon: float
    city_name: Optional[str] = None
    has_pets: bool
    has_children: bool

class UserResponse(BaseModel):
    id: PydanticObjectId
    full_name: str
    email: EmailStr
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    city_name: Optional[str] = None
    has_pets: bool
    has_children: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse