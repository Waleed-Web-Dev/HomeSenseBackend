from datetime import datetime, timezone
from typing import Optional, List
from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field


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