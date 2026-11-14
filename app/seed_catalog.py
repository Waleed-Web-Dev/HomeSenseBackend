import json
import asyncio
from app.database import init_db  # Adjust import path if database.py is in root
from app.schemas.models import SpeciesCatalog


async def seed():
    # 1. Initialize Beanie and connect using your .env MONGODB_URL & DATABASE_NAME
    await init_db()

    # 2. Load the 47-plant JSON generated from Colab
    with open("../ml_weights/catalog_seed.json", "r") as f:
        species_data = json.load(f)

    # 3. Clear existing catalog and insert fresh Beanie documents
    await SpeciesCatalog.delete_all()

    docs = [SpeciesCatalog(**item) for item in species_data]
    await SpeciesCatalog.insert_many(docs)

    print(f"Successfully seeded {len(docs)} species into `species_catalog` via Beanie!")


if __name__ == "__main__":
    asyncio.run(seed())