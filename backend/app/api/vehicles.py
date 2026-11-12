"""
Vehicles API Endpoints — Vehicle registry management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from backend.app.schemas.vehicle import VehicleCreate, VehicleResponse
from backend.app.database.models import Vehicle
from backend.app.database.session import get_db

router = APIRouter(prefix="/api/vehicles", tags=["Vehicles"])


@router.post("", response_model=VehicleResponse)
async def create_vehicle(
    request: VehicleCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new vehicle record."""
    v = Vehicle(
        id=uuid.uuid4(),
        name=request.name or "My Vehicle",
        vehicle_model=request.vehicle_model,
        vehicle_age=request.vehicle_age,
        fuel_type=request.fuel_type,
        transmission_type=request.transmission_type,
    )
    db.add(v)
    await db.commit()
    await db.refresh(v)
    return VehicleResponse.model_validate(v)


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(
    vehicle_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get vehicle by ID."""
    try:
        v_uuid = uuid.UUID(vehicle_id)
        stmt = select(Vehicle).where(Vehicle.id == v_uuid)
        res = await db.execute(stmt)
        v = res.scalar_one_or_none()
        if not v:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        return VehicleResponse.model_validate(v)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vehicle ID format")
