"""
Pydantic schemas for vehicle-related data.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum


class FeatureStatusEnum(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    UNCERTAIN = "uncertain"


class FeatureSourceEnum(str, Enum):
    USER = "user"
    DATABASE = "database"
    ML_ESTIMATOR = "ml_estimator"
    TOOL = "tool"


class FeatureValue(BaseModel):
    """Structured representation of a single feature's state."""
    value: Any = None
    status: FeatureStatusEnum = FeatureStatusEnum.UNKNOWN
    source: FeatureSourceEnum = FeatureSourceEnum.USER
    confidence: Optional[float] = None


class FeatureState(BaseModel):
    """Complete feature state for a vehicle assessment."""
    Vehicle_Age: FeatureValue = Field(default_factory=FeatureValue)
    Reported_Issues: FeatureValue = Field(default_factory=FeatureValue)
    Maintenance_History: FeatureValue = Field(default_factory=FeatureValue)
    Service_History: FeatureValue = Field(default_factory=FeatureValue)
    Accident_History: FeatureValue = Field(default_factory=FeatureValue)
    Tire_Condition: FeatureValue = Field(default_factory=FeatureValue)
    Brake_Condition: FeatureValue = Field(default_factory=FeatureValue)
    Battery_Status: FeatureValue = Field(default_factory=FeatureValue)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to dict for ML pipeline."""
        return {
            name: field.model_dump()
            for name, field in [
                ("Vehicle_Age", self.Vehicle_Age),
                ("Reported_Issues", self.Reported_Issues),
                ("Maintenance_History", self.Maintenance_History),
                ("Service_History", self.Service_History),
                ("Accident_History", self.Accident_History),
                ("Tire_Condition", self.Tire_Condition),
                ("Brake_Condition", self.Brake_Condition),
                ("Battery_Status", self.Battery_Status),
            ]
        }

    def get_known_features(self) -> list:
        """Return list of known feature names."""
        return [
            name for name, field in self.to_dict().items()
            if field["status"] == "known"
        ]

    def get_unknown_features(self) -> list:
        """Return list of unknown feature names."""
        return [
            name for name, field in self.to_dict().items()
            if field["status"] == "unknown"
        ]

    def get_declined_features(self) -> list:
        """Return list of features user said 'I don't know' about."""
        return [
            name for name, field in self.to_dict().items()
            if field["status"] == "unavailable"
        ]

    def has_any_known(self) -> bool:
        """Check if at least one feature is known."""
        return len(self.get_known_features()) > 0


class VehicleCreate(BaseModel):
    name: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_age: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission_type: Optional[str] = None


class VehicleResponse(BaseModel):
    id: UUID
    name: Optional[str]
    vehicle_model: Optional[str]
    vehicle_age: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}
