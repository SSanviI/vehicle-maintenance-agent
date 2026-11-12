"""
SQLAlchemy ORM models for the vehicle maintenance system.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, JSON, Enum as SAEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship as sa_relationship, DeclarativeBase
import enum


class Base(DeclarativeBase):
    pass


class FeatureStatus(str, enum.Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    UNCERTAIN = "uncertain"


class FeatureSource(str, enum.Enum):
    USER = "user"
    DATABASE = "database"
    ML_ESTIMATOR = "ml_estimator"
    TOOL = "tool"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# =============================================================================
# User
# =============================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(255), unique=True, nullable=True)
    display_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    vehicles = sa_relationship("Vehicle", back_populates="owner", lazy="selectin")
    conversations = sa_relationship("Conversation", back_populates="user", lazy="selectin")



# =============================================================================
# Vehicle
# =============================================================================

class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    name = Column(String(255), nullable=True)  # User-given name
    vehicle_model = Column(String(100), nullable=True)
    vehicle_age = Column(Integer, nullable=True)
    fuel_type = Column(String(50), nullable=True)
    transmission_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = sa_relationship("User", back_populates="vehicles")
    features = sa_relationship("VehicleFeature", back_populates="vehicle", lazy="selectin")
    conversations = sa_relationship("Conversation", back_populates="vehicle", lazy="selectin")


# =============================================================================
# Conversation
# =============================================================================

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    vehicle_id = Column(PGUUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=True)
    title = Column(String(255), default="New Conversation")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Structured state
    feature_state = Column(JSON, default=dict)
    asked_questions = Column(JSON, default=list)  # Features already asked about
    declined_features = Column(JSON, default=list)  # Features user said "I don't know"

    # Relationships
    user = sa_relationship("User", back_populates="conversations")
    vehicle = sa_relationship("Vehicle", back_populates="conversations")
    messages = sa_relationship("Message", back_populates="conversation", 
                          lazy="selectin", order_by="Message.created_at")
    predictions = sa_relationship("Prediction", back_populates="conversation", lazy="selectin")
    summaries = sa_relationship("ConversationSummary", back_populates="conversation", lazy="selectin")

    __table_args__ = (
        Index("idx_conversation_user", "user_id"),
        Index("idx_conversation_updated", "updated_at"),
    )


# =============================================================================
# Message
# =============================================================================

class Message(Base):
    __tablename__ = "messages"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(SAEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)  # Extra info (prediction cards, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = sa_relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("idx_message_conversation", "conversation_id"),
        Index("idx_message_created", "created_at"),
    )


# =============================================================================
# Vehicle Features (Structured Feature State)
# =============================================================================

class VehicleFeature(Base):
    __tablename__ = "vehicle_features"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vehicle_id = Column(PGUUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=True)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(String(255), nullable=True)
    status = Column(SAEnum(FeatureStatus), default=FeatureStatus.UNKNOWN)
    source = Column(SAEnum(FeatureSource), default=FeatureSource.USER)
    confidence = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    vehicle = sa_relationship("Vehicle", back_populates="features")

    __table_args__ = (
        Index("idx_vf_vehicle", "vehicle_id"),
        Index("idx_vf_conversation", "conversation_id"),
    )


# =============================================================================
# Feature Provenance
# =============================================================================

class FeatureProvenance(Base):
    __tablename__ = "feature_provenance"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    old_value = Column(String(255), nullable=True)
    new_value = Column(String(255), nullable=True)
    old_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    source = Column(SAEnum(FeatureSource), nullable=False)
    message_id = Column(PGUUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Prediction
# =============================================================================

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    prediction_class = Column(Integer, nullable=False)  # 0 or 1
    model_score = Column(Float, nullable=False)
    features_used = Column(JSON, default=list)
    missing_features = Column(JSON, default=list)
    feature_state_snapshot = Column(JSON, default=dict)
    model_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = sa_relationship("Conversation", back_populates="predictions")

    __table_args__ = (
        Index("idx_prediction_conversation", "conversation_id"),
    )


# =============================================================================
# Conversation Summary (for context compaction)
# =============================================================================

class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    summary_text = Column(Text, nullable=False)
    messages_covered = Column(Integer, default=0)  # How many messages were compacted
    feature_state_at_summary = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = sa_relationship("Conversation", back_populates="summaries")


# =============================================================================
# Google User (OAuth Identity)
# =============================================================================

class GoogleUser(Base):
    __tablename__ = "google_users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_user_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(320), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    oauth_token = sa_relationship("OAuthToken", back_populates="google_user", uselist=False, lazy="selectin")
    emergency_contact = sa_relationship("EmergencyContact", back_populates="google_user", uselist=False, lazy="selectin")
    email_logs = sa_relationship("EmailLog", back_populates="google_user", lazy="selectin")


# =============================================================================
# OAuth Token Storage (Encrypted at Rest)
# =============================================================================

class OAuthToken(Base):
    __tablename__ = "oauth_tokens"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_user_id = Column(String(255), ForeignKey("google_users.google_user_id"), unique=True, nullable=False)
    access_token_encrypted = Column(Text, nullable=False)
    refresh_token_encrypted = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    scopes = Column(Text, nullable=True)  # Space-separated scopes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    google_user = sa_relationship("GoogleUser", back_populates="oauth_token")

    __table_args__ = (
        Index("idx_oauth_token_google_user", "google_user_id"),
    )


# =============================================================================
# Emergency Contact
# =============================================================================

class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_user_id = Column(String(255), ForeignKey("google_users.google_user_id"), unique=True, nullable=False)
    contact_name = Column(String(255), nullable=False)
    relationship = Column(String(100), nullable=True)
    contact_email = Column(String(320), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    google_user = sa_relationship("GoogleUser", back_populates="emergency_contact")

    __table_args__ = (
        Index("idx_emergency_contact_google_user", "google_user_id"),
    )


# =============================================================================
# Email Audit Log
# =============================================================================

class EmailLogStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class EmailLog(Base):
    __tablename__ = "email_logs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_user_id = Column(String(255), ForeignKey("google_users.google_user_id"), nullable=False)
    contact_email = Column(String(320), nullable=False)
    subject = Column(String(500), nullable=False)
    status = Column(SAEnum(EmailLogStatus), default=EmailLogStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)
    prediction_summary = Column(Text, nullable=True)
    prediction = Column(Float, nullable=True)  # ML model_score at time of send
    email_id = Column(String(255), nullable=True)  # Gmail message ID
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    google_user = sa_relationship("GoogleUser", back_populates="email_logs")

    __table_args__ = (
        Index("idx_email_log_google_user", "google_user_id"),
        Index("idx_email_log_created", "created_at"),
    )

