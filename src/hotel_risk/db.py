from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Iterator

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import get_settings
from .domain import BatchStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class HotelPropertyORM(Base):
    __tablename__ = "hotel_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    region: Mapped[str] = mapped_column(String(100), default="demo")
    currency: Mapped[str] = mapped_column(String(16), default="EUR")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    batches: Mapped[list["ScoringBatchORM"]] = relationship(back_populates="property")


class ScoringBatchORM(Base):
    __tablename__ = "scoring_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    property_id: Mapped[int | None] = mapped_column(ForeignKey("hotel_properties.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(250), index=True)
    source: Mapped[str] = mapped_column(String(100), default="csv")
    file_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default=BatchStatus.CREATED.value, index=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    high_risk_count: Mapped[int] = mapped_column(Integer, default=0)
    total_booking_value: Mapped[float] = mapped_column(Float, default=0.0)
    total_expected_loss: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    property: Mapped[HotelPropertyORM | None] = relationship(back_populates="batches")
    bookings: Mapped[list["BookingORM"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class BookingORM(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("scoring_batches.id"), index=True)
    external_booking_id: Mapped[str] = mapped_column(String(128), index=True)
    hotel: Mapped[str] = mapped_column(String(100), index=True)
    arrival_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    market_segment: Mapped[str] = mapped_column(String(100), index=True)
    distribution_channel: Mapped[str] = mapped_column(String(100), index=True)
    deposit_type: Mapped[str] = mapped_column(String(100), index=True)
    customer_type: Mapped[str] = mapped_column(String(100), index=True)
    adr: Mapped[float] = mapped_column(Float, default=0.0)
    total_nights: Mapped[int] = mapped_column(Integer, default=1)
    booking_value: Mapped[float] = mapped_column(Float, default=0.0)
    raw_json: Mapped[str] = mapped_column(Text)

    batch: Mapped[ScoringBatchORM] = relationship(back_populates="bookings")
    prediction: Mapped["PredictionORM"] = relationship(back_populates="booking", cascade="all, delete-orphan", uselist=False)


class PredictionORM(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"), unique=True, index=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    cancellation_probability: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    risk_category: Mapped[str] = mapped_column(String(32), index=True)
    expected_loss: Mapped[float] = mapped_column(Float, index=True)
    business_priority_score: Mapped[float] = mapped_column(Float, index=True)
    recommended_action: Mapped[str] = mapped_column(Text)
    top_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    booking: Mapped[BookingORM] = relationship(back_populates="prediction")


class ScoringJobORM(Base):
    __tablename__ = "scoring_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    queue_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default=BatchStatus.CREATED.value, index=True)
    worker_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuditLogORM(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    entity_type: Mapped[str] = mapped_column(String(100))
    entity_id: Mapped[str] = mapped_column(String(100))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


def make_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    connect_args = {}
    kwargs = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        if url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(url, connect_args=connect_args, future=True, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    init_db()
    with SessionLocal() as session:
        yield session


def audit(session: Session, event_type: str, entity_type: str, entity_id: str | int, payload: dict | None = None) -> None:
    session.add(AuditLogORM(event_type=event_type, entity_type=entity_type, entity_id=str(entity_id), payload_json=json.dumps(payload or {}, ensure_ascii=False)))


def batch_to_dict(batch: ScoringBatchORM) -> dict:
    return {
        "id": batch.id,
        "name": batch.name,
        "source": batch.source,
        "file_name": batch.file_name,
        "status": batch.status,
        "row_count": batch.row_count,
        "high_risk_count": batch.high_risk_count,
        "total_booking_value": batch.total_booking_value,
        "total_expected_loss": batch.total_expected_loss,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "updated_at": batch.updated_at.isoformat() if batch.updated_at else None,
        "error_message": batch.error_message,
    }
