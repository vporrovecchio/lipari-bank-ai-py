from decimal import Decimal
import uuid
from datetime import UTC, date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())

JSON_O_JSONB = JSON().with_variant(JSONB(), "postgresql")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String, comment="'system' | 'user' | 'assistant' | 'tool'")
    content: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_eur: Mapped[float] = mapped_column(default=0.0)
    model_used: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    document_id: Mapped[str] = mapped_column(String, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    visibility: Mapped[str] = mapped_column(String(32), default="public", index=True)

class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="operator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)     # il codice cliente
    full_name: Mapped[str] = mapped_column(String(120))
    operator: Mapped[str] = mapped_column(String(64), index=True)     # chi lo ha in portafoglio
    accounts: Mapped[list["Account"]] = relationship(back_populates="customer")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(34), primary_key=True)     # l'IBAN
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    label: Mapped[str] = mapped_column(String(64))                    # "principale", "risparmio"
    balance: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    customer: Mapped["Customer"] = relationship(back_populates="accounts")
    movements: Mapped[list["Movement"]] = relationship(back_populates="account")


class Movement(Base):
    __tablename__ = "movements"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    booking_date: Mapped[date] = mapped_column(Date)       # la data contabile
    description: Mapped[str] = mapped_column(String(200))
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))  
    account: Mapped["Account"] = relationship(back_populates="movements")          # negativo = uscita


class ComplianceAlert(Base):
    __tablename__ = "compliance_alerts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    opened_by: Mapped[str] = mapped_column(String(64))                 # username dal token
    reason: Mapped[str] = mapped_column(Text)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

class AgentRunState(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)      # il run_id del Giorno 7
    username: Mapped[str] = mapped_column(String(64), index=True)      # chi ha chiesto
    role: Mapped[str] = mapped_column(String(32))                      # i tool si rifanno per lui
    status: Mapped[str] = mapped_column(String(32), index=True)
    # awaiting_approval | running | done | rejected
    messages: Mapped[list[dict[str, Any]]] = mapped_column(JSON_O_JSONB)   # ← lo stato
    pending_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON_O_JSONB)
    description: Mapped[str] = mapped_column(Text)                     # cosa si sta approvando
    steps: Mapped[int] = mapped_column(Integer)
    cost_eur: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    tool_calls: Mapped[list[str]] = mapped_column(JSON_O_JSONB)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    endpoint: Mapped[str] = mapped_column(String(32), index=True)  # "chat", "advice", "agent"
    username: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(64))
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_eur: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
)