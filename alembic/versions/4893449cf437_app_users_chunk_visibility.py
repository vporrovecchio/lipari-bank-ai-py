"""app_users + chunk visibility

Revision ID: 4893449cf437
Revises: 2bcc00a5c258
"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "4893449cf437"
down_revision: str | Sequence[str] | None = "2bcc00a5c258"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- app_users ---
    op.create_table(
        "app_users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("full_name", sa.String(length=128), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_app_users_username"), "app_users", ["username"], unique=True)

    # --- timestamp -> timestamptz (valori esistenti interpretati come UTC) ---
    op.alter_column(
        "chat_messages", "created_at",
        existing_type=sa.DateTime(), type_=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "chat_sessions", "started_at",
        existing_type=sa.DateTime(), type_=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="started_at AT TIME ZONE 'UTC'",
    )

    # --- document_chunks ---
    # Gli embedding a 1536 dim non sono convertibili a 768: vanno rigenerati.
    op.execute("TRUNCATE TABLE document_chunks")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")

    op.add_column(
        "document_chunks",
        sa.Column("visibility", sa.String(length=32), nullable=False, server_default="public"),
    )
    op.alter_column("document_chunks", "visibility", server_default=None)
    op.create_index("ix_document_chunks_visibility", "document_chunks", ["visibility"])

    op.alter_column(
        "document_chunks", "embedding",
        existing_type=Vector(1536), type_=Vector(768), nullable=False,
    )
    op.alter_column(
        "document_chunks", "chunk_metadata",
        existing_type=sa.JSON(), nullable=False,
    )

    op.create_index(
        "ix_document_chunks_embedding", "document_chunks", ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.execute("TRUNCATE TABLE document_chunks")
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")

    op.alter_column("document_chunks", "chunk_metadata", existing_type=sa.JSON(), nullable=True)
    op.alter_column(
        "document_chunks", "embedding",
        existing_type=Vector(768), type_=Vector(1536), nullable=True,
    )
    op.drop_index("ix_document_chunks_visibility", table_name="document_chunks")
    op.drop_column("document_chunks", "visibility")

    op.create_index(
        "ix_document_chunks_embedding", "document_chunks", ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.alter_column(
        "chat_sessions", "started_at",
        existing_type=sa.DateTime(timezone=True), type_=sa.DateTime(),
        existing_nullable=False,
    )
    op.alter_column(
        "chat_messages", "created_at",
        existing_type=sa.DateTime(timezone=True), type_=sa.DateTime(),
        existing_nullable=False,
    )
    op.drop_index(op.f("ix_app_users_username"), table_name="app_users")
    op.drop_table("app_users")