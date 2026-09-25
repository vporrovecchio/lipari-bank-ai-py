"""add pgvector

Revision ID: 2bcc00a5c258
Revises: 4ac18f3b2bbd
Create Date: 2026-09-14 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "2bcc00a5c258"
down_revision: str | Sequence[str] | None = "4ac18f3b2bbd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("chunk_metadata", sa.JSON(), default={}),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.execute("DROP EXTENSION IF EXISTS vector")