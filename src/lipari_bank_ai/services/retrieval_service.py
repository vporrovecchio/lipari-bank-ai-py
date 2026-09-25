from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.llm.embedding_client import EmbeddingClient


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    metadata: dict[str, Any]


class RetrievalService:
    def __init__(self, session: AsyncSession, embedding_client: EmbeddingClient) -> None:
        self.session = session
        self.embedding_client = embedding_client

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = await self.embedding_client.embed_one(query)

        # SQL raw per pgvector operator
        stmt = text("""
            SELECT id, document_id, content, chunk_metadata,
                   1 - (embedding <=> :query_emb) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> :query_emb
            LIMIT :top_k
        """)

        result = await self.session.execute(stmt, {
            "query_emb": str(query_embedding),
            "top_k": top_k
        })
        rows = result.fetchall()

        return [
            RetrievalResult(
                chunk_id=row.id,
                document_id=row.document_id,
                content=row.content,
                similarity=row.similarity,
                metadata=row.chunk_metadata,
            )
            for row in rows
        ]