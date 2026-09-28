from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from lipari_bank_ai.auth.acl import visible_to

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

        stmt = text("""
            SELECT id, document_id, content, chunk_metadata, similarity
            FROM (
                SELECT id, document_id, chunk_index, content, chunk_metadata,
                    1 - (embedding <=> :query_emb) AS similarity,
                    row_number() OVER (
                        PARTITION BY document_id, chunk_index
                        ORDER BY 1 - (embedding <=> :query_emb) DESC
                    ) AS rn
                FROM document_chunks
            ) dedup
            WHERE rn = 1
            ORDER BY similarity DESC
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
    
    async def search_for_user(
        self, query_vec: list[float], role: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """I passaggi più vicini FRA QUELLI che questo ruolo può vedere."""

        # SQL raw per pgvector operator
        # Il dedup (row_number) va DENTRO il filtro di visibilità, altrimenti una copia
        # non visibile potrebbe "vincere" la partizione e far scartare quella visibile.
        stmt = text("""
            SELECT id, document_id, content, chunk_metadata, similarity
            FROM (
                SELECT id, document_id, chunk_index, content, chunk_metadata,
                       1 - (embedding <=> CAST(:q AS vector)) AS similarity,
                       row_number() OVER (
                           PARTITION BY document_id, chunk_index
                           ORDER BY 1 - (embedding <=> CAST(:q AS vector)) DESC
                       ) AS rn
                FROM document_chunks
                WHERE 1 - (embedding <=> CAST(:q AS vector)) >= :soglia
                    AND visibility = ANY(:livelli)
            ) dedup
            WHERE rn = 1
            ORDER BY similarity DESC
            LIMIT :k
            """)

        righe = await self.session.execute(
            stmt,
            {"q": str(query_vec), "k": top_k, "soglia": 0.1,
            "livelli": visible_to(role)},
        )

        return [
            RetrievalResult(
                chunk_id=str(r.id),
                document_id=r.document_id,
                content=r.content,
                similarity=float(r.similarity),
                metadata=r.chunk_metadata
            )
            for r in righe
        ]