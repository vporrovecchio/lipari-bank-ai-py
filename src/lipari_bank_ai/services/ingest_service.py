from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.models import DocumentChunk
from lipari_bank_ai.lib.chunking import chunk_text
from lipari_bank_ai.llm.embedding_client import EmbeddingClient


class IngestService:
    def __init__(self, session: AsyncSession, embedding_client: EmbeddingClient) -> None:
        self.session = session
        self.embedding_client = embedding_client

    async def ingest_document(
        self, document_id: str, content: str, metadata: dict[str, Any] | None = None,
    ) -> int:
        # Ho tutti documenti markdown con paragrafi separati da ## quindi non ho bisogno di overlap
        chunks = chunk_text(content, chunk_size=500, overlap=0)
        embeddings = await self.embedding_client.embed(chunks)

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            db_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=idx,
                content=chunk,
                embedding=embedding,
                chunk_metadata=metadata or {},
            )
            self.session.add(db_chunk)

        await self.session.commit()
        return len(chunks)