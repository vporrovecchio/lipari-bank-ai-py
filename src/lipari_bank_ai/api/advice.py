from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.services.ingest_service import IngestService
from lipari_bank_ai.services.rag_service import RAGService
from lipari_bank_ai.services.retrieval_service import RetrievalService
from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse, IngestRequest, IngestResponse

router = APIRouter(prefix="/api/ai", tags=["Advice"])


@router.post("/advice", response_model=AdviceResponse)
async def advice(req: AdviceRequest, db: AsyncSession = Depends(get_db)) -> AdviceResponse:
    embedding_client = EmbeddingClient()
    retrieval = RetrievalService(db, embedding_client)
    rag = RAGService(retrieval, get_llm_provider())
    return await rag.answer(req)


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, db: AsyncSession = Depends(get_db)) -> IngestResponse:
    embedding_client = EmbeddingClient()
    service = IngestService(db, embedding_client)
    count = await service.ingest_document(req.document_id, req.content, req.metadata)
    return IngestResponse(chunk_count=count, embedding_dim=settings.embedding_dim)