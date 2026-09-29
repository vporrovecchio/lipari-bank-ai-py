from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.auth.deps import UserContext, get_current_user
from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.services.ingest_service import IngestService
from lipari_bank_ai.services.rag_service import RAGService
from lipari_bank_ai.services.retrieval_service import RetrievalService
from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse, IngestRequest, IngestResponse
from observability.ledger import CostLedger

router = APIRouter(prefix="/api/ai", tags=["Advice"])


def get_rag_service(db: Annotated[AsyncSession, Depends(get_db)]) -> RAGService:
    retrieval = RetrievalService(db, EmbeddingClient())
    return RAGService(retrieval, get_llm_provider())


@router.post("/advice", response_model=AdviceResponse)
async def advice(
    req: AdviceRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[RAGService, Depends(get_rag_service)],
    session: Annotated[AsyncSession, Depends(get_db)],  # la stessa sessione del servizio
) -> AdviceResponse:
    risposta = await service.advice(req.question, user)
    # Giorno 9: la generazione nel registro. La riscrittura qui non c'è: vedi il Code Blueprint
    CostLedger(session).aggiungi(
        endpoint="advice",
        username=user.username,
        model=settings.default_model,
        tokens=risposta.tokens_used,
        cost_eur=Decimal(str(risposta.cost_eur)),
    )
    await session.commit()
    return risposta


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, db: AsyncSession = Depends(get_db)) -> IngestResponse:
    embedding_client = EmbeddingClient()
    service = IngestService(db, embedding_client)
    count = await service.ingest_document(req.document_id, req.content, req.metadata, req.visibility)
    return IngestResponse(chunk_count=count, embedding_dim=settings.embedding_dim)