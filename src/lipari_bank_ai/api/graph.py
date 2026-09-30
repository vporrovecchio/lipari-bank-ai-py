from decimal import Decimal
from functools import cache
from typing import Annotated

import instructor
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.auth.deps import UserContext, get_current_user
from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.graph_rag.estrazione import Estrattore
from lipari_bank_ai.graph_rag.repository import GrafoRepository
from lipari_bank_ai.graph_rag.service import GraphAdviceResponse, GraphRagService
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_embedder, get_instructor, get_llm_provider
from lipari_bank_ai.llm.prompt import load_prompt
from observability.ledger import CostLedger
from lipari_bank_ai.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/ai", tags=["Graph"])


class GraphAdviceRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@cache
def prompt_risposta() -> str:
    return load_prompt("grafo_risposta_v1")


def get_estrattore(
    client: Annotated[instructor.AsyncInstructor, Depends(get_instructor)],
) -> Estrattore:
    return Estrattore(
        client,
        settings.default_model,
        load_prompt("estrazione_grafo_v1"),
        load_prompt("grafo_domanda_v1"),
    )


def get_graph_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    embedder: Annotated[EmbeddingClient, Depends(get_embedder)],
    estrattore: Annotated[Estrattore, Depends(get_estrattore)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> GraphRagService:
    return GraphRagService(
        GrafoRepository(session),
        RetrievalService(session),
        embedder,
        estrattore,
        llm,
        prompt_risposta(),
    )


@router.post("/graph-advice", response_model=GraphAdviceResponse)
async def graph_advice(
    req: GraphAdviceRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    service: Annotated[GraphRagService, Depends(get_graph_service)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> GraphAdviceResponse:
    risposta = await service.answer(req.question, user)
    CostLedger(session).aggiungi(  # il registro del Giorno 9: anche il grafo costa
        endpoint="graph",
        username=user.username,
        model=settings.default_model,
        cost_eur=Decimal(str(risposta.cost_eur)),
    )
    await session.commit()
    return risposta