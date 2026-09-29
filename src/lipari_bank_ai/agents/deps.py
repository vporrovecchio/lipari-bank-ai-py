from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.repos import AccountRepository, MovementRepository
from lipari_bank_ai.db.runs import RunRepository
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.services.alerts import AlertService
from lipari_bank_ai.services.retrieval_service import RetrievalService


@dataclass(frozen=True)
class Deps:
    accounts: AccountRepository
    movements: MovementRepository
    alerts: AlertService
    runs: RunRepository
    retrieval: RetrievalService
    embedder: EmbeddingClient
    openai: AsyncOpenAI
    model: str


@lru_cache
def _openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="ollama", base_url=f"{settings.ollama_url}/v1", timeout=120.0
    )


def get_deps(db: Annotated[AsyncSession, Depends(get_db)]) -> Deps:
    return Deps(
        accounts=AccountRepository(db),
        movements=MovementRepository(db),
        alerts=AlertService(db),
        runs=RunRepository(db),
        retrieval=RetrievalService(db, embedding_client=EmbeddingClient()),
        embedder=EmbeddingClient(),
        openai=_openai_client(),
        model=settings.agent_model,
    )