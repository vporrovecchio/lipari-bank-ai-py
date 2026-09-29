from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.repos import AccountRepository, MovementRepository
from lipari_bank_ai.db.runs import RunRepository
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_openai
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


def crea_deps(
    db: AsyncSession,
    *,
    embedder: EmbeddingClient | None = None,
    openai: AsyncOpenAI | None = None,
) -> Deps:
    """I servizi su una sessione. `embedder` e `openai` si possono sostituire.

    La produzione non li passa e prende i client condivisi; l'eval passa i suoi,
    perché deve parlare con lo stesso modello del resto della misurazione.
    """
    emb = embedder or EmbeddingClient()
    return Deps(
        accounts=AccountRepository(db),
        movements=MovementRepository(db),
        alerts=AlertService(db),
        runs=RunRepository(db),
        retrieval=RetrievalService(db, embedding_client=emb),
        embedder=emb,
        openai=openai or get_openai(),
        model=settings.agent_model,
    )


def get_deps(db: Annotated[AsyncSession, Depends(get_db)]) -> Deps:
    return crea_deps(db)
