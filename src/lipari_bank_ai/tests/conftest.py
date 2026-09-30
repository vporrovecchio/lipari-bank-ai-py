from collections.abc import AsyncIterator, Iterator
from datetime import date
from decimal import Decimal
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from lipari_bank_ai.agents.deps import Deps
from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.db.models import Account, Customer, Movement
from lipari_bank_ai.db.repos import AccountRepository, MovementRepository
from lipari_bank_ai.db.runs import RunRepository
from lipari_bank_ai.db.session import Base
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.services.alerts import AlertService
from lipari_bank_ai.services.retrieval_service import RetrievalService

os.environ["OPENAI_API_KEY"] = "sk-test-mai-valida"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-mai-valida"

CLIENTE_DI_MARCO = "C-10234"           # nel portafoglio di mbianchi
CLIENTE_ALTRUI = "C-20417"             # nel portafoglio di un collega, pgalli
CONTO_DI_MARCO = "IT60X0542811101000000123"
CONTO_ALTRUI = "IT60X0542811101000000789"
TABELLE_DEL_GIORNO = ["customers", "accounts", "movements", "compliance_alerts", "agent_runs", "llm_calls"]


def risposta(*, tool: str | None = None, argomenti: str = "{}", testo: str | None = None,
             prompt_tokens: int = 10) -> ChatCompletion:
    """Una risposta del modello: finta, ma con la forma vera dell'SDK."""
    messaggio: dict[str, Any] = {"role": "assistant", "content": testo}
    if tool:
        messaggio["tool_calls"] = [{"id": "call_1", "type": "function",
                                    "function": {"name": tool, "arguments": argomenti}}]
    return ChatCompletion.model_validate({
        "id": "finto", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": messaggio,
                     "finish_reason": "tool_calls" if tool else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                  "total_tokens": prompt_tokens + 5},
    })


@pytest.fixture
def marco() -> UserContext:
    return UserContext(username="mbianchi", role="operator")


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        # solo le tabelle di oggi: quelle dei giorni precedenti usano tipi di PostgreSQL
        await conn.run_sync(lambda c: Base.metadata.create_all(
            c, tables=[Base.metadata.tables[nome] for nome in TABELLE_DEL_GIORNO]))
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        s.add_all([
            Customer(id=CLIENTE_DI_MARCO, full_name="Paolo Ferri", operator="mbianchi"),
            Customer(id=CLIENTE_ALTRUI, full_name="Anna Greco", operator="pgalli"),
            Account(id=CONTO_DI_MARCO, customer_id=CLIENTE_DI_MARCO, label="principale",
                    balance=Decimal("48200.00")),
            Account(id=CONTO_ALTRUI, customer_id=CLIENTE_ALTRUI, label="principale",
                    balance=Decimal("15300.00")),
            Movement(account_id=CONTO_DI_MARCO, booking_date=date(2026, 9, 20),
                     description="Bonifico estero", amount=Decimal("-9800.00")),
        ])
        await s.commit()
        yield s
    await engine.dispose()


@pytest.fixture
def deps_reali(session: AsyncSession) -> Deps:
    return Deps(accounts=AccountRepository(session), movements=MovementRepository(session),
                alerts=AlertService(session), retrieval=RetrievalService(session, embedding_client=EmbeddingClient()),
                embedder=EmbeddingClient(), openai=AsyncMock(), model="gpt-4o-mini", runs=RunRepository(session))


@pytest.fixture
def deps_spia() -> MagicMock:
    deps = MagicMock()
    deps.accounts.of_user = AsyncMock(return_value=None)
    deps.accounts.of_customer = AsyncMock(return_value=[])
    deps.movements.recent = AsyncMock(return_value=[])
    deps.alerts.apri = AsyncMock()
    return deps

@pytest.fixture(scope="session", autouse=True)
def modelli_finti() -> Iterator[None]:
    """Dal Giorno 4 nessun test chiama un modello vero: costerebbe, e risponderebbe ogni volta
    in un modo diverso. La chat parla con un finto che ripete la domanda, la categorizzazione
    usa la regola del Giorno 2. Un test che vuole un modello preciso mette il suo."""
    from lipari_bank_ai.api.categorize import get_categorize_service
    from lipari_bank_ai.llm.factory import get_llm_provider
    from lipari_bank_ai.main import app
    from lipari_bank_ai.tests.finti import CategorizzazioneARegole, ModelloEco

    app.dependency_overrides[get_llm_provider] = ModelloEco
    app.dependency_overrides[get_categorize_service] = CategorizzazioneARegole
    yield
    app.dependency_overrides.clear()