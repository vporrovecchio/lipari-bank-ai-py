from collections.abc import AsyncIterator, Iterator
from decimal import Decimal

import fakeredis
import httpx2
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

import lipari_bank_ai.main
from lipari_bank_ai.scripts import misura_tetti, seed_all
from lipari_bank_ai.api.advice import get_rewriter
from lipari_bank_ai.db.models import AppUser, Customer, DocumentChunk
from lipari_bank_ai.db.session import AsyncSessionLocal, engine
from lipari_bank_ai.llm import factory
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.embeddings import CachedEmbedder
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.llm.rewriter import QueryRewriter
from lipari_bank_ai.main import app
from lipari_bank_ai.services.ingest_service import IngestService
from lipari_bank_ai.tests.finti import Registro, openai_finto

PUBBLICO = "# Bonifici estero\nI bonifici verso il Venezuela sono extra-SEPA: costo €15.00."


@pytest.fixture(autouse=True)
async def database_pulito() -> AsyncIterator[None]:
    await engine.dispose()
    async with engine.begin() as c:
        await c.execute(
            text(
                "TRUNCATE llm_calls, document_chunks, agent_runs, compliance_alerts, movements, "
                "accounts, customers, app_users, chat_messages, chat_sessions CASCADE"
            )
        )
    yield
    # anche all'uscita: in ordine alfabetico test_g10 gira prima di test_g2, che usa il
    # TestClient con un suo loop, e troverebbe nel pool connessioni aperte da questo
    await engine.dispose()


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()  # un Redis in memoria, che si può anche spegnere


def redis(server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server)


def embedder_contato() -> tuple[EmbeddingClient, Registro]:
    reg = Registro()

    def gestore(r: httpx2.Request) -> httpx2.Response:
        reg.annota(r)
        return gestore_embedding(r)

    return EmbeddingClient(openai_finto(gestore), "text-embedding-3-small"), reg


# ---------------------------------------------------------------- la cache degli embedding
async def test_la_cache_degli_embedding_chiede_solo_i_mancanti(
    server: fakeredis.FakeServer,
) -> None:
    base, reg = embedder_contato()
    emb = CachedEmbedder(base, redis(server))
    primi = await emb.embed(["uno", "due"])
    secondi = await emb.embed(["due", "tre", "uno"])  # due già noti, uno nuovo
    assert secondi[0] == primi[1] and secondi[2] == primi[0]
    assert [r["input"] for r in reg.richieste] == [["uno", "due"], ["tre"]]  # un lotto a giro
    assert (emb.hit, emb.miss) == (2, 3)


async def test_un_altro_worker_trova_i_vettori_del_primo(server: fakeredis.FakeServer) -> None:
    base, reg = embedder_contato()
    await CachedEmbedder(base, redis(server)).embed_one("bonifico estero")
    altro = CachedEmbedder(base, redis(server))  # un altro processo, lo stesso Redis
    await altro.embed_one("bonifico estero")
    assert len(reg.richieste) == 1 and altro.hit == 1


async def test_senza_redis_si_paga_ma_si_risponde(server: fakeredis.FakeServer) -> None:
    base, reg = embedder_contato()
    emb = CachedEmbedder(base, redis(server))
    server.connected = False  # Redis giù
    vettore = await emb.embed_one("bonifico estero")
    assert len(vettore) == 1536 and len(reg.richieste) == 1


def test_la_factory_mette_la_cache_solo_se_c_e_redis(
    monkeypatch: pytest.MonkeyPatch, server: fakeredis.FakeServer
) -> None:
    assert type(factory.get_embedder()) is EmbeddingClient  # nei test REDIS_URL è vuota
    monkeypatch.setattr(factory, "get_redis", lambda: redis(server))
    emb = factory.get_embedder()
    assert isinstance(emb, CachedEmbedder) and isinstance(emb, EmbeddingClient)

# ---------------------------------------------------------------- le due sonde
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_pronto_con_il_database_e_la_cache_spenta() -> None:
    async with client() as c:
        r = await c.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"database": "ok", "cache": "spenta"}}


async def test_senza_database_vivo_ma_non_pronto_e_senza_topologia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # un database che non c'è: la porta 1 non risponde, e l'URL ha una password riconoscibile
    spento = create_async_engine("postgresql+asyncpg://lipari:segreta-42@127.0.0.1:1/lipari_ai")
    monkeypatch.setattr(lipari_bank_ai.main, "engine", spento)
    async with client() as c:
        pronto = await c.get("/ready")
        vivo = await c.get("/health")
    assert pronto.status_code == 503 and pronto.json()["checks"]["database"] == "ko"
    assert "segreta-42" not in pronto.text and "127.0.0.1" not in pronto.text  # niente mappa
    assert vivo.status_code == 200  # la liveness non tocca il database: è viva
    await spento.dispose()


async def test_la_cache_giu_non_toglie_l_istanza_dal_traffico(
    monkeypatch: pytest.MonkeyPatch, server: fakeredis.FakeServer
) -> None:
    server.connected = False
    monkeypatch.setattr(src.main, "get_redis", lambda: redis(server))
    async with client() as c:
        r = await c.get("/ready")
    assert r.status_code == 200 and r.json()["checks"] == {"database": "ok", "cache": "ko"}


# ---------------------------------------------------------------- i tetti
def test_i_tetti_si_rispettano_con_un_numero_senza_peggiorare_l_altro() -> None:
    partenza = {"p95_s": 4.0, "costo_medio_eur": 0.0010}
    verdetto = misura_tetti.verdetto
    assert verdetto(partenza, {"p95_s": 2.9, "costo_medio_eur": 0.0010})[0]  # tempo
    assert verdetto(partenza, {"p95_s": 4.0, "costo_medio_eur": 0.0007})[0]  # costo
    assert not verdetto(partenza, {"p95_s": 2.9, "costo_medio_eur": 0.0011})[0]  # costo su
    assert not verdetto(partenza, {"p95_s": 3.5, "costo_medio_eur": 0.0009})[0]  # poco
    assert misura_tetti.p95([1.0] * 19 + [10.0]) < 10.0  # un caso su venti non è il p95