import asyncio
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from typing import Any

import httpx2
import instructor
import pytest
from httpx import ASGITransport, AsyncClient
from openai.types import CompletionUsage
from sqlalchemy import func, select, text

from scripts import estrai_grafo
from lipari_bank_ai.agents.loop import costo_chiamata
from lipari_bank_ai.api.graph import get_estrattore
from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.db.models import GraphEdge, GraphExtraction, GraphNode, GraphReview, LlmCall
from lipari_bank_ai.db.session import AsyncSessionLocal, Base, engine
from lipari_bank_ai.graph_rag.estrazione import Estrattore, verificabili
from lipari_bank_ai.graph_rag.modelli import (
    Entita,
    EntitaNominate,
    GrafoEstratto,
    Relazione,
    TipoEntita,
    TipoRelazione,
)
from lipari_bank_ai.graph_rag.repository import MAX_HOP, Arco, GrafoRepository
from lipari_bank_ai.graph_rag.resolution import candidati, chiave, normalizza
from lipari_bank_ai.graph_rag.service import GraphAdviceResponse, GraphRagService
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.main import app
from lipari_bank_ai.services.ingest_service import IngestService
from lipari_bank_ai.services.retrieval_service import RetrievalResult, RetrievalService
from lipari_bank_ai.tests.finti import (
    GIULIA,
    MARCO,
    ModelloFisso,
    Registro,
    chiamata,
    completion,
    embedder_finto,
    openai_finto,
)

S014 = """# Segnalazione S-2026-014

La segnalazione S-2026-014 è stata aperta dall'operatore Marco Bianchi.
La segnalazione S-2026-014 riguarda il conto IT60X0542811101000000123.
Il conto IT60X0542811101000000123 è intestato al cliente C-10234, Paolo Ferri.
La segnalazione S-2026-014 riguarda due bonifici verso Delta Trading Ltd, per 19.300 euro."""
CIRCOLARE = """# Circolare 7/2026

La circolare 7/2026 elenca Delta Trading Ltd, con sede a Panama."""


@pytest.fixture(autouse=True)
async def grafo_vuoto() -> AsyncIterator[None]:
    await engine.dispose()
    async with engine.begin() as c:
        await c.execute(
            text(
                "TRUNCATE graph_edges, graph_nodes, graph_revisione, graph_estratti, "
                "document_chunks, llm_calls CASCADE"
            )
        )
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
def override_ripristinati() -> Iterator[None]:
    prima = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(prima)


def e(tipo: TipoEntita, menzione: str, identificativo: str | None = None) -> Entita:
    return Entita(tipo=tipo, menzione=menzione, identificativo=identificativo)


def r(da: str, tipo: TipoRelazione, a: str, citazione: str = "frase") -> Relazione:
    return Relazione(da=da, tipo=tipo, a=a, citazione=citazione)


SEGNALAZIONE = e("Segnalazione", "S-2026-014", "S-2026-014")
DELTA = e("Controparte", "Delta Trading Ltd")
CIRC = e("Circolare", "circolare 7/2026", "7/2026")


# ---------------------------------------------------------------- l'identità, senza database
def test_si_unisce_da_solo_solo_cio_che_una_regola_meccanica_rende_uguale() -> None:
    assert normalizza("Inversiones Caribe S.A.") == normalizza("INVERSIONES CARIBE SA")
    assert chiave(e("Controparte", "Inversiones Caribe S.A.")) == chiave(
        e("Controparte", "inversiones caribe sa")
    )
    # il codice è l'identità: con il codice, il nome non conta
    assert chiave(e("Cliente", "Paolo Ferri", "C-10234")) == "Cliente:C-10234"
    assert chiave(e("Cliente", "il sig. Ferri (C-10234)")) == "Cliente:C-10234"
    # simili non vuol dire uguali: due chiavi diverse
    assert chiave(e("Controparte", "Delta Trading Ltd")) != chiave(
        e("Controparte", "DELTA TRADING LIMITED")
    )


def test_un_nome_simile_diventa_un_caso_per_una_persona() -> None:
    esistenti = {"Controparte:delta trading ltd": "Delta Trading Ltd", "Cliente:C-10234": "Paolo"}
    coda = candidati("Controparte:delta trading limited", esistenti, "Controparte")
    assert [k for k, _ in coda] == ["Controparte:delta trading ltd"]
    assert 0.75 <= coda[0][1] < 1
    # chi ha un codice non si riconcilia per somiglianza: C-10234 e C-10243 sono due clienti
    assert candidati("Cliente:C-10243", {"Cliente:C-10234": "x"}, "Cliente") == []


def test_una_relazione_senza_la_sua_frase_nel_testo_si_scarta() -> None:
    estratto = GrafoEstratto(
        entita=[SEGNALAZIONE, DELTA],
        relazioni=[
            # la frase c'è, con un apostrofo e una maiuscola diversi: si tiene
            r(
                "S-2026-014",
                "VERSO",
                "Delta Trading Ltd",
                "la segnalazione S-2026-014 RIGUARDA due bonifici verso Delta Trading Ltd",
            ),
            # la frase è del modello, non del testo: si scarta
            r("S-2026-014", "VERSO", "Delta Trading Ltd", "S-2026-014 è collegata a Delta"),
        ],
    )
    tenute, scartate = verificabili(estratto, S014)
    assert len(tenute) == 1 and len(scartate) == 1


# ---------------------------------------------------------------- il grafo su Postgres
async def scrivi(
    entita: list[Entita], relazioni: list[Relazione], doc: str, livello: str = "internal"
) -> tuple[int, int]:
    async with AsyncSessionLocal() as s:
        esito = await GrafoRepository(s).upsert(entita, relazioni, doc, livello)
        await s.commit()
        return esito


async def conta(modello: type[Base]) -> int:
    async with AsyncSessionLocal() as s:
        return int(await s.scalar(select(func.count()).select_from(modello)) or 0)


async def espandi(chiavi: list[str], role: str, hop: int = MAX_HOP) -> list[Arco]:
    async with AsyncSessionLocal() as s:
        return await GrafoRepository(s).espandi(chiavi, role, hop)


async def test_lo_stesso_passaggio_due_volte_scrive_gli_stessi_archi() -> None:
    entita = [SEGNALAZIONE, DELTA]
    relazioni = [r("S-2026-014", "VERSO", "Delta Trading Ltd")]
    assert await scrivi(entita, relazioni, "segnalazione_S-2026-014") == (1, 0)
    assert await scrivi(entita, relazioni, "segnalazione_S-2026-014") == (0, 0)  # niente di nuovo
    assert await conta(GraphEdge) == 1 and await conta(GraphNode) == 2
    # un estremo che il modello non ha elencato fra le entità: l'arco non si scrive
    assert await scrivi(entita, [r("S-2026-014", "RIGUARDA", "IT99")], "x") == (0, 1)


async def test_il_tetto_dei_salti_si_abbassa_e_non_si_alza() -> None:
    # una catena di cinque nodi: quattro salti dal primo all'ultimo
    nodi = [e("Segnalazione", f"S-2026-10{i}", f"S-2026-10{i}") for i in range(5)]
    catena = [r(f"S-2026-10{i}", "RIGUARDA", f"S-2026-10{i + 1}") for i in range(4)]
    await scrivi(nodi, catena, "catena")
    tutti = await espandi(["Segnalazione:S-2026-100"], "operator", hop=10)
    assert max(a.distanza for a in tutti) == MAX_HOP == 3 and len(tutti) == 3
    uno = await espandi(["Segnalazione:S-2026-100"], "operator", hop=1)
    assert [(a.da, a.a) for a in uno] == [("Segnalazione:S-2026-100", "Segnalazione:S-2026-101")]


async def test_un_arco_che_il_ruolo_non_vede_non_si_attraversa() -> None:
    conto = e("Conto", "IT60X0542811101000000123", "IT60X0542811101000000123")
    await scrivi([CIRC, DELTA], [r("7/2026", "ELENCA", "Delta Trading Ltd")], "circolare")
    await scrivi(
        [SEGNALAZIONE, DELTA],
        [r("S-2026-014", "VERSO", "Delta Trading Ltd")],
        "s14",
        "compliance_only",
    )
    # un arco pubblico, raggiungibile SOLO passando dalla segnalazione riservata
    await scrivi(
        [SEGNALAZIONE, conto],
        [r("S-2026-014", "RIGUARDA", conto.menzione)],
        "pubblico",
        "public",
    )
    operatore = await espandi(["Circolare:7/2026"], "operator")
    compliance = await espandi(["Circolare:7/2026"], "compliance_lead")
    assert [a.document_id for a in operatore] == ["circolare"]  # nemmeno l'arco pubblico
    assert {a.document_id for a in compliance} == {"circolare", "s14", "pubblico"}
    assert await espandi(["Circolare:7/2026"], "ruolo_inventato") == []  # vede il solo pubblico


async def test_i_nomi_simili_vanno_in_coda_e_la_coppia_decisa_non_torna() -> None:
    await scrivi([SEGNALAZIONE, DELTA], [r("S-2026-014", "VERSO", DELTA.menzione)], "s14")
    limited = e("Controparte", "DELTA TRADING LIMITED")
    s21 = e("Segnalazione", "S-2026-021", "S-2026-021")
    await scrivi([s21, limited], [r("S-2026-021", "VERSO", limited.menzione)], "s21")
    assert await conta(GraphNode) == 4  # due controparti: nessuno le ha unite
    async with AsyncSessionLocal() as s:
        caso = (await s.scalars(select(GraphReview))).one()
    assert (caso.chiave, caso.candidata) == (
        "Controparte:delta trading limited",
        "Controparte:delta trading ltd",
    )
    # lo stesso nome, in un documento nuovo: la coppia è già in coda, anche letta al contrario
    await scrivi([CIRC, DELTA], [r("7/2026", "ELENCA", DELTA.menzione)], "circolare")
    assert await conta(GraphReview) == 1


async def test_unire_sposta_gli_archi_e_le_estrazioni_nuove_seguono() -> None:
    await scrivi([SEGNALAZIONE, DELTA], [r("S-2026-014", "VERSO", DELTA.menzione)], "s14")
    limited = e("Controparte", "DELTA TRADING LIMITED")
    s21 = e("Segnalazione", "S-2026-021", "S-2026-021")
    await scrivi([s21, limited], [r("S-2026-021", "VERSO", limited.menzione)], "s21")
    async with AsyncSessionLocal() as s:
        caso = (await s.scalars(select(GraphReview))).one()
        caso.stato = "unite"
        await GrafoRepository(s).unisci(caso.chiave, caso.candidata)
        await s.commit()
    archi = await espandi(["Controparte:delta trading ltd"], "operator", hop=1)
    assert {a.da for a in archi} == {"Segnalazione:S-2026-014", "Segnalazione:S-2026-021"}
    assert await espandi(["Controparte:delta trading limited"], "operator", hop=1) == archi
    # dal nodo vecchio non parte più niente: altrimenti lo stesso fatto si conterebbe due volte
    dalla_s21 = await espandi(["Segnalazione:S-2026-021"], "operator", hop=1)
    assert [a.a for a in dalla_s21] == ["Controparte:delta trading ltd"]
    # un documento nuovo che scrive LIMITED finisce sul nodo canonico
    s30 = e("Segnalazione", "S-2026-030", "S-2026-030")
    await scrivi([s30, limited], [r("S-2026-030", "VERSO", limited.menzione)], "s30")
    assert len(await espandi(["Controparte:delta trading ltd"], "operator", hop=1)) == 3


async def test_dimenticare_un_documento_toglie_solo_i_suoi_archi() -> None:
    await scrivi([SEGNALAZIONE, DELTA], [r("S-2026-014", "VERSO", DELTA.menzione)], "s14")
    await scrivi([CIRC, DELTA], [r("7/2026", "ELENCA", DELTA.menzione)], "circolare")
    async with AsyncSessionLocal() as s:
        await GrafoRepository(s).dimentica("s14")
        await s.commit()
    rimasti = await espandi(["Controparte:delta trading ltd"], "operator")
    assert [a.document_id for a in rimasti] == ["circolare"]


# ---------------------------------------------------------------- l'estrazione, dal comando
def estrazione_di(testo: str) -> dict[str, Any]:
    """Quello che un modello estrarrebbe dalla S-2026-014: due relazioni vere, una inventata."""
    if "S-2026-014" not in testo:
        return {"entita": [], "relazioni": []}
    relazioni = [
        {
            "da": "S-2026-014",
            "tipo": "APERTA_DA",
            "a": "Marco Bianchi",
            "citazione": "La segnalazione S-2026-014 è stata aperta dall'operatore Marco Bianchi",
        },
        {
            "da": "Marco Bianchi",
            "tipo": "VERSO",
            "a": "Delta Trading Ltd",
            "citazione": "Marco Bianchi ha disposto i bonifici verso Delta",
        },  # non è nel testo
    ]
    if "Delta Trading Ltd" in testo:
        relazioni.append(
            {
                "da": "S-2026-014",
                "tipo": "VERSO",
                "a": "Delta Trading Ltd",
                "citazione": "riguarda due bonifici verso Delta Trading Ltd",
            }
        )
    return {
        "entita": [
            {"tipo": "Segnalazione", "menzione": "S-2026-014", "identificativo": "S-2026-014"},
            {"tipo": "Operatore", "menzione": "Marco Bianchi", "identificativo": None},
            {"tipo": "Controparte", "menzione": "Delta Trading Ltd", "identificativo": None},
        ],
        "relazioni": relazioni,
    }


def instructor_finto(reg: Registro) -> instructor.AsyncInstructor:
    def gestore(richiesta: httpx2.Request) -> httpx2.Response:
        corpo = reg.annota(richiesta)
        argomenti = estrazione_di(corpo["messages"][-1]["content"])
        return completion(
            None, prompt=1000, uscita=200, tool_calls=[chiamata("c1", "GrafoEstratto", argomenti)]
        )

    return instructor.from_openai(openai_finto(gestore))


async def test_l_estrazione_scarta_l_inventato_e_non_ripaga_lo_stesso_testo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async with AsyncSessionLocal() as s:
        await IngestService(s, embedder_finto()).ingest(
            "segnalazione_S-2026-014", S014, None, "compliance_only"
        )
        await IngestService(s, embedder_finto()).ingest("commissioni_bonifico", "Costo €1.")
    reg = Registro()
    monkeypatch.setattr(estrai_grafo, "get_instructor", lambda: instructor_finto(reg))

    await estrai_grafo.main(estrai_grafo.PREFISSI)
    assert len(reg.richieste) == 1  # le commissioni non portano relazioni: non si leggono
    assert await conta(GraphEdge) == 2  # tre estratte, una scartata
    async with AsyncSessionLocal() as s:
        fatto = await s.get(GraphExtraction, "segnalazione_S-2026-014")
        livelli = set(await s.scalars(select(GraphEdge.visibility)))
    assert fatto is not None and (fatto.relazioni, fatto.scartate) == (2, 1)
    assert livelli == {"compliance_only"}  # l'arco eredita il livello del documento
    assert "33%" in capsys.readouterr().out  # la quota scartata, stampata

    await estrai_grafo.main(estrai_grafo.PREFISSI)
    assert len(reg.richieste) == 1  # lo stesso testo: niente seconda chiamata

    # il testo cambia e non nomina più la controparte: il suo arco non deve restare
    async with AsyncSessionLocal() as s:
        await IngestService(s, embedder_finto()).ingest(
            "segnalazione_S-2026-014",
            S014.replace("verso Delta Trading Ltd", "verso un conto estero"),
            None,
            "compliance_only",
        )
    await estrai_grafo.main(estrai_grafo.PREFISSI)
    assert len(reg.richieste) == 2
    async with AsyncSessionLocal() as s:
        tipi = list(await s.scalars(select(GraphEdge.tipo)))
    assert tipi == ["APERTA_DA"]


async def test_il_costo_dell_estrazione_conta_anche_i_tentativi_respinti() -> None:
    reg = Registro()
    estrattore = Estrattore(instructor_finto(reg), "gpt-4o-mini", "P", "PD")
    _, speso = await estrattore.estrai(S014)
    una = costo_chiamata(
        CompletionUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200), "gpt-4o-mini"
    )
    assert speso == una > 0

    # un tipo fuori dall'elenco chiuso: instructor rimanda l'errore al modello e ritenta.
    # Il tentativo respinto si è pagato, e il costo lo conta: l'usage è la somma dei tentativi
    tentativi = Registro()

    def gestore(richiesta: httpx2.Request) -> httpx2.Response:
        tentativi.annota(richiesta)
        argomenti = estrazione_di(S014)
        if len(tentativi.richieste) == 1:
            argomenti["relazioni"][0]["tipo"] = "CONOSCE"  # fuori dall'elenco chiuso
        return completion(
            None, prompt=1000, uscita=200, tool_calls=[chiamata("c1", "GrafoEstratto", argomenti)]
        )

    ritenta = Estrattore(instructor.from_openai(openai_finto(gestore)), "gpt-4o-mini", "P", "PD")
    _, speso = await ritenta.estrai(S014)
    assert len(tentativi.richieste) == 2 and speso == 2 * una


# ---------------------------------------------------------------- la risposta ibrida
class EstrattoreFinto(Estrattore):
    """Riconosce nella domanda le entità scelte dal test, e costa una cifra fissa."""

    def __init__(self, nominate: list[Entita]) -> None:
        self._nominate = nominate

    async def nominate(self, domanda: str) -> tuple[EntitaNominate, Decimal]:
        return EntitaNominate(entita=self._nominate), Decimal("0.00005")


async def corpus_ibrido() -> None:
    async with AsyncSessionLocal() as s:
        ingest = IngestService(s, embedder_finto())
        await ingest.ingest("segnalazione_S-2026-014", S014, None, "compliance_only")
        await ingest.ingest("circolare_controparti_07_2026", CIRCOLARE, None, "internal")
    await scrivi(
        [SEGNALAZIONE, DELTA, e("Cliente", "Paolo Ferri", "C-10234")],
        [
            r("S-2026-014", "VERSO", DELTA.menzione, "verso Delta Trading Ltd"),
            r("S-2026-014", "RIGUARDA", "C-10234", "intestato al cliente C-10234, Paolo Ferri"),
        ],
        "segnalazione_S-2026-014",
        "compliance_only",
    )
    await scrivi([CIRC, DELTA], [r("7/2026", "ELENCA", DELTA.menzione, "elenca Delta")], "circ")


# vicina, per l'embedder finto, al solo passaggio della segnalazione
DOMANDA = "La segnalazione riguarda bonifici verso Delta Trading Ltd: quali clienti?"


async def chiedi(user: UserContext, modello: ModelloFisso) -> GraphAdviceResponse:
    async with AsyncSessionLocal() as s:
        servizio = GraphRagService(
            GrafoRepository(s),
            RetrievalService(s),
            embedder_finto(),
            EstrattoreFinto([DELTA]),
            modello,
            "PROMPT",
        )
        return await servizio.answer(DOMANDA, user)


async def test_la_risposta_cita_archi_e_passaggi_e_un_marcatore_inventato_non_e_una_fonte() -> None:
    await corpus_ibrido()
    modello = ModelloFisso("Paolo Ferri [arco-1] [arco-2], come dice [fonte-1]. Altro [arco-99].")
    esito = await chiedi(GIULIA, modello)
    contesto = modello.ricevuti[0][1].content
    assert contesto.startswith("RELAZIONI\n[arco-1]") and "\nPASSAGGI\n[fonte-1]" in contesto
    assert "«intestato al cliente C-10234, Paolo Ferri»" in contesto  # la frase, con l'arco
    assert [f.tipo for f in esito.fonti] == ["arco", "arco", "passaggio"]  # l'arco-99 no
    assert esito.entita == ["Controparte:delta trading ltd"] and esito.archi == 3
    assert esito.cost_eur == pytest.approx(0.00015)  # risposta + riconoscimento della domanda


async def test_l_operatore_non_riceve_dal_grafo_cio_che_il_recupero_gli_nasconde() -> None:
    await corpus_ibrido()
    modello = ModelloFisso("Nel contesto non ci sono clienti.")
    esito = await chiedi(MARCO, modello)
    contesto = modello.ricevuti[0][1].content
    assert "C-10234" not in contesto and "S-2026-014" not in contesto
    assert esito.archi == 1  # solo la circolare, che è interna


class GrafoLento(GrafoRepository):
    def __init__(self, via: asyncio.Event, arrivato: asyncio.Event) -> None:
        self.via, self.arrivato = via, arrivato

    async def esistenti(self, chiavi: list[str]) -> list[str]:
        return chiavi

    async def espandi(self, chiavi: list[str], role: str, hop: int = MAX_HOP) -> list[Arco]:
        self.arrivato.set()
        await self.via.wait()  # aspetta il recupero: se fossero in fila, non finirebbe mai
        return []


class RecuperoCheSblocca(RetrievalService):
    def __init__(self, via: asyncio.Event, arrivato: asyncio.Event) -> None:
        self.via, self.arrivato = via, arrivato

    async def search_for_user(
        self, query_vec: list[float], role: str, top_k: int = 5, soglia: float = 0.0
    ) -> list[RetrievalResult]:
        await self.arrivato.wait()  # il grafo è già partito
        self.via.set()
        return []


async def test_grafo_e_testo_si_cercano_insieme() -> None:
    via, arrivato = asyncio.Event(), asyncio.Event()
    servizio = GraphRagService(
        GrafoLento(via, arrivato),
        RecuperoCheSblocca(via, arrivato),
        embedder_finto(),
        EstrattoreFinto([DELTA]),
        ModelloFisso("Il contesto non basta."),
        "PROMPT",
    )
    esito = await asyncio.wait_for(servizio.answer("Delta?", MARCO), timeout=2)
    assert esito.archi == 0


async def test_l_endpoint_risponde_e_registra_il_costo() -> None:
    await corpus_ibrido()
    app.dependency_overrides[get_estrattore] = lambda: EstrattoreFinto([DELTA])
    app.dependency_overrides[get_llm_provider] = lambda: ModelloFisso("Delta [arco-1].")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        risposta = await c.post("/api/ai/graph-advice", json={"question": "Chi elenca Delta?"})
        corta = await c.post("/api/ai/graph-advice", json={"question": "D?"})
    assert risposta.status_code == 200, risposta.text
    assert risposta.json()["fonti"][0]["document_id"] == "circ"  # Marco vede la sola circolare
    assert corta.status_code == 422
    async with AsyncSessionLocal() as s:
        righe = list(await s.scalars(select(LlmCall)))
    assert [(x.endpoint, x.username) for x in righe] == [("graph", "mbianchi")]
    assert righe[0].cost_eur == Decimal("0.000150")