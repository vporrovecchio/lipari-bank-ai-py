import asyncio
import io
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from lipari_bank_ai.agents.deps import crea_deps
from lipari_bank_ai.agents.loop import run_agent
from lipari_bank_ai.agents.prompts import AGENT_SYSTEM
from lipari_bank_ai.agents.tools import build_tools_for
from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import AsyncSessionLocal
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_embedder, get_llm_provider, get_openai
from lipari_bank_ai.llm.prompt import load_prompt
from lipari_bank_ai.llm.rewriter import QueryRewriter
from lipari_bank_ai.llm.types import LLMResponse, Message
from lipari_bank_ai.services.categorize_service import CategorizeService
from lipari_bank_ai.services.retrieval_service import RetrievalService
from lipari_bank_ai.types.categorize import CategorizeRequest

DATASETS = Path(__file__).parent / "datasets"
# Le soglie stanno nel codice, versionate: abbassarle si vede in una pull request.
# Questi sono punti di partenza: le tue si scelgono guardando i tuoi numeri (la decisione di oggi).
SOGLIE = {"categorize": 0.80, "advice": 0.70, "traiettorie": 0.80}
K = 5  # lo stesso top_k del recupero in produzione: misurare con un k diverso è misurare altro


@dataclass
class Metriche:
    nome: str
    n: int
    punteggio: float
    costo_eur: Decimal = Decimal("0")
    latenza_media_ms: int = 0
    fallimenti: list[dict[str, Any]] = field(default_factory=list)
    dettagli: dict[str, Any] = field(default_factory=dict)


def leggi(path: Path) -> list[dict[str, Any]]:
    """Un caso per riga; le righe vuote non contano."""
    return [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]


class Spesa:
    """Un LLMProvider che somma il costo di ciò che passa: le chiamate che il servizio nasconde."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.totale = Decimal("0")

    async def complete(
        self, messages: list[Message], max_tokens: int = 500
    ) -> LLMResponse:
        risposta = await self.llm.complete(messages, max_tokens)
        # il provider conta in float, il registro in Decimal: si passa dalla stringa,
        # perché 0.1 + 0.2 non è 0.3 e il confronto del gate non deve dipendere da quello
        self.totale += Decimal(str(risposta.cost_eur))
        return risposta



# ---------------------------------------------------------------- la classificazione
async def eval_categorize(path: Path, servizio: CategorizeService,
                          spesa: Spesa) -> Metriche:
    casi = await asyncio.to_thread(leggi, path)  # leggere è bloccante: fuori dal loop
    corretti, fallimenti, coppie, tempo = 0, [], Counter[tuple[str, str]](), 0.0
    for caso in casi:
        atteso = caso["expected"]["category"]
        t0 = time.perf_counter()
        try:
            ottenuto: str = (await servizio.categorize(CategorizeRequest(**caso["input"]))).category
        except Exception as exc:  # noqa: BLE001 — un caso che esplode è un fallimento, non un crash
            ottenuto = f"ERRORE: {type(exc).__name__}"
        tempo += time.perf_counter() - t0
        coppie[(atteso, ottenuto)] += 1
        if ottenuto == atteso:
            corretti += 1
        else:
            fallimenti.append(
                {
                    "id": caso["id"],
                    "descrizione": caso["input"]["description"],
                    "atteso": atteso,
                    "ottenuto": ottenuto,
                    "nota": caso.get("nota", ""),
                }
            )
    # la matrice di confusione: le coppie (atteso, ottenuto). Una categoria che non compare mai
    # fra gli «ottenuti» è una colonna vuota: il modello non la produce
    prodotte = {o for _, o in coppie}
    mai_prodotte = sorted({a for a, _ in coppie} - prodotte)
    return Metriche(
        nome="categorize",
        n=len(casi),
        punteggio=corretti / len(casi) if casi else 0.0,
        costo_eur=spesa.totale,
        latenza_media_ms=int(tempo / len(casi) * 1000) if casi else 0,
        fallimenti=fallimenti,
        dettagli={
            "mai_prodotte": mai_prodotte,
            "coppie": {f"{a}→{o}": n for (a, o), n in coppie.items()},
        },
    )


# ---------------------------------------------------------------- il recupero
async def riscrivi(riscrittore: QueryRewriter, domanda: str, cache: dict[str, str]) -> str:
    """La stessa domanda costa una riscritta sola: la valutazione rivede le stesse query.

    In produzione non serve (ogni domanda arriva una volta sola), in una valutazione sì:
    i dataset ripetono le domande e pagare la riscrittura due volte falsa il costo.
    """
    if (gia := cache.get(domanda)) is None:
        gia = await riscrittore.rewrite(domanda)
        cache[domanda] = gia
    return gia


async def eval_advice(
    path: Path,
    riscrittore: QueryRewriter,
    embedder: EmbeddingClient,
    retrieval: RetrievalService,
    k: int = K,
) -> Metriche:
    """Recall@k sul recupero, per lo stesso percorso della produzione: riscrittura compresa.

    `doc_id` null nel caso vuol dire «non c'è risposta»: il caso passa se niente supera la soglia.
    """
    casi = await asyncio.to_thread(leggi, path)
    trovati, mancati = 0, []
    cache: dict[str, str] = {}
    for caso in casi:
        domanda, ruolo = caso["input"]["question"], caso["input"]["role"]
        riscritta = await riscrivi(riscrittore, domanda, cache)
        vettore = await embedder.embed_one(riscritta)
        passaggi = await retrieval.search_for_user(vettore, ruolo, top_k=k)
        documenti = [p.document_id for p in passaggi]
        atteso = caso["expected"]["doc_id"]
        if (atteso is None and not documenti) or (atteso is not None and atteso in documenti):
            trovati += 1
        else:
            # i mancati sono il dato utile: l'aggregato dice SE, questi dicono DOVE
            mancati.append(
                {
                    "id": caso["id"],
                    "atteso": atteso,
                    "recuperati": documenti,
                    "query_riscritta": riscritta,
                    "nota": caso.get("nota", ""),
                }
            )
    return Metriche(
        nome="advice",
        n=len(casi),
        punteggio=trovati / len(casi) if casi else 0.0,
        fallimenti=mancati,
        dettagli={"k": k},
    )


# ---------------------------------------------------------------- le traiettorie
CONTROLLI = ("tool_richiesti", "tool_vietati", "passi", "terminazione", "citazione")


async def eval_traiettorie(
    path: Path, client: AsyncOpenAI, model: str, embedder: EmbeddingClient
) -> Metriche:
    """L'agente sul percorso: quali tool, quanti passi, come finisce, se cita."""
    casi = await asyncio.to_thread(leggi, path)
    superati, fallimenti, costo = 0, [], Decimal("0")
    violazioni = Counter[str]()
    for caso in casi:
        utente = UserContext(**caso["input"]["user"])  # un utente del seed, con il suo portafoglio
        async with AsyncSessionLocal() as s:
            deps = crea_deps(s, embedder=embedder, openai=client)
            run = await run_agent(
                messaggi=[
                    {"role": "system", "content": AGENT_SYSTEM},
                    {"role": "user", "content": caso["input"]["message"]},
                ],
                tools=build_tools_for(utente, deps),
                client=client,
                model=model,
                max_steps=8,  # più alto di passi_max: si misura quanti ne usa, non dove sbatte
                runs=deps.runs,
                user=utente,
            )
            chiamati = set(run.tool_calls)
            if run.stopped_by == "awaiting_approval":
                # il tool che aspetta l'approvazione non è in tool_calls: non è ancora partito,
                # ma il modello l'ha chiesto, e per tool_vietati conta
                stato = await deps.runs.get(run.run_id)
                chiamati |= {c["function"]["name"] for c in (stato.pending_calls if stato else [])}
        costo += run.cost_eur
        att = caso["expected"]
        da_citare = att.get("documento_da_citare")
        esiti = {
            "tool_richiesti": set(att["tool_richiesti"]) <= chiamati,
            "tool_vietati": not (set(att.get("tool_vietati", [])) & chiamati),
            "passi": run.steps <= att["passi_max"],
            "terminazione": run.stopped_by == att["terminazione"],
            # l'agente cita con l'identificativo che search_documents gli restituisce
            "citazione": f"[{da_citare}]" in run.reply if da_citare else True,
        }
        violati = [c for c in CONTROLLI if not esiti[c]]
        violazioni.update(violati)
        # un caso passa solo se passa tutto: un tool vietato non vale «quattro su cinque»
        if violati:
            fallimenti.append(
                {
                    "id": caso["id"],
                    "violati": violati,
                    "tool_chiamati": sorted(chiamati),
                    "passi": run.steps,
                    "stopped_by": run.stopped_by,
                    "nota": caso.get("nota", ""),
                }
            )
        else:
            superati += 1
    return Metriche(
        nome="traiettorie",
        n=len(casi),
        punteggio=superati / len(casi) if casi else 0.0,
        costo_eur=costo,
        fallimenti=fallimenti,
        dettagli={"violazioni_per_controllo": dict(violazioni)},
    )


# ---------------------------------------------------------------- il cancello
def cancello(risultati: list[Metriche], soglie: dict[str, float]) -> bool:
    """Tutto sopra soglia? Una metrica senza soglia è un errore, non un passaggio."""
    return all(m.punteggio >= soglie[m.nome] for m in risultati)


def rapporto(risultati: list[Metriche], soglie: dict[str, float]) -> str:
    # solo caratteri che ogni terminale sa stampare: su Windows un'uscita rediretta su file
    # usa la codifica del sistema, e un simbolo che non c'è fa cadere il runner alla fine
    righe = []
    for m in risultati:
        ok = m.punteggio >= soglie[m.nome]
        esito = "ok" if ok else "SOTTO SOGLIA"
        righe.append(
            f"{m.nome:12} {m.punteggio:.2f} ({m.n - len(m.fallimenti)}/{m.n})  "
            f"soglia {soglie[m.nome]:.2f}  {esito:12}  costo €{m.costo_eur:.4f}"
        )
        for f in m.fallimenti[:5]:  # i casi, non i numeri: fra un mese ti servono questi
            dettaglio = {k: v for k, v in f.items() if k != "id"}
            righe.append(f"    x {f['id']}: {json.dumps(dettaglio, ensure_ascii=False)}")
        if m.dettagli.get("mai_prodotte"):
            righe.append(f"    categorie mai prodotte: {', '.join(m.dettagli['mai_prodotte'])}")
    return "\n".join(righe)


async def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        # le note dei casi le scrivi tu, e possono contenere qualunque carattere: uno che la
        # codifica del terminale non ha diventa «?» invece di un'eccezione a fine esecuzione
        sys.stdout.reconfigure(errors="replace")
    openai = get_openai()
    embedder = get_embedder()
    # il modello che il loop sa valutare: PREZZI_PER_1K non conosce default_model,
    # e un run_agent su un modello senza prezzo si ferma prima di cominciare
    model = settings.agent_model
    # due contatori, uno per servizio: il costo di ciascuno va nel suo report
    classificazione = Spesa(get_llm_provider(model))
    riscrittura = Spesa(get_llm_provider(model))
    async with AsyncSessionLocal() as s:
        risultati = [
            await eval_categorize(
                DATASETS / "categorize.jsonl",
                CategorizeService(classificazione),
                classificazione,
            ),
            await eval_advice(
                DATASETS / "advice.jsonl",
                QueryRewriter(riscrittura, load_prompt("rewrite_system")),
                embedder,
                RetrievalService(s, embedder),
            ),
        ]
    risultati[0].costo_eur = classificazione.totale
    risultati[1].costo_eur = riscrittura.totale  # le riscritture; gli embedding non passano da qui
    risultati.append(
        await eval_traiettorie(
            DATASETS / "agent_traiettorie.jsonl", openai, model, embedder
        )
    )
    print(rapporto(risultati, SOGLIE))
    passato = cancello(risultati, SOGLIE)
    print("GATE:", "passato" if passato else "NON passato")
    return 0 if passato else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
