import asyncio
import re

from pydantic import BaseModel

from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.graph_rag.estrazione import Estrattore
from lipari_bank_ai.graph_rag.repository import Arco, GrafoRepository
from lipari_bank_ai.graph_rag.resolution import chiave
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.services.rag_service import contesto
from lipari_bank_ai.services.retrieval_service import RetrievalResult, RetrievalService

MARCATORE_ARCO = re.compile(r"\[arco-(\d+)\]")
MARCATORE_FONTE = re.compile(r"\[fonte-(\d+)\]")


class Fonte(BaseModel):
    tipo: str  # "arco" o "passaggio"
    document_id: str
    estratto: str


class GraphAdviceResponse(BaseModel):
    answer: str
    fonti: list[Fonte]
    entita: list[str]  # le chiavi del grafo da cui è partita la ricerca
    archi: int
    cost_eur: float


def componi_contesto(archi: list[Arco], passaggi: list[RetrievalResult]) -> str:
    """Le relazioni con la loro frase e il loro documento, poi i passaggi: due fonti etichettate."""
    relazioni = "\n".join(
        f"[arco-{i}] {a.da} -{a.tipo}-> {a.a} «{a.citazione}» (documento: {a.document_id})"
        for i, a in enumerate(archi, start=1)
    )
    return f"RELAZIONI\n{relazioni or '(nessuna)'}\n\nPASSAGGI\n{contesto(passaggi) or '(nessuno)'}"


def fonti_citate(risposta: str, archi: list[Arco], passaggi: list[RetrievalResult]) -> list[Fonte]:
    """Le fonti che la risposta cita davvero: un marcatore inventato non diventa una fonte."""
    fonti: list[Fonte] = []
    for n in dict.fromkeys(MARCATORE_ARCO.findall(risposta)):
        if 0 < int(n) <= len(archi):
            a = archi[int(n) - 1]
            fonti.append(Fonte(tipo="arco", document_id=a.document_id, estratto=a.citazione))
    for n in dict.fromkeys(MARCATORE_FONTE.findall(risposta)):
        if 0 < int(n) <= len(passaggi):
            p = passaggi[int(n) - 1]
            fonti.append(
                Fonte(tipo="passaggio", document_id=p.document_id, estratto=p.content[:200])
            )
    return fonti


class GraphRagService:
    def __init__(
        self,
        grafo: GrafoRepository,
        retrieval: RetrievalService,
        embedder: EmbeddingClient,
        estrattore: Estrattore,
        llm: LLMProvider,
        prompt: str,
    ) -> None:
        self.grafo = grafo
        self.retrieval = retrieval
        self.embedder = embedder
        self.estrattore = estrattore
        self.llm = llm
        self.prompt = prompt

    async def answer(self, domanda: str, user: UserContext) -> GraphAdviceResponse:
        # 1. le entità nominate, e fra quelle quali sono nodi del grafo
        nominate, costo_domanda = await self.estrattore.nominate(domanda)
        chiavi = await self.grafo.esistenti([chiave(e) for e in nominate.entita])

        # 2 e 3. il grafo e il testo sono indipendenti: si cercano insieme. Tutti e due con i
        # livelli del ruolo: il grafo non deve far vedere ciò che il recupero nasconde
        async def passaggi_per_ruolo() -> list[RetrievalResult]:
            return await self.retrieval.search_for_user(
                await self.embedder.embed_one(domanda), user.role
            )

        archi, passaggi = await asyncio.gather(
            self.grafo.espandi(chiavi, user.role), passaggi_per_ruolo()
        )

        # 4. il contesto è l'unione, e ogni riga dice da dove viene
        risposta = await self.llm.complete(
            [
                Message(role="system", content=self.prompt),
                Message(
                    role="user",
                    content=f"{componi_contesto(archi, passaggi)}\n\nDOMANDA\n{domanda}",
                ),
            ],
            max_tokens=500,
        )
        return GraphAdviceResponse(
            answer=risposta.content,
            fonti=fonti_citate(risposta.content, archi, passaggi),
            entita=chiavi,
            archi=len(archi),
            cost_eur=float(risposta.cost_eur + costo_domanda),  # anche il riconoscimento
        )