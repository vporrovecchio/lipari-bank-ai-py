from lipari_bank_ai.graph_rag.repository import GrafoRepository
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.services.retrieval_service import RetrievalService


class GraphRagService:
    """Grafo e recupero testuale insieme: il contesto è l'unione dei due."""

    def __init__(self, grafo: GrafoRepository, retrieval: RetrievalService,
                 llm: LLMProvider, prompt: str) -> None:
        self.grafo = grafo
        self.retrieval = retrieval
        self.llm = llm
        self.prompt = prompt

    async def answer(self, domanda: str, user: AuthenticatedUser) -> Risposta:
        """Riconosce le entità nella domanda, espande il grafo, recupera i testi, risponde."""
        # 1. le entita' nominate: structured output del Giorno 4 su un modello Pydantic
        nominate = await self.llm.structured(domanda, EntitaNominate)

        # 2. il grafo si espande dalle entita' riconosciute, col tetto sulla profondita'
        archi = await self.grafo.espandi([e.chiave for e in nominate.entita])

        # 3. i testi arrivano dal recupero di ieri, con l'ACL del ruolo: il grafo non
        #    sostituisce il retrieval, gli si aggiunge
        passaggi = await self.retrieval.search_for_user(
            await self.embedder.embed_one(domanda), user.role
        )

        # 4. il contesto e' l'unione dei due, e ognuna delle due parti porta la sua fonte
        contesto = componi_contesto(archi=archi, passaggi=passaggi)
        risposta = await self.llm.complete([
            Message(role="system", content=self.prompt),
            Message(role="user", content=f"CONTESTO\n{contesto}\n\nDOMANDA\n{domanda}"),
        ])
        return Risposta(
            testo=risposta.content,
            fonti=[a["documento_ref"] for a in archi] + [r.document_id for r in passaggi],
        )