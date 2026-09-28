import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any
from venv import logger

from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.exceptions import LLMProviderError
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.rewriter import QueryRewriter
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.services.retrieval_service import RetrievalResult, RetrievalService
from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse, Citation

ADVICE_SYSTEM = """Sei un advisor bancario LipariBank esperto.

Hai accesso a documenti ufficiali (regolamenti, tariffe, condizioni). Rispondi alla domanda
dell'utente basandoti ESCLUSIVAMENTE sui CONTESTO forniti.

Regole:
- Se la risposta non è nei contesti, dillo onestamente ("Non ho informazioni su...").
- Cita sempre il documento da cui prendi l'informazione: [doc_id: <id>].
- Tono professionale, sintetico.
- Risposta in italiano.
"""

@dataclass
class Fasi:
    """Millisecondi per fase, per il log strutturato della richiesta."""

    auth_ms: int = 0
    rewrite_ms: int = 0
    embedding_ms: int = 0
    retrieval_ms: int = 0
    prompt_ms: int = 0
    llm_ms: int = 0
    cache_hit: bool = False
    used_fallback: bool = False

    @property
    def total_ms(self) -> int:
        return (
            self.auth_ms + self.rewrite_ms + self.embedding_ms
            + self.retrieval_ms + self.prompt_ms + self.llm_ms
        )

class RAGService:
    def __init__(
            self,
            retrieval: RetrievalService,
            llm: LLMProvider,
        ) -> None:
            self.retrieval = retrieval
            self.llm = llm

    async def answer(self, req: AdviceRequest) -> AdviceResponse:
        # 1. Retrieve top-k chunks
        chunks = await self.retrieval.retrieve(req.question, top_k=5)

        if not chunks:
            return AdviceResponse(
                answer="Non ho documenti correlati alla tua domanda.",
                citations=[],
                tokens_used=0,
                cost_eur=0,
            )

        # 2. Build context string
        context_parts = [
            f"""
                [doc_id: {c.document_id}, chunk: {c.chunk_id},
                similarity: {c.similarity:.2f}]\n{c.content}
            """
            for c in chunks
        ]
        context = "\n\n---\n\n".join(context_parts)

        # 3. Generate
        user_prompt = f"""CONTESTI:
            {context}

            DOMANDA: {req.question}

            RISPOSTA (con citazioni):"""

        llm_response = await self.llm.complete(
            messages=[
                Message(role="system", content=ADVICE_SYSTEM),
                Message(role="user", content=user_prompt),
            ],
            max_tokens=800,
        )

        # 4. Build citations from retrieved chunks
        citations = [
            Citation(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                excerpt=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                similarity=c.similarity,
            )
            for c in chunks
        ]

        return AdviceResponse(
            answer=llm_response.content,
            citations=citations,
            tokens_used=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
        )

    async def advice(self, question: str, user: UserContext) -> AdviceResponse:
        fasi = Fasi()

        rewriter = QueryRewriter(self.llm, question)
        embedder = EmbeddingClient()

        with _cronometro(fasi, "rewrite_ms"):
            search_query = await rewriter.rewrite(question)

        with _cronometro(fasi, "embedding_ms"):
            query_vec = await embedder.embed_one(search_query)

        with _cronometro(fasi, "retrieval_ms"):
            chunks = await self.retrieval.search_for_user(query_vec, user.role)

        with _cronometro(fasi, "prompt_ms"):
            # La riscritta serve al retrieval. Al modello va la domanda dell'utente.
            prompt = self._build_prompt(question=question, chunks=chunks)

        with _cronometro(fasi, "llm_ms"):
            answer, tokens_used, cost_eur, fasi.used_fallback = await self._generate_or_degrade(
                prompt, chunks
            )

        logger.info(
            "advice_completata",
            extra={
                "username": user.username,
                "role": user.role,
                "chunk_count": len(chunks),
                "chunk_ids": [c.chunk_id for c in chunks],
                **asdict(fasi),
            },
        )

        return AdviceResponse(
            answer=answer,
            citations=[
                Citation(
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    excerpt=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    similarity=c.similarity,
                )
                for c in chunks
            ],
            tokens_used=tokens_used,
            cost_eur=cost_eur,
            rewritten_query=search_query,
        )

    async def _generate_or_degrade(
        self, prompt: str, chunks: list[RetrievalResult]
    ) -> tuple[str, int, float, bool]:
        try:
            response = await self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                max_tokens=500,
            )
            return response.content, response.tokens_used, response.cost_eur, False
        except LLMProviderError:
            logger.warning("generator_non_disponibile_fallback_su_chunk")
            estratti = "\n\n".join(
                f"[fonte-{i}] {c.document_id}\n{c.content}"
                for i, c in enumerate(chunks[:3], start=1)
            )
            return (
                "⚠️ Risposta parziale: il servizio di sintesi non è momentaneamente "
                "disponibile. Di seguito i passaggi dei documenti più pertinenti "
                f"alla tua domanda.\n\n{estratti}"
            ), 0, 0.0, True

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievalResult]) -> str:
        context = "\n\n---\n\n".join(
            f"[doc_id: {c.document_id}, chunk: {c.chunk_id}, "
            f"similarity: {c.similarity:.2f}]\n{c.content}"
            for c in chunks
        )
        return f"""CONTESTI:
            {context}

            DOMANDA: {question}

            RISPOSTA (con citazioni):"""

@contextmanager
def _cronometro(fasi: Fasi, campo: str) -> Generator[Any, Any, Any]:
    inizio = time.perf_counter()
    try:
        yield
    finally:
        setattr(fasi, campo, int((time.perf_counter() - inizio) * 1000))