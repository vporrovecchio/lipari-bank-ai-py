"""Il confronto prima/dopo della commessa, prodotto da un comando e non scritto a mano.

Per ogni caso di evals/datasets/grafo.jsonl chiede la stessa cosa all'advisor e al grafo, con
l'utente del seed che ha il ruolo del caso, e conta quante delle risposte attese compaiono.
`--ruolo operator` rifà tutto con il ruolo più basso: è la prova dei permessi.

uv run python -m scripts.confronta_grafo
uv run python -m scripts.confronta_grafo --ruolo operator
uv run python -m scripts.confronta_grafo --dataset evals/datasets/grafo_generato.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from lipari_bank_ai.api.graph import get_estrattore, prompt_risposta
from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.config import settings
from lipari_bank_ai.db.session import AsyncSessionLocal, engine
from lipari_bank_ai.graph_rag.repository import GrafoRepository
from lipari_bank_ai.graph_rag.service import GraphRagService
from lipari_bank_ai.llm.factory import get_embedder, get_instructor, get_llm_provider
from lipari_bank_ai.llm.prompt import load_prompt
from lipari_bank_ai.llm.rewriter import QueryRewriter
from lipari_bank_ai.services.rag_service import RAGService
from lipari_bank_ai.services.retrieval_service import RetrievalService

DATASET = Path("evals/datasets/grafo.jsonl")
UTENTI = {"operator": "mbianchi", "compliance_lead": "grossi", "risk_lead": "lverdi"}


def trovate(risposta: str, attese: list[str]) -> int:
    return sum(a.lower() in risposta.lower() for a in attese)


async def main(casi: list[dict[str, Any]], ruolo: str | None) -> None:
    llm, embedder = get_llm_provider(), get_embedder()
    async with AsyncSessionLocal() as s:
        retrieval = RetrievalService(s)
        rag = RAGService(
            retrieval,
            embedder,
            llm,
            QueryRewriter(llm, load_prompt("rewrite_system_v1")),
            load_prompt("advice_system_v1"),
        )
        grafo = GraphRagService(
            GrafoRepository(s),
            retrieval,
            embedder,
            get_estrattore(get_instructor()),
            llm,
            prompt_risposta(),
        )
        print(f"{'caso':8} {'ruolo':16} {'RAG':>5} {'grafo':>6}   atteso")
        for caso in casi:
            r = ruolo or caso["ruolo"]
            utente = UserContext(username=UTENTI[r], role=r)
            prima = await rag.advise(caso["domanda"], utente)
            dopo = await grafo.answer(caso["domanda"], utente)
            n = len(caso["attese"])
            print(
                f"{caso['id']:8} {r:16} {trovate(prima.answer, caso['attese'])}/{n:<3} "
                f"{trovate(dopo.answer, caso['attese'])}/{n:<4}  {', '.join(caso['attese'])}"
            )
            print(f"         RAG:   {prima.answer[:150]}")
            print(f"         grafo: {dopo.answer[:150]}  [{dopo.archi} archi]")
    await engine.dispose()
    print(f"modello: {settings.default_model}")


if __name__ == "__main__":
    argomenti = argparse.ArgumentParser()
    argomenti.add_argument("--ruolo", choices=list(UTENTI), default=None)
    argomenti.add_argument("--dataset", type=Path, default=DATASET)
    a = argomenti.parse_args()
    casi = [json.loads(r) for r in a.dataset.read_text(encoding="utf-8").splitlines() if r.strip()]
    asyncio.run(main(casi, a.ruolo))  # il file si legge fuori dal loop