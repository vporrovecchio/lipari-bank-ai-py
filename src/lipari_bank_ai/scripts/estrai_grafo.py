"""Estrae entità e relazioni dai documenti che portano relazioni, e solo da quelli nuovi.

I passaggi e il loro livello di visibilità sono quelli dell'indice del Giorno 5: il grafo eredita
l'ACL del documento da cui viene ogni arco. Alla fine stampa il costo, gli scarti e la coda.

uv run python -m scripts.estrai_grafo                       # segnalazioni e circolari
uv run python -m scripts.estrai_grafo --prefissi segnalazione_
"""

import argparse
import asyncio
import hashlib
from collections import defaultdict
from decimal import Decimal

from sqlalchemy import func, select

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.models import DocumentChunk, GraphExtraction, GraphReview
from lipari_bank_ai.db.session import AsyncSessionLocal, engine
from lipari_bank_ai.graph_rag.estrazione import Estrattore, verificabili
from lipari_bank_ai.graph_rag.repository import GrafoRepository
from lipari_bank_ai.llm.factory import get_instructor
from lipari_bank_ai.llm.prompt import load_prompt

# estrarre da tutto è la scelta che nessuno rivede: si estrae dai documenti che portano relazioni
PREFISSI = ("segnalazione_", "circolare_")


async def main(prefissi: tuple[str, ...]) -> None:
    estrattore = Estrattore(
        get_instructor(),
        settings.default_model,
        load_prompt("estrazione_grafo_v1"),
        load_prompt("grafo_domanda_v1"),
    )
    costo, passaggi_letti, archi, scartate, orfane = Decimal("0"), 0, 0, 0, 0
    async with AsyncSessionLocal() as session:
        righe = await session.execute(
            select(
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.visibility,
            ).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        documenti: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for doc, _, testo, livello in righe.all():
            if doc.startswith(prefissi):
                documenti[doc].append((testo, livello))
        grafo = GrafoRepository(session)
        for doc, pezzi in documenti.items():
            impronta = hashlib.sha256("".join(t for t, _ in pezzi).encode()).hexdigest()
            fatto = await session.get(GraphExtraction, doc)
            if fatto is not None and fatto.impronta == impronta:
                continue  # incrementale: lo stesso testo non si ripaga
            if fatto is not None:
                await grafo.dimentica(doc)  # il testo è cambiato: i suoi fatti vecchi, via
            tenute_doc, scartate_doc = 0, 0
            for testo, livello in pezzi:
                estratto, speso = await estrattore.estrai(testo)
                costo += speso
                passaggi_letti += 1
                tenute, via = verificabili(estratto, testo)
                scritti, senza_estremo = await grafo.upsert(estratto.entita, tenute, doc, livello)
                tenute_doc += scritti
                scartate_doc += len(via)
                orfane += senza_estremo
            await session.merge(
                GraphExtraction(
                    document_id=doc, impronta=impronta, relazioni=tenute_doc, scartate=scartate_doc
                )
            )
            await session.commit()  # un documento alla volta: un errore a metà non perde i primi
            archi += tenute_doc
            scartate += scartate_doc
            print(f"{doc}: {tenute_doc} archi, {scartate_doc} scartati")
        coda = await session.scalar(
            select(func.count()).select_from(GraphReview).where(GraphReview.stato == "da_rivedere")
        )
    await engine.dispose()
    totale = archi + scartate
    print(
        f"{passaggi_letti} passaggi, {archi} archi, {scartate} relazioni scartate "
        f"({scartate / totale:.0%} delle estratte)"
        if totale
        else f"{passaggi_letti} passaggi"
    )
    print(f"costo dell'estrazione: €{costo:.6f}   in coda di revisione: {coda}   orfane: {orfane}")


if __name__ == "__main__":
    argomenti = argparse.ArgumentParser()
    argomenti.add_argument("--prefissi", nargs="+", default=list(PREFISSI))
    asyncio.run(main(tuple(argomenti.parse_args().prefissi)))