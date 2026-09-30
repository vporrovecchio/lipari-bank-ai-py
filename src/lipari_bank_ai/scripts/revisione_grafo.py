"""uv run python -m scripts.revisione_grafo                 # l'elenco dei casi da rivedere
uv run python -m scripts.revisione_grafo unite <id>      # sono la stessa entità
uv run python -m scripts.revisione_grafo diverse <id>    # sono due entità

Unire sposta gli archi sulla chiave candidata, e da quel momento anche le estrazioni nuove la
usano. La decisione resta scritta nella coda, con chi era cosa e quanto si somigliava.
"""

import asyncio
import sys

from sqlalchemy import select

from lipari_bank_ai.db.models import GraphReview
from lipari_bank_ai.db.session import AsyncSessionLocal, engine
from lipari_bank_ai.graph_rag.repository import GrafoRepository


async def main(argomenti: list[str]) -> None:
    async with AsyncSessionLocal() as session:
        if len(argomenti) == 2 and argomenti[0] in {"unite", "diverse"}:
            caso = await session.get(GraphReview, argomenti[1])
            if caso is None:
                raise SystemExit(f"Nessun caso {argomenti[1]}")
            caso.stato = argomenti[0]
            if caso.stato == "unite":
                await GrafoRepository(session).unisci(caso.chiave, caso.candidata)
            await session.commit()
            print(f"{caso.chiave} e {caso.candidata}: {caso.stato}")
        else:
            casi = await session.scalars(
                select(GraphReview)
                .where(GraphReview.stato == "da_rivedere")
                .order_by(GraphReview.somiglianza.desc())
            )
            for c in casi:
                print(
                    f"{c.id}  {c.somiglianza:.2f}  «{c.chiave}» ~ «{c.candidata}»  {c.document_id}"
                )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))