"""Il servizio `seed` del compose. Si lancia a ogni `docker compose up`, quindi è idempotente:
utenti e conti si saltano da soli se ci sono, e l'indice si ricostruisce solo se è vuoto.

uv run python -m scripts.seed_all            # SEED_REINDEX=1 per reindicizzare comunque
"""

import asyncio
import os

from sqlalchemy import func, select

from lipari_bank_ai.evals.ingest_fixtures import main as indicizza
from lipari_bank_ai.scripts.seed_accounts import main as seed_accounts
from lipari_bank_ai.scripts.seed_users import main as seed_users
from lipari_bank_ai.db.models import DocumentChunk
from lipari_bank_ai.db.session import AsyncSessionLocal, engine


async def _passaggi_indicizzati() -> int:
    async with AsyncSessionLocal() as session:
        return await session.scalar(select(func.count()).select_from(DocumentChunk)) or 0


async def main() -> None:
    await seed_users()  # Marco, Giulia e Lucia, dal Giorno 6: salta chi c'è già
    await seed_accounts()  # i due clienti, i tre conti, i movimenti del Giorno 3: idem
    # l'indicizzazione è la sola parte che paga, perché calcola gli embedding: si fa se
    # l'indice è vuoto, e con la cache davanti un secondo giro non ripaga i passaggi uguali
    if await _passaggi_indicizzati() and os.environ.get("SEED_REINDEX") != "1":
        print("Indice già popolato: niente da fare (SEED_REINDEX=1 per reindicizzare).")
    else:
        await indicizza()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())