import hashlib
import json
import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from lipari_bank_ai.llm.embedding_client import EmbeddingClient

log = logging.getLogger(__name__)


class CachedEmbedder(EmbeddingClient):
    """Un EmbeddingClient: stessi metodi, stesso contratto, e chi lo riceve non se ne accorge.

    Lo stesso testo con lo stesso modello produce sempre lo stesso vettore: si calcola una
    volta e poi si rilegge. È una sottoclasse, e non un oggetto a parte, perché i servizi dei
    giorni scorsi sono annotati con EmbeddingClient, e mypy guarda il nome della classe.
    """

    TTL_S = 7 * 24 * 3600  # un vettore non invecchia: dipende solo dal testo e dal modello

    def __init__(self, inner: EmbeddingClient, cache: Redis) -> None:
        super().__init__(inner.client, inner.model)
        self._cache = cache
        self.hit = 0
        self.miss = 0

    def chiave(self, text: str) -> str:
        # il modello nella chiave: vettori di modelli diversi vivono in spazi diversi
        return f"emb:{self.model}:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        chiavi = [self.chiave(t) for t in texts]
        try:
            letti = await self._cache.mget(chiavi)
        except RedisError:
            log.warning("cache.non_disponibile", extra={"cache": "embedding"})
            self.miss += len(texts)
            return await super().embed(texts)  # senza cache si paga, e si risponde lo stesso
        vettori: list[list[float] | None] = [None if r is None else json.loads(r) for r in letti]
        mancanti = [i for i, v in enumerate(vettori) if v is None]
        if mancanti:
            # una sola chiamata per tutti i mancanti: il lotto resta un lotto
            nuovi = await super().embed([texts[i] for i in mancanti])
            pipe = self._cache.pipeline(transaction=False)
            for i, vettore in zip(mancanti, nuovi, strict=True):
                vettori[i] = vettore
                pipe.set(chiavi[i], json.dumps(vettore), ex=self.TTL_S)
            try:
                await pipe.execute()
            except RedisError:
                log.warning("cache.non_disponibile", extra={"cache": "embedding"})
        self.hit += len(texts) - len(mancanti)
        self.miss += len(mancanti)
        return [v for v in vettori if v is not None]