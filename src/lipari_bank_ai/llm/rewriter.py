import hashlib
import logging
from dataclasses import dataclass
from decimal import Decimal

from redis.asyncio import Redis
from redis.exceptions import RedisError

from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.types import LLMResponse, Message

log = logging.getLogger(__name__)

TTL_S = 3600  # un'ora: le domande si ripetono nella stessa mattina, e il beneficio finisce lì


@dataclass
class Riscrittura:
    """Cosa è andato al recupero, se veniva dalla cache, e quanto è costato ottenerlo."""

    testo: str
    da_cache: bool = False
    tokens: int = 0
    cost_eur: Decimal = Decimal("0")


class QueryRewriter:
    """Porta la domanda dell'utente in una forma che il recupero sa cercare."""

    def __init__(
        self, llm: LLMProvider, prompt: str, enabled: bool = True, cache: Redis | None = None
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.enabled = enabled  # l'interruttore: spento, il recupero cerca sull'originale
        # Giorno 10: con Redis la cache è una per tutti i worker; senza, resta quella del processo
        self._redis = cache
        self._locale: dict[str, str] = {}
        # l'impronta del prompt entra nella chiave: un prompt nuovo è una cache nuova
        self._versione = hashlib.sha256(prompt.encode()).hexdigest()[:8]

    async def _riscrivi(
        self, question: str, history: list[str] | None
    ) -> tuple[str | None, LLMResponse | None]:
        """La riscrittura e la risposta che l'ha prodotta; testo None se è da scartare."""
        coda = "\n".join(history[-3:]) if history else ""
        try:
            risposta = await self.llm.complete(
                [
                    Message(role="system", content=self.prompt),
                    Message(role="user", content=f"{coda}\n\nDomanda: {question}".strip()),
                ],
                max_tokens=80,
            )
        except Exception:
            log.warning("rewriter.fallito", extra={"domanda": question})
            return None, None

        riscritta = risposta.content.strip().strip('"')
        # una riscrittura vuota, o molto più lunga della domanda, è una risposta
        # travestita: si scarta, e il recupero prosegue sull'originale
        if not riscritta or len(riscritta) > 4 * len(question) + 80:
            log.warning("rewriter.scartata", extra={"uscita": riscritta[:120]})
            return None, risposta  # scartata, ma pagata
        return riscritta, risposta

    async def rewrite(self, question: str, history: list[str] | None = None) -> str:
        """Riformula la domanda; se non ci riesce, ritorna l'originale.

        Una riscrittura mancata non deve impedire una risposta.
        """
        if not self.enabled:
            return question
        riscritta, _ = await self._riscrivi(question, history)
        return riscritta or question

    def chiave(self, question: str) -> str:
        # l'hash e non il testo: una domanda può contenere il nome di un cliente
        return f"rewrite:{self._versione}:{hashlib.sha256(question.encode()).hexdigest()}"

    async def _leggi(self, chiave: str) -> str | None:
        if self._redis is None:
            return self._locale.get(chiave)
        try:
            valore = await self._redis.get(chiave)
        except RedisError:
            # la cache è un risparmio, non una dipendenza: senza, si riscrive e basta
            log.warning("cache.non_disponibile", extra={"cache": "rewrite"})
            return None
        return valore.decode() if isinstance(valore, bytes) else valore

    async def _scrivi(self, chiave: str, valore: str) -> None:
        if self._redis is None:
            self._locale[chiave] = valore
            return
        try:
            await self._redis.set(chiave, valore, ex=TTL_S)
        except RedisError:
            log.warning("cache.non_disponibile", extra={"cache": "rewrite"})

    async def riscrivi_per_ricerca(self, question: str) -> Riscrittura:
        """La riscrittura per il recupero, dalla cache se c'è, con il suo costo."""
        if not self.enabled:
            return Riscrittura(question)
        chiave = self.chiave(question)
        if (cached := await self._leggi(chiave)) is not None:
            return Riscrittura(cached, da_cache=True)
        riscritta, risposta = await self._riscrivi(question, None)
        pagata = Riscrittura(
            riscritta or question,
            tokens=risposta.tokens_used if risposta else 0,
            cost_eur=risposta.cost_eur if risposta else Decimal("0"),
        )
        if riscritta is not None:  # il ripiego non va in cache: la prossima volta si riprova
            await self._scrivi(chiave, riscritta)
        return pagata

    async def rewrite_cached(self, question: str) -> str:
        """Come rewrite, ma la stessa domanda si riscrive una volta sola."""
        return (await self.riscrivi_per_ricerca(question)).testo