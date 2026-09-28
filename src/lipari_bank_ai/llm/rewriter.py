import logging

from lipari_bank_ai.llm.client import LLMProvider, Message

log = logging.getLogger(__name__)


class QueryRewriter:
    """Porta la domanda dell'utente in una forma che il recupero sa cercare."""

    # il provider va costruito con temperature 0: e' una traduzione, non una risposta
    def __init__(self, llm: LLMProvider, prompt: str) -> None:
        self.llm = llm
        self.prompt = prompt

    async def rewrite(self, question: str, history: list[str] | None = None) -> str:
        """Riformula la domanda in una forma che il recupero può cercare.

        Se il modello non risponde o risponde male, ritorna la domanda originale:
        una riscrittura mancata non deve impedire una risposta.
        """
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
            return question

        riscritta = risposta.content.strip().strip('"')
        # una riscrittura vuota, o molto piu' lunga della domanda, e' una risposta
        # travestita: si scarta, e il recupero prosegue sull'originale
        if not riscritta or len(riscritta) > 4 * len(question) + 80:
            log.warning("rewriter.scartata", extra={"uscita": riscritta[:120]})
            return question
        return riscritta