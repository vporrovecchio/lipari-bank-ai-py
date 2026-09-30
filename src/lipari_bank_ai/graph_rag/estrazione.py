from decimal import Decimal

import instructor

from lipari_bank_ai.agents.loop import costo_chiamata
from lipari_bank_ai.graph_rag.modelli import EntitaNominate, GrafoEstratto, Relazione
from lipari_bank_ai.graph_rag.resolution import normalizza


class Estrattore:
    """Una chiamata al modello per passaggio: estrae, non normalizza e non riconcilia."""

    def __init__(
        self, client: instructor.AsyncInstructor, model: str, prompt: str, prompt_domanda: str
    ) -> None:
        self.client = client
        self.model = model
        self.prompt = prompt
        self.prompt_domanda = prompt_domanda

    async def estrai(self, testo: str) -> tuple[GrafoEstratto, Decimal]:
        """Il grafo di un passaggio, e quanto è costato: l'estrazione è la voce che pesa."""
        estratto, grezza = await self.client.chat.completions.create_with_completion(
            model=self.model,
            response_model=GrafoEstratto,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": testo},
            ],
            max_retries=2,
            max_tokens=1500,
            temperature=0.0,
        )
        return estratto, costo_chiamata(grezza.usage, self.model)

    async def nominate(self, domanda: str) -> tuple[EntitaNominate, Decimal]:
        """Le entità che la domanda nomina: la stessa tecnica, con il suo costo."""
        nominate, grezza = await self.client.chat.completions.create_with_completion(
            model=self.model,
            response_model=EntitaNominate,
            messages=[
                {"role": "system", "content": self.prompt_domanda},
                {"role": "user", "content": domanda},
            ],
            max_retries=1,
            max_tokens=300,
            temperature=0.0,
        )
        return nominate, costo_chiamata(grezza.usage, self.model)


def verificabili(estratto: GrafoEstratto, testo: str) -> tuple[list[Relazione], list[Relazione]]:
    """Le relazioni la cui citazione sta nel testo, e quelle scartate perché non ci sta.

    Il confronto è sulla forma normalizzata, perché il modello cambia spesso un apostrofo o
    una maiuscola; e sui primi ottanta caratteri, perché a volte tronca la frase.
    """
    t = normalizza(testo)
    tenute: list[Relazione] = []
    scartate: list[Relazione] = []
    for r in estratto.relazioni:
        (tenute if normalizza(r.citazione)[:80] in t else scartate).append(r)
    return tenute, scartate