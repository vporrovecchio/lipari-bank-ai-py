from pathlib import Path

from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.types.categorize import CategorizeRequest, CategorizeResponse

CATEGORIZE_SYSTEM = """You are an expert at categorizing Italian bank transactions.

Categories:
- UTILITIES (luce, gas, acqua, internet, telefono)
- GROCERIES (supermercati, alimentari, market)
- TRANSPORT (carburante, treno, mezzi, parcheggi)
- RESTAURANTS (ristoranti, bar, fast food)
- ENTERTAINMENT (cinema, palestra, abbonamenti streaming)
- OTHER (tutto il resto)

Subcategory: specifica più precisa in italiano (es. "ENERGY", "SUPERMARKET", "FUEL").
Confidence: tua sicurezza 0.0-1.0.
Reasoning: 1-2 frasi spiegando la scelta.

Return only a valid JSON object without markdown delimiters and with these fields:
category, subcategory, confidence, reasoning.
"""


SYSTEM_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "chat_system_v1.md"
).read_text(encoding="utf-8")


class CategorizeService:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    async def categorize(self, req: CategorizeRequest) -> CategorizeResponse:
        llm = self.llm or get_llm_provider()
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="system", content=CATEGORIZE_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Description: {req.description}\n"
                    f"Amount: €{req.amount} {req.currency}"
                ),
            ),
        ]

        response = await llm.complete(messages, max_tokens=500)
        return CategorizeResponse.model_validate_json(response.content)