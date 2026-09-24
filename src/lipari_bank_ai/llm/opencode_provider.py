from opencode_ai import AsyncOpencode
from opencode_ai.types import TextPartInputParam

from lipari_bank_ai.llm.types import LLMResponse, Message


class OpencodeProvider:
    PRICING = {
        "opencode-big-pickle": (0.0, 0.0),  # Big Pickle è gratuito
    }

    def __init__(self, api_key: str, model: str = "opencode-big-pickle") -> None:
        self.client = AsyncOpencode(base_url="http://localhost:4096")
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 500) -> LLMResponse:
        # Opencode è stateful e ricorda le conversazioni tramite session ID.
        # Per semplicita al momento ne creiamo sempre una nuova
        session = await self.client.session.create()

        parts = [TextPartInputParam(text=m.content, type="text") for m in messages]

        response = await self.client.session.chat(
            id=session.id,
            model_id="opencode-big-pickle",
            provider_id="opencode/big-pickle",
            parts=parts,
        )

        assert response.model_extra is not None
        info = response.model_extra["info"]
        parts_list = response.model_extra["parts"]

        text = next(
            (p["text"] for p in parts_list if p.get("type") == "text"),
            "",
        )

        input_tokens = int(info["tokens"]["input"])
        output_tokens = int(info["tokens"]["output"])

        return LLMResponse(
            content=text,
            tokens_used=input_tokens + output_tokens,
            cost_eur=0.0,
            model=self.model,
        )