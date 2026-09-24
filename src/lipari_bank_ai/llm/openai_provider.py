from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from lipari_bank_ai.llm.types import LLMResponse, Message


class OpenAIProvider:
    PRICING = {  # EUR per 1k tokens (input/output)
        "gpt-4o-mini": (0.00014, 0.00056),
        "gpt-4o": (0.0023, 0.0091),
    }

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 500) -> LLMResponse:
        openai_messages: list[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": m.role, "content": m.content})
            for m in messages
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        input_cost, output_cost = self.PRICING[self.model]
        cost_eur = (input_tokens * input_cost + output_tokens * output_cost) / 1000

        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_used=usage.total_tokens if usage else 0,
            cost_eur=cost_eur,
            model=self.model,
        )