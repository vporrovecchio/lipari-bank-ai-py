from lipari_bank_ai.config import settings
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.openai_provider import OpenAIProvider
from lipari_bank_ai.llm.anthropic_provider import AnthropicProvider
from lipari_bank_ai.llm.opencode_provider import OpencodeProvider


def get_llm_provider(model: str | None = None) -> LLMProvider:
    selected = model or settings.default_model
    if selected.startswith("gpt"):
        return OpenAIProvider(settings.openai_api_key, selected)
    elif selected.startswith("claude"):
        return AnthropicProvider(settings.anthropic_api_key, selected)
    elif selected.startswith("opencode"):
        return OpencodeProvider(settings.opencode_api_key, selected)
    else:
        raise ValueError(f"Unknown model: {selected}")