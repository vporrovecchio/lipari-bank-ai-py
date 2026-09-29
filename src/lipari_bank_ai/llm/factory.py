from functools import lru_cache

from openai import AsyncOpenAI

from lipari_bank_ai.config import settings
from lipari_bank_ai.llm.anthropic_provider import AnthropicProvider
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.openai_provider import OpenAIProvider
from lipari_bank_ai.llm.opencode_provider import OpencodeProvider


def get_llm_provider(model: str | None = None) -> LLMProvider:
    selected = model or settings.default_model
    if selected.startswith("gpt"):
        return OpenAIProvider(settings.openai_api_key, selected)
    elif selected.startswith("claude"):
        return AnthropicProvider(settings.anthropic_api_key, selected)
    elif selected.startswith("opencode"):
        return OpencodeProvider(settings.opencode_api_key, selected)
    elif selected == settings.agent_model:
        # il modello dell'agente gira in locale e parla il protocollo OpenAI
        return OpenAIProvider("ollama", selected, base_url=f"{settings.ollama_url}/v1")
    else:
        raise ValueError(f"Unknown model: {selected}")


@lru_cache
def get_openai() -> AsyncOpenAI:
    """Il client grezzo, per chi parla all'API senza passare da un provider.

    C'è una sola istanza: due client sulla stessa connessione non aggiungono nulla
    e il pool di connessioni è condiviso.
    """
    return AsyncOpenAI(
        api_key="ollama", base_url=f"{settings.ollama_url}/v1", timeout=120.0
    )


def get_embedder() -> EmbeddingClient:
    """Il client degli embedding. Senza stato, quindi una nuova istanza ogni volta."""
    return EmbeddingClient()
