import httpx

from lipari_bank_ai.config import settings


class EmbeddingClient:
    def __init__(self) -> None:
        self.url = f"{settings.ollama_url}/api/embed"
        self.model = settings.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts via Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.url,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        return [list(item) for item in payload["embeddings"]]

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]