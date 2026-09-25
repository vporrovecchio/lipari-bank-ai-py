import asyncio
from pathlib import Path

import httpx


async def main() -> None:
    docs_dir = Path("data/docs")
    paths = await asyncio.to_thread(lambda: sorted(docs_dir.glob("*.md")))

    print(f"Paths: {paths}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        for path in paths:
            content = await asyncio.to_thread(path.read_text)
            response = await client.post(
                "http://localhost:8000/api/ai/documents/ingest",
                json={
                    "document_id": path.stem,
                    "content": content,
                    "metadata": {"source": str(path), "title": path.stem.replace("_", " ")},
                },
            )
            print(f"{path.name}: {response.json()}")


if __name__ == "__main__":
    asyncio.run(main())