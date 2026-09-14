from datetime import UTC, datetime

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import settings


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    app_name: str
    version: str


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Bootcamp Python AI Powered v1 — Lipari Consulting",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="UP",
        timestamp=datetime.now(UTC).isoformat(),
        app_name=settings.app_name,
        version="1.0.0",
    )
