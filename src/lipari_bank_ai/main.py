import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from lipari_bank_ai.api import admin, advice, agent, auth, categorize, chat
from lipari_bank_ai.config import settings
from lipari_bank_ai.exceptions import AppError
from observability.json_log import configura_log, request_id

configura_log()  # Giorno 9: le righe di log diventano JSON, con il request_id
logger = logging.getLogger(__name__)

ID_VALIDO = re.compile(r"[\w.-]{1,64}")   # l'id del chiamante, o un UUID nuovo: niente spazi

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Bootcamp Python AI Powered v1",
)

@app.exception_handler(AppError)
async def app_exception_handler(req: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "timestamp": datetime.now(UTC).isoformat(),
            "status": exc.status_code,
            "error": exc.code,
            "message": exc.message,
            "path": req.url.path,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(req: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,  # default FastAPI
        content={
            "timestamp": datetime.now(UTC).isoformat(),
            "status": 422,
            "error": "VALIDATION_ERROR",
            "message": "Input non valido",
            "path": req.url.path,
            "details": [f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()],
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(req: Request, exc: Exception) -> JSONResponse:
    # logger.exception(exc) in G7
    return JSONResponse(
        status_code=500,
        content={
            "timestamp": datetime.now(UTC).isoformat(),
            "status": 500,
            "error": "INTERNAL_ERROR",
            "message": "Errore inatteso",
            "path": req.url.path,
        },
    )

@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Se il chiamante ne manda uno, lo stesso id attraversa i due sistemi; se non lo manda,
    # o manda qualcosa che non ha la forma di un id, se ne genera uno nuovo.
    ricevuto = request.headers.get("X-Request-Id", "")
    rid = ricevuto if ID_VALIDO.fullmatch(ricevuto) else str(uuid.uuid4())
    request_id.set(rid)  # Giorno 9: da qui ogni riga di log di questa richiesta lo porta
    inizio = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    response.headers["X-Process-Time"] = f"{perf_counter() - inizio:.4f}"
    return response

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "UP"}


app.include_router(chat.router)
app.include_router(categorize.router)
app.include_router(advice.router)
app.include_router(auth.router)
app.include_router(agent.router)
app.include_router(admin.router)
