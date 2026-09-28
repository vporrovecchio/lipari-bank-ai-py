import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.auth.passwords import hash_password, verify_password
from lipari_bank_ai.auth.tokens import ACCESS_TOKEN_TTL, create_access_token
from lipari_bank_ai.db.models import AppUser
from lipari_bank_ai.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_DUMMY_HASH = hash_password("dummy-hash")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Verifica le credenziali ed emette un access token a scadenza.

    Credenziali sbagliate: 401, e il messaggio NON dice quale delle due era sbagliata.
    """
    user = await db.scalar(select(AppUser).where(AppUser.username == form.username))

    hash_to_check = user.password_hash if user is not None else _DUMMY_HASH
    password_ok = await run_in_threadpool(verify_password, form.password, hash_to_check)

    if user is None or not password_ok:
        if user is None:
            logger.warning("Login fallito: utente inesistente %r", form.username)
        else:
            logger.warning("Login fallito: password errata per %r", form.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.username, role=user.role)
    return TokenResponse(
        access_token=token,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )