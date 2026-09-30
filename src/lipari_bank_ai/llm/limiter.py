from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def per_utente(request: Request) -> str:
    username = getattr(request.state, "username", None)
    if username:
        return f"user:{username}"
    return get_remote_address(request)


limiter = Limiter(key_func=per_utente)