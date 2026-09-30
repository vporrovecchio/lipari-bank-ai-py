from functools import cache

from redis import Redis
from lipari_bank_ai.config import settings


@cache  # un client per processo, con il suo pool: come quello di OpenAI nella factory
def get_redis() -> Redis | None:
    """Il client Redis, o None se REDIS_URL è vuota: allora ogni cache resta nel processo."""
    if not settings.redis_url:
        return None
    # timeout brevi: una cache che risponde tardi costa più di una cache che non risponde
    return Redis.from_url(settings.redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)


async def chiudi_redis() -> None:
    if (redis := get_redis()) is not None:
        await redis.aclose()