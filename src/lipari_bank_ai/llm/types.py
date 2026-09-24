from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMResponse(BaseModel):
    content: str
    tokens_used: int
    cost_eur: float
    model: str