from typing import Any, Literal
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMResponse(BaseModel):
    content: str
    tokens_used: int
    cost_eur: float
    model: str

class ToolSpec(BaseModel):
    """Uno strumento, nella forma che conosce il tuo codice: ogni provider la traduce nella sua."""

    name: str
    description: str  # l'unica cosa che il modello legge per decidere se usarlo
    parameters: dict[str, Any]  # lo schema JSON degli argomenti


class ToolCall(BaseModel):
    """Il modello chiede di eseguire uno strumento. Non lo esegue: lo chiede."""

    id: str
    name: str
    arguments: str  # JSON prodotto dal modello: testo non fidato, da validare prima di usarlo