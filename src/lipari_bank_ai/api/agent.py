from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from lipari_bank_ai.agents.deps import Deps, get_deps
from lipari_bank_ai.agents.loop import run_agent
from lipari_bank_ai.agents.prompts import AGENT_SYSTEM
from lipari_bank_ai.agents.tools import build_tools_for
from lipari_bank_ai.auth.deps import UserContext, get_current_user

router = APIRouter(prefix="/api/ai", tags=["agent"])


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    run_id: str
    reply: str
    steps: int
    tool_calls: list[str]
    stopped_by: str        # "model" | "max_steps" | "budget": si distingue senza leggere il testo
    cost_eur: float


@router.post("/agent", response_model=AgentResponse)
async def agent(
    payload: AgentRequest,
    user: Annotated[UserContext, Depends(get_current_user)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> AgentResponse:
    run = await run_agent(
        messaggi=[{"role": "system", "content": AGENT_SYSTEM},
                  {"role": "user", "content": payload.message}],
        tools=build_tools_for(user, deps),      # ← i tool nascono qui, per lui
        client=deps.openai,
        model=deps.model,
    )
    return AgentResponse(
        run_id=run.run_id, reply=run.reply, steps=run.steps,
        tool_calls=run.tool_calls, stopped_by=run.stopped_by, cost_eur=float(run.cost_eur),
    )