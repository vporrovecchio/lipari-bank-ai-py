import asyncio
import sys

from lipari_bank_ai.agents.deps import get_deps
from lipari_bank_ai.agents.loop import run_agent
from lipari_bank_ai.agents.mcp_client import client_per, tools_dal_server
from lipari_bank_ai.agents.prompts import AGENT_SYSTEM
from lipari_bank_ai.agents.tools import build_tools_for
from lipari_bank_ai.auth.deps import UserContext
from lipari_bank_ai.auth.tokens import decode_token
from lipari_bank_ai.db.session import AsyncSessionLocal


async def main(token: str, domanda: str) -> None:
    payload = decode_token(token)                  # lo stesso token che riceverà il server
    user = UserContext(username=str(payload["sub"]), role=str(payload.get("role", "public")))
    async with AsyncSessionLocal() as db, client_per(token) as server:
        deps = get_deps(db)
        locali = [t for t in build_tools_for(user, deps) if t.name != "search_documents"]
        remoti = await tools_dal_server(server, nomi={"search_policy"})
        run = await run_agent(
            messaggi=[{"role": "system", "content": AGENT_SYSTEM},
                      {"role": "user", "content": domanda}],
            tools=locali + remoti,                 # il ciclo non sa quale vive altrove
            client=deps.openai, model=deps.model, runs=deps.runs, user=user,
        )
    print(f"{run.stopped_by} · passi {run.steps} · tool {run.tool_calls}\n\n{run.reply}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))