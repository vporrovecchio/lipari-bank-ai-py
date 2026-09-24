from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.repos import ChatRepository
from lipari_bank_ai.exceptions import ChatSessionNotFoundError
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.types.chat import ChatRequest, ChatResponse


class ChatService:
    def __init__(self, session: AsyncSession, llm: LLMProvider, system_prompt: str) -> None:
        self.session = session
        self.repo = ChatRepository(session)
        self.llm = llm
        self.system_prompt = system_prompt

    async def chat(self, req: ChatRequest, user_id: str) -> ChatResponse:
        if req.session_id != "new":
            chat = await self.repo.find_session(req.session_id)
            if not chat:
                raise ChatSessionNotFoundError(req.session_id)
        else:
            chat = await self.repo.create_session(user_id=user_id)

        # Build history
        history_messages = await self.repo.list_messages(chat.id)
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        for m in history_messages:
            messages.append(Message(role=m.role, content=m.content))
        messages.append(Message(role="user", content=req.message))

        # Save user message
        await self.repo.add_message(chat.id, "user", req.message)

        # Call LLM
        llm_response = await self.llm.complete(messages, max_tokens=500)

        # Save assistant message
        await self.repo.add_message(
            chat.id, "assistant", llm_response.content,
            tokens=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
            model_used=llm_response.model,
        )

        await self.session.commit()

        return ChatResponse(
            session_id=chat.id,
            reply=llm_response.content,
            tokens_used=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
            model_used=llm_response.model,
            created_at=datetime.now(UTC),
        )