from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import ChatSession, ChatMessage


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, user_id: str) -> ChatSession:
        chat = ChatSession(user_id=user_id)
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def find_session(self, session_id: str) -> ChatSession | None:
        stmt = select(ChatSession).where(ChatSession.id == session_id).options(
            selectinload(ChatSession.messages)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        cost_eur: float = 0.0,
        model_used: str | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tokens=tokens,
            cost_eur=cost_eur,
            model_used=model_used,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def list_messages(self, session_id: str) -> list[ChatMessage]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())