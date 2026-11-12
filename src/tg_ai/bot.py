import asyncio
import logging
import os
from contextlib import suppress

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType

from .config import Settings
from .memory import Memory
from .participation import Participation
from .storage import Store
from .worker import Worker

log = logging.getLogger(__name__)


def addressed(message, bot_id: int, username: str) -> bool:
    if message.chat.type == ChatType.PRIVATE:
        return True
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == bot_id:
        return True
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []
    for entity in entities:
        if entity.type == "mention":
            value = text.encode("utf-16-le")[
                entity.offset * 2 : (entity.offset + entity.length) * 2
            ].decode("utf-16-le")
            if value.casefold() == f"@{username}".casefold():
                return True
        if entity.type == "text_mention" and entity.user and entity.user.id == bot_id:
            return True
    return False


async def run():
    # Never export chat content through inherited LangSmith tracing settings.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    store = Store()
    await store.open(settings.database_path)
    memory = Memory(settings, store)
    participation = Participation(settings)
    bot = Bot(settings.telegram_bot_token.get_secret_value())
    dp = Dispatcher()
    worker = Worker(bot, settings, store, memory, participation)
    worker_task = None

    try:
        me = await bot.get_me()
        worker_task = asyncio.create_task(worker.run())

        @dp.message(F.text | F.caption | F.photo | F.document | F.rich_message)
        async def receive(message):
            allowed = message.chat.id in settings.allowed_chat_ids
            log.info(
                "Incoming message: chat_id=%s chat_type=%s topic_id=%s message_id=%s allowed=%s",
                message.chat.id,
                message.chat.type,
                message.message_thread_id or 0,
                message.message_id,
                allowed,
            )
            if not allowed:
                return
            if message.from_user and message.from_user.is_bot:
                return
            # Intake only persists and enqueues; it never waits for the model.
            if not await store.add(message):
                return
            scope = (message.chat.id, message.message_thread_id or 0)
            participation.observe(scope)
            worker.submit(message, addressed(message, me.id, me.username or ""))

        log.info(
            "Bot @%s started; allowed chats: %s", me.username, settings.allowed_chat_ids
        )
        await dp.start_polling(bot, handle_as_tasks=False, allowed_updates=["message"])
    finally:
        if worker_task is not None:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        await bot.session.close()
        await store.db.close()
