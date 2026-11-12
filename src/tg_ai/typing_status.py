import asyncio
import logging
from contextlib import asynccontextmanager, suppress

log = logging.getLogger(__name__)


@asynccontextmanager
async def typing_status(bot, message, interval=4):
    async def refresh():
        while True:
            try:
                await asyncio.wait_for(bot.send_chat_action(
                    chat_id=message.chat.id,
                    message_thread_id=message.message_thread_id,
                    action='typing',
                ), timeout=5)
            except Exception as exc:
                # A status failure must not prevent the actual reply.
                log.debug('Typing status failed: %s', type(exc).__name__)
            await asyncio.sleep(interval)

    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
