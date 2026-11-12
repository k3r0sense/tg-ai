import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from tg_ai.typing_status import typing_status


class TypingTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_and_cleanup(self):
        event = asyncio.Event()
        calls = []
        async def send(**kwargs):
            calls.append(kwargs)
            if len(calls) >= 2:
                event.set()
        bot = SimpleNamespace(send_chat_action=send)
        message = SimpleNamespace(chat=SimpleNamespace(id=123), message_thread_id=7)
        async with typing_status(bot, message, interval=0.001):
            await asyncio.wait_for(event.wait(), 1)
        count = len(calls)
        await asyncio.sleep(0.01)
        self.assertEqual(len(calls), count)
        self.assertEqual(calls[0], {'chat_id':123, 'message_thread_id':7, 'action':'typing'})

    async def test_status_error_does_not_break_reply(self):
        bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=RuntimeError))
        message = SimpleNamespace(chat=SimpleNamespace(id=123), message_thread_id=None)
        async with typing_status(bot, message, interval=0.001):
            await asyncio.sleep(0.01)
        bot.send_chat_action.assert_awaited()
