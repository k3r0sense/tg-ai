import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from tg_ai.worker import Worker


def msg(number=1):
    return SimpleNamespace(chat=SimpleNamespace(id=1),message_thread_id=None,message_id=number,
                           photo=[],document=None,caption=None)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setup_worker(self):
        settings = SimpleNamespace(vision_enabled=True,rich_messages_enabled=True)
        store = SimpleNamespace(append_observation=AsyncMock(),add=AsyncMock())
        memory = SimpleNamespace(describe_image=AsyncMock(return_value='фото'),
            compact=AsyncMock(),answer=AsyncMock(return_value='ответ'),decide=AsyncMock(return_value='SILENT'))
        participation = SimpleNamespace(eligible=lambda scope: False,sent=lambda scope: None)
        return Worker(SimpleNamespace(),settings,store,memory,participation)

    async def test_direct_priority_fifo(self):
        w = self.setup_worker()
        w.submit(msg(1),False)
        w.submit(msg(2),True)
        w.submit(msg(3),True)
        self.assertEqual([(await w.queue.get())[3].message_id for _ in range(3)], [2,3,1])

    async def test_plain_text_never_runs_vision_or_compaction(self):
        w = self.setup_worker()
        await w.process(msg(),False,time.monotonic())
        w.memory.describe_image.assert_not_called()
        w.memory.compact.assert_not_called()
        w.store.append_observation.assert_not_called()

    async def test_image_only_runs_vision_once(self):
        w = self.setup_worker()
        message = msg()
        message.photo = [SimpleNamespace(file_id='photo')]
        with patch('tg_ai.worker.image_data_url', new=AsyncMock(return_value='data:image/jpeg;base64,AA==')):
            await w.process(message,False,time.monotonic())
        w.memory.describe_image.assert_awaited_once()
        w.memory.compact.assert_not_called()
        w.store.append_observation.assert_awaited_once()

    async def test_summary_does_not_requeue_itself(self):
        import asyncio
        from contextlib import suppress
        w = self.setup_worker()
        w.settings.summary_idle_seconds = 0
        w.settings.summary_interval_seconds = 0
        w.memory.compact.return_value = True
        w.dirty.add((1,0))
        w.completed[(1,0)] = 1
        task = asyncio.create_task(w.run())
        try:
            await asyncio.sleep(4.2)
            w.memory.compact.assert_awaited_once()
            self.assertFalse(w.dirty)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
