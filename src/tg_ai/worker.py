"""One model worker; intake stays responsive and direct requests take priority."""
import asyncio
import itertools
import logging
import time
from contextlib import AsyncExitStack

from .delivery import deliver
from .typing_status import typing_status
from .vision import image_attachment, image_data_url

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, bot, settings, store, memory, participation):
        self.bot, self.settings, self.store = bot, settings, store
        self.memory, self.participation = memory, participation
        self.queue = asyncio.PriorityQueue()
        self.counter = itertools.count()
        self.dirty = set()
        self.latest = {}
        self.completed = {}
        self.last_input = time.monotonic()
        self.last_summary = float('-inf')

    def submit(self, message, direct):
        self.last_input = time.monotonic()
        scope = (message.chat.id, message.message_thread_id or 0)
        self.latest[scope] = message.message_id
        self.queue.put_nowait((0 if direct else 1, next(self.counter), time.monotonic(), message, direct))

    async def run(self):
        while True:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=2)
            except TimeoutError:
                now = time.monotonic()
                if (self.dirty and now - self.last_input >= self.settings.summary_idle_seconds
                        and now - self.last_summary >= self.settings.summary_interval_seconds):
                    scope = self.dirty.pop()
                    started = time.monotonic()
                    self.last_summary = started
                    log.info('Background summary started: chat=%s', scope[0])
                    try:
                        cutoff = self.completed[scope]
                        await self.memory.compact(*scope, max_batches=1, through_message_id=cutoff)
                    except Exception as exc:
                        log.warning('Background summary failed: %s', type(exc).__name__)
                    log.info('Timing: stage=summary chat=%s elapsed=%.2fs', scope[0], time.monotonic()-started)
                continue
            _, _, queued, message, direct = job
            try:
                await self.process(message, direct, queued)
            except Exception as exc:
                log.error('Processing failed: chat=%s error=%s', message.chat.id, type(exc).__name__)
                if direct:
                    try:
                        await message.reply('Не получилось обработать запрос. Попробуй ещё раз.')
                    except Exception:
                        log.warning('Could not deliver error notice')
            finally:
                scope = (message.chat.id, message.message_thread_id or 0)
                self.completed[scope] = max(self.completed.get(scope, 0), message.message_id)
                self.dirty.add(scope)
                self.queue.task_done()

    async def process(self, message, direct, queued):
        started = time.monotonic()
        scope = (message.chat.id, message.message_thread_id or 0)
        log.info('Timing: stage=queue chat=%s elapsed=%.2fs', scope[0], started-queued)
        async with AsyncExitStack() as status:
            if direct:
                await status.enter_async_context(typing_status(self.bot, message))
            attachment = image_attachment(message)
            if attachment:
                vision_start = time.monotonic()
                description = 'Изображение не обработано: зрение отключено.'
                if self.settings.vision_enabled:
                    data_url = None
                    try:
                        data_url = await image_data_url(self.bot, attachment)
                        description = await self.memory.describe_image(data_url, message.caption or '')
                    except Exception as exc:
                        log.warning('Vision failed: %s', type(exc).__name__)
                        description = 'Не удалось прочитать изображение. Не делай предположений о содержимом.'
                    finally:
                        # Release base64 before starting text generation.
                        data_url = None
                await self.store.append_observation(scope[0], message.message_id, description)
                log.info('Timing: stage=vision chat=%s elapsed=%.2fs', scope[0], time.monotonic()-vision_start)
            mode = 'direct' if direct else 'silent'
            # Coalesce proactive decisions: only the newest queued human message is considered.
            if not direct and self.latest.get(scope) == message.message_id and self.participation.eligible(scope):
                decision_start = time.monotonic()
                decision = await self.memory.decide(*scope, self.participation.spontaneous(), message_id=message.message_id)
                mode = {'REPLY': 'reply', 'RANDOM': 'random'}.get(decision, 'silent')
                log.info('Timing: stage=decision chat=%s elapsed=%.2fs', scope[0], time.monotonic()-decision_start)
            log.info('Participation: chat_id=%s topic_id=%s mode=%s', *scope, mode)
            if mode != 'silent':
                if not direct:
                    await status.enter_async_context(typing_status(self.bot, message))
                answer_start = time.monotonic()
                answer = await self.memory.answer(*scope, mode=mode, message_id=message.message_id)
                log.info('Timing: stage=answer chat=%s elapsed=%.2fs', scope[0], time.monotonic()-answer_start)
                async for sent, stored_text in deliver(self.bot, message, answer,
                    spontaneous=mode == 'random', rich_enabled=self.settings.rich_messages_enabled):
                    self.participation.sent(scope)
                    await self.store.add(sent, text_override=stored_text)
                    self.completed[scope] = max(self.completed.get(scope, 0), sent.message_id)
        log.info('Timing: stage=total chat=%s elapsed=%.2fs', scope[0], time.monotonic()-started)
