import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tg_ai.config import Settings
from tg_ai.memory import Memory
from tg_ai.storage import Store
from tg_ai.bot import addressed


def message(number, chat=1, text='hello'):
    return SimpleNamespace(chat=SimpleNamespace(id=chat, type='group'), message_thread_id=None,
        message_id=number, from_user=SimpleNamespace(id=5, full_name='Test', is_bot=False),
        sender_chat=None, date=datetime.now(timezone.utc), reply_to_message=None,
        text=text, caption=None, entities=[], caption_entities=[])


class Chain:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError('offline')
        return 'Краткая сводка'


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store()
        await self.store.open(str(Path(self.tmp.name) / 'test.db'))
        self.settings = Settings(_env_file=None, telegram_bot_token='test', allowed_chat_ids=[1],
                                 recent_bytes=1000, batch_bytes=1000)

    async def asyncTearDown(self):
        await self.store.db.close()
        self.tmp.cleanup()

    async def test_deduplication_and_scope(self):
        self.assertTrue(await self.store.add(message(1)))
        self.assertFalse(await self.store.add(message(1)))
        await self.store.add(message(1, chat=2))
        _, rows = await self.store.state(1, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(await self.store.state(1, 99), ('', []))

    async def test_compaction_preserves_archive_and_latest(self):
        for i in range(8):
            await self.store.add(message(i, text='текст ' * 80))
        chain = Chain()
        memory = Memory(self.settings, self.store, chain)
        await memory.compact(1, 0)
        summary, rows = await self.store.state(1, 0)
        self.assertTrue(summary)
        self.assertEqual(rows[-1]['message_id'], 7)
        self.assertLess(len(rows), 8)
        async with self.store.db.execute('SELECT count(*) FROM messages') as cur:
            self.assertEqual((await cur.fetchone())[0], 8)
        before = len(chain.calls)
        await memory.compact(1, 0)
        self.assertEqual(len(chain.calls), before)

    async def test_failure_does_not_advance_summary(self):
        for i in range(3):
            await self.store.add(message(i, text='x' * 800))
        with self.assertRaises(RuntimeError):
            await Memory(self.settings, self.store, Chain(True)).compact(1, 0)
        summary, rows = await self.store.state(1, 0)
        self.assertEqual(summary, '')
        self.assertEqual(len(rows), 3)

    def test_mention_after_emoji(self):
        msg = message(1, text='😀 @testbot привет')
        msg.entities = [SimpleNamespace(type='mention', offset=3, length=8)]
        self.assertTrue(addressed(msg, 123, 'testbot'))
        self.assertFalse(addressed(msg, 123, 'anotherbot'))

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            Settings(_env_file=None, telegram_bot_token='test', allowed_chat_ids=[1],
                     llm_base_url='https://example.com/v1')

    async def test_confirmation_keeps_assistant_role_and_reply(self):
        question = message(1, text='Объясни слово')
        clarification = message(2, text='Ты спрашиваешь значение слова?')
        clarification.from_user.is_bot = True
        confirmation = message(3, text='да именно так')
        confirmation.reply_to_message = clarification
        for item in (question, clarification, confirmation):
            await self.store.add(item)
        chain = Chain()
        await Memory(self.settings, self.store, chain).answer(1, 0)
        turns = chain.calls[-1]
        self.assertEqual([turn.type for turn in turns], ['system', 'human', 'ai', 'human'])
        self.assertIn('да именно так', turns[-1].content)
        self.assertIn('Ты спрашиваешь значение слова?', turns[-1].content)
        self.assertIn('не повторяй', turns[0].content)

    async def test_reply_outside_archive_survives(self):
        current = message(10, text='да')
        current.reply_to_message = message(9, text='Внешнее сообщение')
        await self.store.add(current)
        chain = Chain()
        await Memory(self.settings, self.store, chain).answer(1, 0)
        self.assertIn('Внешнее сообщение', chain.calls[-1][-1].content)
        self.assertEqual(len((await self.store.state(1, 0))[1]), 1)

    async def test_legacy_reply_lookup_is_scoped(self):
        await self.store.add(message(1, chat=2, text='Чужой секрет'))
        current = message(2)
        await self.store.add(current)
        await self.store.db.execute('UPDATE messages SET reply_id=1 WHERE chat=1')
        await self.store.db.commit()
        chain = Chain()
        await Memory(self.settings, self.store, chain).answer(1, 0)
        self.assertNotIn('Чужой секрет', chain.calls[-1][-1].content)
        self.assertIn('недоступно', chain.calls[-1][-1].content)

    async def test_vision_input_is_multimodal_and_stays_local_chain(self):
        chain = Chain()
        memory = Memory(self.settings, self.store, chain)
        await memory.describe_image('data:image/jpeg;base64,AA==', 'мем')
        blocks = chain.calls[-1][-1].content
        self.assertEqual(blocks[1]['type'], 'image_url')
        await self.store.add(message(1))
        await self.store.append_observation(1, 1, 'Описание мема')
        self.assertIn('Описание мема', (await self.store.state(1, 0))[1][0]['text'])

    async def test_invalid_decision_silences_bot(self):
        await self.store.add(message(1))
        self.assertEqual(await Memory(self.settings, self.store, Chain()).decide(1, 0, True), 'SILENT')

    async def test_outgoing_rich_markdown_stored(self):
        item = message(1, text='')
        item.from_user.is_bot = True
        await self.store.add(item, text_override='# Ответ\n\n**Важно**')
        rows = (await self.store.state(1, 0))[1]
        self.assertEqual(rows[0]['text'], '# Ответ\n\n**Важно**')

    async def test_answer_uses_requested_message_not_newer_intake(self):
        await self.store.add(message(1,text='Первый вопрос'))
        await self.store.add(message(2,text='Новый вопрос'))
        chain = Chain()
        await Memory(self.settings,self.store,chain).answer(1,0,message_id=1)
        self.assertIn('Первый вопрос',chain.calls[-1][-1].content)
        self.assertNotIn('Новый вопрос',chain.calls[-1][-1].content)

    async def test_background_compaction_leaves_new_arrivals(self):
        for i in range(1,6):
            await self.store.add(message(i,text='x'*800))
        memory = Memory(self.settings,self.store,Chain())
        await memory.compact(1,0,max_batches=1,through_message_id=3)
        _, rows = await self.store.state(1,0)
        self.assertIn(4,[r['message_id'] for r in rows])
        self.assertIn(5,[r['message_id'] for r in rows])
