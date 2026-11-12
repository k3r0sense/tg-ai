import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendRichMessage
from tg_ai.delivery import deliver
from tg_ai.message_text import rich_text


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setup_mocks(self):
        message = SimpleNamespace(chat=SimpleNamespace(id=-123), message_thread_id=7,
            message_id=42, reply=AsyncMock(return_value='plain'), answer=AsyncMock(return_value='plain'))
        return SimpleNamespace(send_rich_message=AsyncMock(return_value='rich')), message

    async def test_rich_roundtrip_and_media(self):
        bot, msg = self.setup_mocks()
        source = '# Заголовок\n\n| A | B |\n|---|---|\n|1|2|\n\n![Фото](https://example.com/image.png)'
        result = [item async for item in deliver(bot, msg, source)]
        self.assertEqual(result, [('rich', source)])
        args = bot.send_rich_message.call_args.kwargs
        self.assertEqual(args['rich_message'].markdown, source)
        self.assertEqual(args['message_thread_id'], 7)
        self.assertEqual(args['reply_parameters'].message_id, 42)
        msg.reply.assert_not_called()

    async def test_rejection_falls_back(self):
        bot, msg = self.setup_mocks()
        bot.send_rich_message.side_effect = TelegramBadRequest(method=SendRichMessage,
                                                               message='cannot parse rich message')
        result = [item async for item in deliver(bot, msg, '# Привет')]
        self.assertEqual(result, [('plain', 'Привет')])
        self.assertEqual(msg.reply.call_args.kwargs['entities'][0].type, 'bold')

    async def test_network_failure_does_not_duplicate(self):
        bot, msg = self.setup_mocks()
        bot.send_rich_message.side_effect = TelegramNetworkError(method=SendRichMessage, message='timeout')
        with self.assertRaises(TelegramNetworkError):
            _ = [item async for item in deliver(bot, msg, '# hello')]
        msg.reply.assert_not_called()

    async def test_spontaneous_and_disabled(self):
        bot, msg = self.setup_mocks()
        _ = [item async for item in deliver(bot, msg, '# hello', spontaneous=True)]
        self.assertIsNone(bot.send_rich_message.call_args.kwargs['reply_parameters'])
        bot.send_rich_message.reset_mock()
        _ = [item async for item in deliver(bot, msg, 'hello', spontaneous=True, rich_enabled=False)]
        bot.send_rich_message.assert_not_called()
        msg.answer.assert_awaited_once()

    def test_incoming_rich_text(self):
        self.assertEqual(rich_text({'blocks':[{'type':'paragraph','text':'Привет'},
            {'type':'table','cells':[[{'text':'Ячейка'}]]}]}), 'Привет\nЯчейка')

    async def test_plain_reply_uses_normal_message_without_bold(self):
        bot, msg = self.setup_mocks()
        _ = [item async for item in deliver(bot, msg, 'Обычная реплика.')]
        bot.send_rich_message.assert_not_called()
        self.assertEqual(msg.reply.call_args.kwargs['entities'], [])

    async def test_only_selected_words_bold(self):
        bot, msg = self.setup_mocks()
        _ = [item async for item in deliver(bot, msg, 'Это **важно**, остальное обычно.')]
        bot.send_rich_message.assert_not_called()
        entities = msg.reply.call_args.kwargs['entities']
        self.assertEqual(len(entities),1)
        self.assertEqual((entities[0].offset,entities[0].length),(4,5))
