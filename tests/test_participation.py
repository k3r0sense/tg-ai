import unittest
from types import SimpleNamespace
from tg_ai.participation import Participation
from tg_ai.config import Settings
from tg_ai.vision import image_attachment


class ParticipationTests(unittest.TestCase):
    def test_cooldown_count_and_scope(self):
        now = [100.0]
        settings = Settings(_env_file=None, telegram_bot_token='test', allowed_chat_ids=[1])
        policy = Participation(settings, clock=lambda: now[0])
        for _ in range(3):
            policy.observe((1, 0))
        self.assertTrue(policy.eligible((1, 0)))
        self.assertFalse(policy.eligible((1, 0)))
        policy.sent((1, 0))
        for _ in range(3):
            policy.observe((1, 0))
        now[0] += 40
        self.assertFalse(policy.eligible((1, 0)))
        now[0] += 60
        self.assertTrue(policy.eligible((1, 0)))
        self.assertFalse(policy.eligible((2, 0)))

    def test_disabled(self):
        settings = Settings(_env_file=None, telegram_bot_token='test', allowed_chat_ids=[1], proactive_enabled=False)
        policy = Participation(settings)
        for _ in range(10):
            policy.observe((1, 0))
        self.assertFalse(policy.eligible((1, 0)))

    def test_photo_and_document(self):
        self.assertEqual(image_attachment(SimpleNamespace(photo=['small','large'], document=None)), 'large')
        document = SimpleNamespace(mime_type='image/png')
        self.assertIs(image_attachment(SimpleNamespace(photo=[], document=document)), document)
        document.mime_type = 'application/pdf'
        self.assertIsNone(image_attachment(SimpleNamespace(photo=[], document=document)))
