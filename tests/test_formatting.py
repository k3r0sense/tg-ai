import unittest
from tg_ai.formatting import formatted_chunks

class FormattingTests(unittest.TestCase):
    def test_bold_emoji_link(self):
        text, entities = list(formatted_chunks('😀 **привет** [сайт](https://example.com)'))[0]
        self.assertEqual(text, '😀 привет сайт')
        self.assertEqual(entities[0].offset, 3)
        self.assertEqual(entities[0].type, 'bold')
        self.assertEqual(entities[1].url, 'https://example.com')

    def test_code_split(self):
        chunks = list(formatted_chunks('```python\n' + '😀' * 20 + '\n```', limit=12))
        self.assertGreater(len(chunks), 1)
        for text, entities in chunks:
            self.assertLessEqual(len(text.encode('utf-16-le')) // 2, 12)
            self.assertTrue(all(e.offset + e.length <= len(text.encode('utf-16-le')) // 2 for e in entities))
            self.assertEqual(entities[0].type, 'pre')

    def test_plain_special_characters(self):
        text, _ = list(formatted_chunks('Цена 1.5! a_b <tag> & ещё'))[0]
        self.assertEqual(text, 'Цена 1.5! a_b <tag> & ещё')
