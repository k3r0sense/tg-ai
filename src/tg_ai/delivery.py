"""Rich Markdown delivery, with fallback only after an explicit API rejection."""
import logging
import re
from markdown_it import MarkdownIt

from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.types import InputRichMessage, ReplyParameters

from .formatting import formatted_chunks

log = logging.getLogger(__name__)


def needs_rich_message(text: str) -> bool:
    tokens = MarkdownIt('commonmark').enable('table').parse(text)
    for token in tokens:
        if token.type in {'heading_open', 'table_open', 'fence', 'code_block',
                          'bullet_list_open', 'ordered_list_open', 'blockquote_open'}:
            return True
        if any(child.type == 'image' for child in (token.children or [])):
            return True
    return bool(re.search(r'\$\$.+?\$\$|(?<!\w)\$[^\n$]+\$', text, re.DOTALL))


async def deliver(bot, message, answer, *, spontaneous=False, rich_enabled=True):
    # Raw source length is conservative: do not split Markdown inside tables/fences.
    if rich_enabled and needs_rich_message(answer) and len(answer.encode('utf-8')) <= 32000:
        try:
            sent = await bot.send_rich_message(
                chat_id=message.chat.id,
                message_thread_id=message.message_thread_id,
                rich_message=InputRichMessage(markdown=answer),
                reply_parameters=None if spontaneous else ReplyParameters(
                    message_id=message.message_id, allow_sending_without_reply=True),
            )
        except (TelegramBadRequest, TelegramNotFound) as exc:
            log.warning('Rich delivery rejected: %s; using text fallback', type(exc).__name__)
        else:
            yield sent, answer
            return
    # Never retry ambiguous transport errors: the rich message may already be sent.
    for chunk, entities in formatted_chunks(answer):
        send = message.answer if spontaneous else message.reply
        sent = await send(chunk, entities=entities, parse_mode=None)
        yield sent, chunk
