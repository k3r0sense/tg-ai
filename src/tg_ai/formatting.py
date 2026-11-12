"""Convert common Markdown into Telegram text/entities, with UTF-16-safe chunks."""
from urllib.parse import urlsplit

from aiogram.types import MessageEntity
from markdown_it import MarkdownIt


def formatted_chunks(markdown: str, limit: int = 3500):
    text = ''
    spans = []
    stack = []

    def emit(value):
        nonlocal text
        text += value

    def walk(tokens):
        for token in tokens:
            kind = token.type
            if kind == 'inline':
                walk(token.children or [])
            elif kind in {'text', 'html_inline', 'html_block'}:
                emit(token.content)
            elif kind in {'softbreak', 'hardbreak'}:
                emit('\n')
            elif kind in {'fence', 'code_block', 'code_inline'}:
                start = len(text)
                emit(token.content)
                spans.append((start, len(text), 'code' if kind == 'code_inline' else 'pre',
                              {'language': token.info.split()[0]} if kind == 'fence' and token.info.strip() else {}))
                if kind != 'code_inline':
                    emit('\n')
            elif kind in {'strong_open', 'em_open', 's_open', 'link_open', 'heading_open'}:
                entity = {'strong_open':'bold', 'em_open':'italic', 's_open':'strikethrough',
                          'link_open':'text_link', 'heading_open':'bold'}[kind]
                extra = {}
                if kind == 'link_open':
                    url = token.attrGet('href') or ''
                    if urlsplit(url).scheme in {'http', 'https'}:
                        extra = {'url': url}
                    else:
                        entity = None
                stack.append((len(text), entity, extra))
            elif kind in {'strong_close', 'em_close', 's_close', 'link_close', 'heading_close'}:
                start, entity, extra = stack.pop()
                if entity and len(text) > start:
                    spans.append((start, len(text), entity, extra))
                if kind == 'heading_close':
                    emit('\n')
            elif kind == 'image':
                emit(token.content + ' (' + (token.attrGet('src') or '') + ')')
            elif kind == 'list_item_open':
                emit('• ')
            elif kind in {'paragraph_close', 'list_item_close', 'blockquote_close'}:
                if not text.endswith('\n'):
                    emit('\n')
            elif kind == 'hr':
                emit('———\n')

    walk(MarkdownIt('commonmark', {'html': False}).enable('strikethrough').parse(markdown))
    text = text.rstrip('\n')
    if not text:
        text = markdown or '…'
    start = 0
    while start < len(text):
        end, units = start, 0
        while end < len(text):
            cost = 2 if ord(text[end]) > 0xFFFF else 1
            if units + cost > limit:
                break
            units += cost
            end += 1
        chunk = text[start:end]
        entities = []
        for left, right, kind, extra in spans:
            a, b = max(left, start), min(right, end)
            if a < b:
                # Telegram does not allow bold/italic/link entities inside code.
                if kind not in {'pre', 'code'} and any(
                    k in {'pre', 'code'} and l < b and r > a for l, r, k, _ in spans
                ):
                    continue
                entities.append(MessageEntity(type=kind,
                    offset=len(text[start:a].encode('utf-16-le')) // 2,
                    length=len(text[a:b].encode('utf-16-le')) // 2, **extra))
        yield chunk, sorted(entities, key=lambda e: (e.offset, -e.length))
        start = end
