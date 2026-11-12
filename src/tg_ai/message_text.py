"""Extract readable text from regular and rich Telegram messages."""


def rich_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(filter(None, (rich_text(v) for v in value)))
    if isinstance(value, dict):
        parts = []
        for key in ('text', 'texts', 'blocks', 'items', 'cells', 'caption', 'credit', 'expression', 'code', 'title'):
            if key in value:
                parts.append(rich_text(value[key]))
        if value.get('type') in {'photo', 'video', 'animation', 'document'}:
            parts.append('[' + value['type'] + ']')
        return '\n'.join(filter(None, parts))
    return ''


def message_text(message):
    plain = message.text or message.caption
    if plain:
        return plain
    rich = getattr(message, 'rich_message', None)
    return rich_text(rich.model_dump(mode='json', exclude_none=True)) if rich else ''
