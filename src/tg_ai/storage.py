import json
import asyncio
from functools import wraps
from pathlib import Path

import aiosqlite
from .message_text import message_text


def serialized_write(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        async with self.write_lock:
            return await method(self, *args, **kwargs)
    return wrapped


def clip(text: str, limit: int) -> str:
    return text.encode('utf-8')[:limit].decode('utf-8', errors='ignore')


def render(row: dict) -> str:
    return json.dumps({k: row[k] for k in ('message_id', 'author', 'created', 'reply_id', 'is_bot', 'text')}, ensure_ascii=False)


class Store:
    async def open(self, path: str):
        self.write_lock = asyncio.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(path)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, chat INTEGER, topic INTEGER, message_id INTEGER,
                author TEXT, created TEXT, reply_id INTEGER, is_bot INTEGER, text TEXT,
                UNIQUE(chat, message_id));
            CREATE INDEX IF NOT EXISTS scope ON messages(chat, topic, id);
            CREATE TABLE IF NOT EXISTS summaries (
                chat INTEGER, topic INTEGER, through_id INTEGER, text TEXT,
                PRIMARY KEY(chat, topic));
            CREATE TABLE IF NOT EXISTS summary_versions (
                chat INTEGER, topic INTEGER, through_id INTEGER, text TEXT,
                created TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        async with self.db.execute('PRAGMA table_info(messages)') as cur:
            columns = {row['name'] for row in await cur.fetchall()}
        if 'reply_context' not in columns:
            await self.db.execute('ALTER TABLE messages ADD COLUMN reply_context TEXT')
        await self.db.commit()

    @serialized_write
    async def add(self, message, text_override: str | None = None) -> bool:
        user = message.from_user
        author = f'{user.full_name} [user:{user.id}]' if user else str(message.sender_chat.id if message.sender_chat else 'unknown')
        reply = message.reply_to_message
        reply_context = None
        if reply:
            reply_user = reply.from_user
            reply_context = json.dumps({
                'author': f'{reply_user.full_name} [user:{reply_user.id}]' if reply_user else 'unknown',
                'text': message_text(reply) or '[нет текстового содержимого]',
            }, ensure_ascii=False)
        cursor = await self.db.execute(
            'INSERT OR IGNORE INTO messages(chat,topic,message_id,author,created,reply_id,is_bot,text,reply_context) VALUES(?,?,?,?,?,?,?,?,?)',
            (message.chat.id, message.message_thread_id or 0, message.message_id,
             author, message.date.isoformat(),
             message.reply_to_message.message_id if message.reply_to_message else None,
             int(bool(user and user.is_bot)), message_text(message) if text_override is None else text_override, reply_context),
        )
        await self.db.commit()
        return cursor.rowcount == 1

    async def reply_context(self, row: dict) -> str:
        if row.get('reply_context'):
            return row['reply_context']
        if row['reply_id'] is None:
            return ''
        async with self.db.execute(
            'SELECT author,text FROM messages WHERE chat=? AND topic=? AND message_id=?',
            (row['chat'], row['topic'], row['reply_id']),
        ) as cur:
            target = await cur.fetchone()
        return json.dumps(dict(target), ensure_ascii=False) if target else '[исходное сообщение недоступно]'

    @serialized_write
    async def append_observation(self, chat: int, message_id: int, text: str):
        await self.db.execute(
            "UPDATE messages SET text=text || ? WHERE chat=? AND message_id=?",
            ('\n[Описание изображения моделью, возможны ошибки]: ' + clip(text, 4000), chat, message_id),
        )
        await self.db.commit()

    async def state(self, chat: int, topic: int, through_message_id: int | None = None):
        async with self.db.execute('SELECT * FROM summaries WHERE chat=? AND topic=?', (chat, topic)) as cur:
            summary = await cur.fetchone()
        through = summary['through_id'] if summary else 0
        async with self.db.execute('SELECT * FROM messages WHERE chat=? AND topic=? AND id>? AND (? IS NULL OR message_id<=?) ORDER BY id', (chat, topic, through, through_message_id, through_message_id)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        return summary['text'] if summary else '', rows

    @serialized_write
    async def save_summary(self, chat: int, topic: int, through: int, text: str):
        try:
            await self.db.execute('INSERT INTO summary_versions(chat,topic,through_id,text) VALUES(?,?,?,?)', (chat, topic, through, text))
            await self.db.execute('INSERT OR REPLACE INTO summaries VALUES(?,?,?,?)', (chat, topic, through, text))
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise
