"""SQLite session history. Each worker uses its own short-lived connection."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid
from src.subtitles import Caption


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created TEXT NOT NULL, name TEXT NOT NULL,
                    mode TEXT NOT NULL, source TEXT, target TEXT NOT NULL,
                    model TEXT NOT NULL, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS captions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    start REAL NOT NULL, end REAL NOT NULL, original TEXT NOT NULL,
                    translated TEXT NOT NULL, language TEXT NOT NULL,
                    speaker_id TEXT
                );
                CREATE INDEX IF NOT EXISTS captions_session ON captions(session_id, id);
            ''')
            try:
                db.execute('ALTER TABLE captions ADD COLUMN speaker_id TEXT')
            except Exception:
                pass

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, name, mode, source, target, model):
        session = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)',
                (session, datetime.now(timezone.utc).isoformat(), name, mode, source, target, model, 'en curso'))
        return session

    def add(self, session, caption):
        speaker = getattr(caption, 'speaker_id', None)
        with self.connection() as db:
            db.execute('INSERT INTO captions(session_id,start,end,original,translated,language,speaker_id) VALUES (?,?,?,?,?,?,?)',
                (session, caption.start, caption.end, caption.original, caption.translated, caption.language, speaker))

    def finish(self, session, state):
        with self.connection() as db:
            db.execute('UPDATE sessions SET state=? WHERE id=?', (state, session))

    def search(self, query=''):
        with self.connection() as db:
            return [dict(row) for row in db.execute('''
                SELECT s.*, (SELECT COUNT(*) FROM captions c WHERE c.session_id=s.id) AS count
                FROM sessions s WHERE ?='' OR instr(lower(s.name),lower(?))>0 OR EXISTS (
                    SELECT 1 FROM captions c WHERE c.session_id=s.id AND
                    (instr(lower(c.original),lower(?))>0 OR instr(lower(c.translated),lower(?))>0)
                ) ORDER BY created DESC LIMIT 200
            ''', (query,query,query,query))]

    def captions(self, session, limit=None):
        with self.connection() as db:
            rows = db.execute('SELECT start,end,original,translated,language,speaker_id FROM captions WHERE session_id=? ORDER BY id LIMIT ?',
                (session, limit if limit is not None else -1))
            return [Caption(**dict(row)) for row in rows]
