"""Session-scoped episode records and consented JPEGs. No video storage."""

import json
from contextlib import contextmanager
import sqlite3
from pathlib import Path


class EpisodeStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, driver TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT,
                    screenshot BLOB, screenshot_claimed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS jobs (
                    episode_id TEXT PRIMARY KEY, owner TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL, last_error TEXT);
            """
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def start(self, episode, owner, driver, screenshot=None, consent=False):
        # The primary key makes creation idempotent. A later consent change cannot add another image.
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO episodes VALUES (?, ?, ?, ?, NULL, ?, 1)",
                (
                    episode.id,
                    owner,
                    driver,
                    episode.started_at.isoformat(),
                    screenshot if consent else None,
                ),
            )

    def end(self, episode, owner):
        with self.connect() as db:
            db.execute(
                "UPDATE episodes SET ended_at=? WHERE id=? AND owner=?",
                (episode.ended_at.isoformat(), episode.id, owner),
            )

    def episodes(self, owner):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,started_at,ended_at FROM episodes WHERE owner=? ORDER BY started_at",
                    (owner,),
                )
            ]

    def screenshots(self, owner):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,started_at,screenshot FROM episodes WHERE owner=? AND screenshot IS NOT NULL ORDER BY started_at",
                    (owner,),
                )
            ]

    def delete_screenshot(self, owner, episode_id):
        with self.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute(
                "UPDATE episodes SET screenshot=NULL WHERE owner=? AND id=?",
                (owner, episode_id),
            )

    def delete_all_screenshots(self, owner):
        with self.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("UPDATE episodes SET screenshot=NULL WHERE owner=?", (owner,))

    def queue(self, owner, episode_id, payload, now):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, ?, 0, ?, NULL)",
                (episode_id, owner, json.dumps(payload), "queued", now),
            )

    def due_job(self, now):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','retrying') AND next_attempt<=? ORDER BY next_attempt LIMIT 1",
                (now,),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE jobs SET status='sending' WHERE episode_id=?",
                    (row["episode_id"],),
                )
            return dict(row) if row else None

    def has_pending(self):
        with self.connect() as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM jobs WHERE status IN ('queued','retrying') LIMIT 1"
                ).fetchone()
            )

    def job_result(self, episode_id, status, attempts, next_attempt, error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,attempts=?,next_attempt=?,last_error=? WHERE episode_id=?",
                (status, attempts, next_attempt, error, episode_id),
            )

    def jobs(self, owner):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT episode_id,status,attempts,last_error FROM jobs WHERE owner=?",
                    (owner,),
                )
            ]
