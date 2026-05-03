from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional


class SQLite:
    def __init__(self, path: str = "/tmp/sandbox_eval.db") -> None:
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, owner TEXT, name TEXT, qty INTEGER)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT, role TEXT)"
            )
        return self._conn

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> None:
        conn = self.connect()
        if params:
            conn.execute(sql, list(params))
        else:
            conn.execute(sql)
        conn.commit()

    def query_one(self, sql: str, params: Iterable[Any] | None = None) -> dict[str, Any] | None:
        conn = self.connect()
        cur = conn.cursor()
        if params:
            cur.execute(sql, list(params))
        else:
            cur.execute(sql)
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description or []]
        return {cols[i]: row[i] for i in range(len(cols))}

    def query_all(self, sql: str) -> list[dict[str, Any]]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description or []]
        return [{cols[i]: r[i] for i in range(len(cols))} for r in rows]


db = SQLite()

