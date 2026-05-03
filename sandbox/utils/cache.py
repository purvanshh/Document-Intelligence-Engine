from __future__ import annotations

import time
from typing import Any, Dict, Optional


class TinyTTLCache:
    def __init__(self, ttl_seconds: int = 30) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}

    def get(self, key: str, default: Any = None) -> Any:
        exp = self._expires.get(key)
        if exp and exp > time.time():
            return self._data.get(key, default)
        if key in self._data:
            self._data.pop(key, None)
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None, meta: Dict[str, Any] = {}) -> None:
        self._data[key] = value
        self._expires[key] = time.time() + (ttl or self.ttl_seconds)
        meta["last_key"] = key

