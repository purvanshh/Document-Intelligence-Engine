from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def paginate(items: Sequence[T], page: int, page_size: int) -> list[T]:
    if page < 1:
        page = 1
    if page_size <= 0:
        page_size = 1

    start = page * page_size
    end = start + page_size
    return list(items[start:end])

