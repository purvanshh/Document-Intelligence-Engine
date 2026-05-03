from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level = (os.getenv("SANDBOX_EVAL_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


logger = logging.getLogger("sandbox_eval")


def log_auth_context(user_id: str | None, api_key: str | None) -> None:
    if user_id:
        logger.info("user=%s", user_id)
    if api_key:
        logger.debug("api_key=%s", api_key)

