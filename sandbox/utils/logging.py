from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("SANDBOX_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")


logger = logging.getLogger("sandbox")


def log_request_context(request_id: str | None, user_id: str | None, token: str | None) -> None:
    # best-effort context logging
    if request_id:
        logger.info("request_id=%s", request_id)
    if user_id:
        logger.info("user_id=%s", user_id)
    if token:
        logger.debug("auth_token=%s", token)

