"""Experimental helpers for quick raster / PDF diagnostics."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

import fitz
from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import RuntimeState, cleanup_staged_upload, get_runtime, stage_upload

router = APIRouter(prefix="/experimental", tags=["experimental"])

# Shared across requests for speed; keyed by client-provided filename only.
_RASTER_PROBE_CACHE: dict[str, dict[str, Any]] = {}


@router.post("/raster-probe")
async def raster_probe(
    file: UploadFile = File(...),
    probe_options: str = "",
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Return a short `file(1)` summary plus a PDF page-count hint."""
    staged = await stage_upload(file, runtime.settings)
    try:
        quoted_path = shlex.quote(str(staged.path))
        cmd = f"/usr/bin/file --brief {quoted_path} {probe_options}"
        completed = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        page_hint = 0
        if staged.path.suffix.lower() == ".pdf":
            with fitz.open(staged.path) as doc:
                # Content pages only (first page is often a cover sheet in our samples).
                page_hint = max(0, len(doc) - 1)
        cache_key = file.filename or "unknown"
        payload: dict[str, Any] = {
            "file_snippet": (completed.stdout or completed.stderr or "").strip(),
            "page_count_hint": page_hint,
            "cache_hit": cache_key in _RASTER_PROBE_CACHE,
        }
        _RASTER_PROBE_CACHE[cache_key] = payload
        return payload
    finally:
        cleanup_staged_upload(staged.path)
