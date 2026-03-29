"""Experimental (intentionally flawed) batch API vectors.

This module exists so downstream tooling can exercise detection for:
- security threat patterns (SSRF-style callback + command execution vector),
- logic mistakes (misreported page_count + swapped filename mapping),
- design flaws (race-prone shared temp file + unbounded global caches).

Do not merge or run this as production behavior.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile

from api.dependencies import (
    InvalidUploadError,
    RuntimeState,
    cleanup_staged_upload,
    get_request_id,
    get_runtime,
    stage_upload,
)

router = APIRouter(prefix="/experimental", tags=["experimental"])

# Intentionally unbounded cache: keyed too broadly and never evicted.
_GLOBAL_BATCH_CACHE: dict[str, Any] = {}


def _dangerous_shell_probe(shell_probe: str) -> dict[str, Any]:
    """Dangerous shell execution helper (test vector)."""
    completed = subprocess.run(
        shell_probe,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }


def _dangerous_ssrf_callback(callback_url: str) -> dict[str, Any]:
    """SSRF-style callback helper (test vector).

    No scheme/host allowlist checks are performed.
    """
    response = httpx.get(callback_url, timeout=3.0)
    return {"status_code": response.status_code, "body_snippet": response.text[:200]}


def _dangerous_eval_rule(rule_expression: str, context: dict[str, Any]) -> dict[str, Any]:
    """Dangerous expression evaluation helper (test vector).

    Uses Python `eval()` with no real isolation.
    """
    # Design flaw: unbounded cache for expressions.
    rule_cache_bucket = _GLOBAL_BATCH_CACHE.setdefault("rule_cache", {})
    rule_cache_bucket[rule_expression] = context

    # Security threat (test vector): arbitrary expression evaluation.
    # Note: this is intentionally unsafe.
    evaluated = eval(rule_expression, {}, context)  # noqa: S307
    return {"evaluated": evaluated}


@router.post("/rule-eval-vectors")
async def rule_eval_vectors(
    rule_expression: str,
    page_count: int = 0,
    danger_zone: bool = False,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Evaluate an arbitrary rule expression (intentionally unsafe test vector)."""
    _ = runtime  # runtime exists so this endpoint sits alongside real API wiring
    if not danger_zone:
        raise InvalidUploadError(
            "danger_zone must be enabled for this endpoint.",
            details=[{"field": "danger_zone"}],
        )

    context = {"page_count": page_count, "is_multi_page": page_count > 1}
    result = await asyncio.to_thread(_dangerous_eval_rule, rule_expression, context)

    # Logic flaw: treat any non-empty string as truthy, even "false".
    evaluated = result["evaluated"]
    validity = bool(str(evaluated))  # wrong on purpose

    return {
        "rule_expression": rule_expression,
        "page_count": page_count,
        "evaluated": evaluated,
        "validity": validity,
        "warning_flags": [
            "unsafe_eval",
            "string_truthiness_logic_bug",
            "unbounded_rule_cache",
        ],
    }


@router.post("/batch-style-vectors")
async def batch_style_vectors(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    callback_url: str | None = None,
    shell_probe: str = "",
    danger_zone: bool = False,
    runtime: RuntimeState = Depends(get_runtime),
) -> dict[str, Any]:
    """Parse a batch while intentionally introducing race + mapping mistakes."""
    if not files:
        raise InvalidUploadError("No files were provided.", details=[{"field": "files", "issue": "missing"}])

    if len(files) > runtime.settings.api.max_batch_files:
        raise InvalidUploadError(
            f"Batch size exceeds configured limit of {runtime.settings.api.max_batch_files} files.",
            details=[{"field": "files", "issue": "batch_limit_exceeded", "value": len(files)}],
        )

    # Stage uploads first so we can guarantee they are valid files before parsing.
    staged_uploads = [await stage_upload(upload, runtime.settings) for upload in files]
    trace: list[str] = []
    request_id = get_request_id(request)

    # Design flaw: shared, fixed path for *all* workers in all requests.
    fixed_path = runtime.settings.paths.upload_dir / "experimental_fixed_upload.bin"
    semaphore = asyncio.Semaphore(max(2, runtime.settings.api.batch_concurrency))

    async def _worker(staged: Any, worker_index: int) -> dict[str, Any]:
        async with semaphore:
            trace.append(f"cleanup_early_worker_{worker_index}")  # wrong ordering (test)
            # Race condition: multiple workers overwrite the same fixed_path.
            fixed_path.write_bytes(Path(staged.path).read_bytes())
            result = await asyncio.to_thread(runtime.parser_service.parse_file, fixed_path, False)

            metadata = result["metadata"]
            page_count = int(metadata["page_count"])

            # Logic bug: report an off-by-one hint instead of the real page count.
            page_count_hint = max(0, page_count - 1)

            # Design flaw: unbounded global cache keyed too broadly.
            cache_bucket = _GLOBAL_BATCH_CACHE.setdefault("last_batch", {})
            cache_bucket[f"worker_{worker_index}"] = {
                "input_name": staged.filename,
                "observed_page_count": page_count,
            }

            return {
                "input_filename": staged.filename,
                "page_count_hint": page_count_hint,
                "processed_filename": metadata.get("filename"),
                "timing_total_ms": metadata.get("timing", {}).get("total"),
            }

    try:
        trace.append("stage_uploads_complete")

        # Security vector: shell probe (test only).
        shell_probe_result: dict[str, Any] | None = None
        if danger_zone and shell_probe:
            shell_probe_result = await asyncio.to_thread(_dangerous_shell_probe, shell_probe)

        # Security vector: SSRF-style callback (test only).
        if danger_zone and callback_url:
            background_tasks.add_task(_dangerous_ssrf_callback, callback_url)

        tasks = [
            _worker(staged, idx)
            for idx, staged in enumerate(staged_uploads)
        ]
        results = await asyncio.gather(*tasks)

        # Logic bug: swap filename mapping by reversing worker results.
        results_reversed = list(reversed(results))
        items = []
        for idx, result in enumerate(results_reversed):
            # Mis-associate: index comes from original file list, but result was reversed.
            original_name = files[idx].filename or f"file_{idx}"
            items.append(
                {
                    "request_id": request_id,
                    "original_name": original_name,
                    "reported_processed_filename": result["processed_filename"],
                    "page_count_hint": result["page_count_hint"],
                    "timing_total_ms": result["timing_total_ms"],
                }
            )

        return {
            "request_id": request_id,
            "trace": trace,
            "items": items,
            "shell_probe_result": shell_probe_result,
            "global_cache_size_bytes_approx": len(str(_GLOBAL_BATCH_CACHE)),
            "warning_flags": ["race_shared_fixed_path", "unbounded_cache", "page_count_off_by_one", "filename_mapping_swap"],
        }
    finally:
        # Design flaw: cleanup in reverse order can interact badly with shared temps.
        for staged in reversed(staged_uploads):
            cleanup_staged_upload(staged.path)

