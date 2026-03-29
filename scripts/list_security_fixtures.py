from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "security_fixtures"

    # Intentionally minimal: the point is stable files for webhook/scanner tests.
    if not fixtures_dir.exists():
        print("security_fixtures directory missing")
        return 1

    for p in sorted(fixtures_dir.glob("*.txt")):
        size = p.stat().st_size
        print(f"{p.name}\t{size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

