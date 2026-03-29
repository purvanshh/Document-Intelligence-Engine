# Security / Logic Fixture Corpus

These files exist to exercise external tooling (SAST/DAST parsers, webhook consumers, alert routing, etc.).

Important:
- Fixtures are **NOT imported** by the application.
- They are **plain-text** examples and are not executed at runtime.

Why:
- We want predictable, repeatable “bad pattern” inputs for scanning systems.
- Keeping this corpus in-repo makes webhook integration tests easy.

Contents:
- `fixtures_vuln_samples.txt`: mixed patterns (shell, SQL strings, fake secrets, unsafe eval)
- `fixtures_logic_design_smells.txt`: typical logic/design smells

Notes:
- Secrets in fixtures are fake placeholders.
- Any scanner should treat these as matches but they are non-executable by design.

