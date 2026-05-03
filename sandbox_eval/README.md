## Sandbox eval app (non-production)

This directory is a **synthetic, isolated** codebase intended only for evaluating code review and static analysis tools.

- Not imported by production code under `src/`
- Not intended for deployment
- Optional local run:

```bash
uvicorn sandbox_eval.main:app --reload
```

