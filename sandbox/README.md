## Sandbox evaluation app (non-production)

This directory contains a **synthetic** mini-application used only for evaluating code review and static analysis tooling.

- It is **not** wired into the production `src/` package.
- Do **not** deploy this code.
- Running locally (optional):

```bash
uvicorn sandbox.main:app --reload
```

