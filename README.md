# Multi-Agent Research Platform

An orchestrator runs a researcher and an analyst together, then a writer. They share a versioned blackboard. A human approves the draft before it is final.

## Run

```bash
poetry install
poetry run uvicorn research_platform.main:app --reload --port 8004
```

## API

- `POST /api/v1/reports` returns 202 with a draft when the cost estimate is inside the ceiling
- `POST /api/v1/reports/{id}/review` with `{"decision":"approve"}` or `{"decision":"revise","notes":"..."}`
- `GET /api/v1/reports/{id}/trace` and `GET /api/v1/reports/{id}/cost`

## Decisions

See `docs/adr`. Budget: under $2 and under 3 minutes. Every specialist appears in the report, including a degraded contribution when that agent fails.
