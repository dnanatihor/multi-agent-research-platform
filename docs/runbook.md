# Runbook

## Draft waiting

`GET /api/v1/reports/{id}` returns `awaiting_review`. Approve or send notes. Notes appear in the narrative as editor notes.

## Degraded section

A section that starts with `Degraded contribution` means that specialist failed. The report can still be approved. Investigate the trace before treating the report as complete research.

## Cost cap

Status `cost_capped` means the estimate exceeded `PLATFORM_MAX_COST_USD` before the interrupt. Narrow the question rather than lifting the ceiling silently.
