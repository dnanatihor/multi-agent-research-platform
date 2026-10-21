# ADR-002: Research and analysis run together, then a human reviews

## Status

Accepted

## Context

Running the specialists one after another spends wall-clock time without a data dependency. Publishing a report without review can ship a draft the writer marked as speculative.

## Decision

The orchestrator graph gathers the researcher and analyst with `asyncio.gather`, then the writer drafts, then `interrupt()` returns the draft. Approve finalises. Revise passes editor notes into the writer and drafts again. A failed specialist appends `Degraded contribution` so the report still names that role. Cost above the ceiling sets `cost_capped` and skips the interrupt.

Each specialist is a deep agent with harness tools removed. They publish through the `publish` tool, not by writing files.

## Constraints

- The trace contains a parallel step.
- Per-agent cost is stored on the report.
- The assembler always emits Research, Analysis, and Narrative headings.

## Options

- One agent with three prompts. Rejected: a failure is not attributable and the tasks are not isolated.
- Automatic publish. Rejected by the product requirement for review before final.

## Failure modes

- One slow specialist holds the gather. The timeout on the model call bounds it. The parallelism test checks wall-clock against a sequential pair.
- A restart loses the interrupt. The draft remains on the report and must be resubmitted. See tech debt.

## Owner

Platform service.
