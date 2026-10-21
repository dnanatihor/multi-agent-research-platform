# Multi-agent research architecture

## Decision

Orchestrator, two parallel specialists, writer, human review. The blackboard and the report are separate records. Agents write the board. The orchestrator writes the report.

## Constraints

- No agent writes the final status except the review node.
- Tool access is publish, read, and search. Harness shell and file tools are denied.
- Correlation id is on the report and in logs.

## SLO

Under $2 per report and under 3 minutes. Every agent contributes a section.

## Radar

- Adopt: LangGraph for the orchestrator, versioned append-only board, Poetry.
- Trial: deepagents for the three roles, one shared harness profile.
- Hold: a general-purpose subagent. It is disabled in the harness profile.

## Adversarial review

- A specialist prompt that asks for shell access has no such tool bound.
- Editor notes are data to the writer. They are not a new system prompt that can grant tools.
