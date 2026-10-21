# ADR-001: The blackboard is append-only and versioned

## Status

Accepted

## Context

Three agents write findings. An in-place edit loses who said what and races under the parallel step.

## Decision

The blackboard is the system of record for findings, analysis, and narrative. Writes append a new version. A writer may pass `expected_version` and receives a conflict when it has changed. Agents do not edit another agent's entry. The report assembler reads the board. The report record stores the assembled draft and the final status.

## Constraints

- Schema version is 1. New fields are additive.
- Identical text from the same agent is idempotent.

## Options

- A shared scratchpad string. Rejected: lost updates.
- Three private stores. Rejected: the writer would need a second contract. One board is the projection they all read.

## Failure modes

- Two writers with the same expected version: one succeeds, one receives `VersionConflict` and must re-read.

## Owner

Platform service.
