"""Orchestrator over three deep agents, a versioned blackboard, and a review interrupt."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import research_platform.policies  # noqa: F401
from research_platform.blackboard import Blackboard, VersionConflict
from research_platform.config import Settings
from research_platform.platform import (
    AppError,
    build_chat_model,
    build_deep_agent,
    correlation_id,
    cost_usd,
    estimate_tokens,
    message_text,
    screen_input,
)
from research_platform.tools import bind_run, publish, read_board, unbind, web_search


class ReportState(TypedDict, total=False):
    report_id: str
    question: str
    research: str
    analysis: str
    narrative: str
    draft: str
    decision: str
    notes: str
    revision_count: int
    parallel_seconds: float
    trajectory: list[dict[str, Any]]
    status: str
    cost_usd: float


@dataclass
class Report:
    report_id: str
    question: str
    status: str
    draft: str = ""
    answer: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    parallel_seconds: float = 0.0
    interrupt: dict[str, Any] | None = None
    correlation_id: str = ""
    costs: dict[str, float] = field(default_factory=dict)


class PlatformService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.reports: dict[str, Report] = {}
        self.boards: dict[str, Blackboard] = {}
        model = build_chat_model(settings, "platform")
        self.agents = {
            "researcher": build_deep_agent(
                model,
                system_prompt="ROLE: researcher. Search, then publish a finding. Do not write the final report.",
                tools=[web_search, publish],
                name="researcher",
            ),
            "analyst": build_deep_agent(
                model,
                system_prompt="ROLE: analyst. Publish an analysis. Do not search.",
                tools=[publish, read_board],
                name="analyst",
            ),
            "writer": build_deep_agent(
                model,
                system_prompt="ROLE: writer. Read the board and write the narrative only.",
                tools=[read_board],
                name="writer",
            ),
        }
        self.graph = self._compile()

    def _compile(self) -> Any:
        graph = StateGraph(ReportState)
        graph.add_node("parallel", self._parallel)
        graph.add_node("draft", self._draft)
        graph.add_node("review", self._review)
        graph.add_node("revise", self._revise)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "parallel")
        graph.add_edge("parallel", "draft")
        graph.add_edge("draft", "review")
        graph.add_conditional_edges("review", self._after_review, {"revise": "revise", "finalize": "finalize"})
        graph.add_edge("revise", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=InMemorySaver())

    async def start(self, question: str) -> Report:
        screen_input(question, 2000)
        report = Report(report_id=str(uuid.uuid4()), question=question, status="running", correlation_id=correlation_id.get())
        self.reports[report.report_id] = report
        self.boards[report.report_id] = Blackboard()
        await self._invoke(report, {"report_id": report.report_id, "question": question, "trajectory": [], "cost_usd": 0.0})
        return report

    async def review(self, report_id: str, decision: str, notes: str = "") -> Report:
        if decision not in {"approve", "revise"}:
            raise AppError(422, "Invalid decision", "Decision must be approve or revise.", "invalid-decision")
        report = self.get(report_id)
        if report.status != "awaiting_review":
            raise AppError(409, "Report is not waiting", "This report is not waiting for review.", "invalid-state")
        await self._invoke(report, Command(resume={"decision": decision, "notes": notes}))
        return report

    async def _invoke(self, report: Report, payload: Any) -> None:
        config = {"configurable": {"thread_id": report.report_id}, "recursion_limit": 40}
        try:
            result = await asyncio.wait_for(self.graph.ainvoke(payload, config), timeout=self.settings.timeout_s)
        except TimeoutError:
            report.status = "timeout"
            return
        interrupts = result.get("__interrupt__") or []
        report.trajectory = result.get("trajectory") or report.trajectory
        report.parallel_seconds = float(result.get("parallel_seconds") or report.parallel_seconds)
        report.cost_usd = round(float(result.get("cost_usd") or 0), 8)
        report.costs = {
            "researcher": self._price(str(result.get("research") or "")),
            "analyst": self._price(str(result.get("analysis") or "")),
            "writer": self._price(str(result.get("narrative") or "")),
        }
        if interrupts:
            report.status = "awaiting_review"
            report.draft = result.get("draft") or ""
            report.interrupt = interrupts[0].value
            return
        report.status = result.get("status") or "completed"
        report.answer = result.get("draft") or ""
        report.draft = report.answer
        report.interrupt = None

    async def _parallel(self, state: ReportState) -> ReportState:
        board = self.boards[state["report_id"]]
        started = time.perf_counter()
        research, analysis = await asyncio.gather(
            self._specialist("researcher", state["question"], board),
            self._specialist("analyst", state["question"], board),
        )
        elapsed = time.perf_counter() - started
        narrative_cost = 0.0
        cost = self._price(research) + self._price(analysis) + narrative_cost
        trajectory = list(state.get("trajectory") or [])
        trajectory.append(
            {
                "step": "parallel",
                "elapsedSeconds": round(elapsed, 4),
                "blackboardVersion": board.version,
                "correlationId": correlation_id.get(),
                "schemaVersion": 1,
            }
        )
        return {
            "research": research,
            "analysis": analysis,
            "parallel_seconds": round(elapsed, 4),
            "cost_usd": cost,
            "trajectory": trajectory,
        }

    async def _draft(self, state: ReportState) -> ReportState:
        board = self.boards[state["report_id"]]
        narrative = await self._specialist("writer", state["question"], board)
        draft = _assemble(
            state.get("research") or board.text_for("researcher"),
            state.get("analysis") or board.text_for("analyst"),
            narrative,
        )
        cost = float(state.get("cost_usd") or 0) + self._price(narrative)
        trajectory = list(state.get("trajectory") or [])
        trajectory.append({"step": "draft", "blackboardVersion": board.version, "correlationId": correlation_id.get(), "schemaVersion": 1})
        status = "cost_capped" if cost > self.settings.max_cost_usd else "drafted"
        return {"narrative": narrative, "draft": draft, "cost_usd": cost, "trajectory": trajectory, "status": status}

    def _review(self, state: ReportState) -> ReportState:
        if state.get("status") == "cost_capped":
            return {"decision": "approve", "status": "cost_capped"}
        if state.get("decision") in {"approve", "revise"}:
            return {}
        decision = interrupt({"reportId": state["report_id"], "draft": state.get("draft", "")})
        return {"decision": decision.get("decision", "approve"), "notes": decision.get("notes", "")}

    def _after_review(self, state: ReportState) -> str:
        if state.get("decision") == "revise" and int(state.get("revision_count") or 0) == 0 and state.get("status") != "cost_capped":
            return "revise"
        return "finalize"

    async def _revise(self, state: ReportState) -> ReportState:
        board = self.boards[state["report_id"]]
        narrative = await self._specialist("writer", state["question"], board, notes=state.get("notes") or "")
        draft = _assemble(state.get("research", ""), state.get("analysis", ""), narrative)
        cost = (
            self._price(state.get("research") or "")
            + self._price(state.get("analysis") or "")
            + self._price(narrative)
        )
        trajectory = list(state.get("trajectory") or [])
        trajectory.append({"step": "revise", "correlationId": correlation_id.get(), "schemaVersion": 1})
        return {
            "narrative": narrative,
            "draft": draft,
            "cost_usd": cost,
            "revision_count": 1,
            "decision": "approve",
            "trajectory": trajectory,
        }

    def _finalize(self, state: ReportState) -> ReportState:
        status = "cost_capped" if state.get("status") == "cost_capped" else "completed"
        return {"status": status, "draft": state.get("draft", "")}

    async def _specialist(self, name: str, question: str, board: Blackboard, notes: str = "") -> str:
        if self.settings.fail_agent == name:
            text = f"Degraded contribution: {name} failed before it could run."
            await board.append(name, "degraded", text)
            return text
        tokens = bind_run(board, name)
        try:
            if self.settings.specialist_delay_s and name in {"researcher", "analyst"}:
                await asyncio.sleep(self.settings.specialist_delay_s)
            prompt = question if not notes else f"{question}\nNOTES: {notes}"
            result = await asyncio.wait_for(
                self.agents[name].ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"configurable": {"thread_id": f"{name}-{uuid.uuid4().hex}"}, "recursion_limit": 20},
                ),
                timeout=self.settings.agent_timeout_s,
            )
            text = message_text(result["messages"][-1]).strip()
        except Exception as exc:
            text = f"Degraded contribution: {type(exc).__name__}"
        finally:
            unbind(tokens)
        if not text:
            text = "Degraded contribution: empty output."
        if name not in {entry.agent for entry in board.entries}:
            await board.append(name, "contribution", text)
        return text

    async def time_pair(self, delay: float, *, parallel: bool) -> float:
        async def work() -> None:
            await asyncio.sleep(delay)

        started = time.perf_counter()
        if parallel:
            await asyncio.gather(work(), work())
        else:
            await work()
            await work()
        return time.perf_counter() - started

    def _price(self, text: str) -> float:
        tokens = estimate_tokens(text)
        return cost_usd(tokens, tokens, self.settings.input_usd_per_token, self.settings.output_usd_per_token)

    def get(self, report_id: str) -> Report:
        report = self.reports.get(report_id)
        if report is None:
            raise AppError(404, "Report not found", "No report exists for that id.", "report-not-found")
        return report


def _assemble(research: str, analysis: str, narrative: str) -> str:
    research = research.strip() or "Degraded contribution: researcher produced nothing."
    analysis = analysis.strip() or "Degraded contribution: analyst produced nothing."
    narrative = narrative.strip() or "Degraded contribution: writer produced nothing."
    return (
        "# Research report\n\n"
        f"## Research\n{research}\n\n"
        f"## Analysis\n{analysis}\n\n"
        f"## Narrative\n{narrative}\n"
    )


async def expect_conflict() -> None:
    board = Blackboard()
    await board.append("researcher", "finding", "one")
    try:
        await board.append("analyst", "analysis", "two", expected_version=0)
    except VersionConflict:
        return
    raise AssertionError("expected a version conflict")
