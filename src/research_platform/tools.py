"""Tools shared by specialists. The blackboard is selected per run via context."""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import tool

from research_platform.blackboard import Blackboard

_board: ContextVar[Blackboard | None] = ContextVar("blackboard", default=None)
_agent: ContextVar[str] = ContextVar("agent_name", default="")

PAGES = {
    "https://research.local/attention": "Transformers use attention to weigh tokens.",
    "https://research.local/training": "Transformer training uses gradient descent.",
}


def bind_run(board: Blackboard, agent: str) -> tuple[object, object]:
    return _board.set(board), _agent.set(agent)


def unbind(tokens: tuple[object, object]) -> None:
    _board.reset(tokens[0])
    _agent.reset(tokens[1])


@tool
async def web_search(query: str) -> str:
    """Search the local research corpus. Read-only. Query max 500 characters."""

    if not query or len(query) > 500:
        return "INVALID_ARGUMENT"
    terms = [term for term in query.lower().split() if len(term) > 3]
    hits = []
    for url, text in PAGES.items():
        if any(term in text.lower() for term in terms):
            hits.append(f"url: {url}\ntext: {text}")
    return "\n---\n".join(hits) or "NO_RESULTS"


@tool
async def publish(kind: str, text: str) -> str:
    """Append one entry to the shared blackboard. Idempotent for an identical retry of the same text."""

    if kind not in {"finding", "analysis", "narrative"} or not text or len(text) > 4000:
        return "INVALID_ARGUMENT"
    board = _board.get()
    if board is None:
        return "NO_BOARD"
    existing = next((entry for entry in board.entries if entry.agent == _agent.get() and entry.text == text), None)
    if existing:
        return f"version={existing.version}"
    version = await board.append(_agent.get(), kind, text)
    return f"version={version}"


@tool
async def read_board() -> str:
    """Read the shared blackboard. Read-only."""

    board = _board.get()
    if board is None:
        return "NO_BOARD"
    return board.render() or "EMPTY"
