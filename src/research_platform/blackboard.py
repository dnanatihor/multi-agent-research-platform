"""Versioned shared state. Writers append; they do not edit history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


class VersionConflict(Exception):
    def __init__(self, actual: int) -> None:
        super().__init__(f"blackboard version is {actual}")
        self.actual = actual


@dataclass
class Entry:
    version: int
    agent: str
    kind: str
    text: str
    schema_version: int = 1


@dataclass
class Blackboard:
    version: int = 0
    entries: list[Entry] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(self, agent: str, kind: str, text: str, expected_version: int | None = None) -> int:
        async with self._lock:
            if expected_version is not None and expected_version != self.version:
                raise VersionConflict(self.version)
            self.version += 1
            self.entries.append(Entry(self.version, agent, kind, text))
            return self.version

    def text_for(self, agent: str) -> str:
        parts = [entry.text for entry in self.entries if entry.agent == agent]
        return "\n".join(parts)

    def render(self) -> str:
        return "\n".join(
            f"[{entry.agent}/{entry.kind}/v{entry.version}] {entry.text}" for entry in self.entries
        )
