"""One scripted policy, branched by the specialist role in the system prompt."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from research_platform.platform import last_human, register_policy, tool_call, tool_message_names, transcript


def _role(messages: list[BaseMessage]) -> str:
    text = transcript(messages)
    for name in ("researcher", "analyst", "writer"):
        if f"ROLE: {name}" in text:
            return name
    return ""


def platform_policy(messages: list[BaseMessage], tool_names: list[str]) -> AIMessage:
    role = _role(messages)
    used = tool_message_names(messages)
    text = transcript(messages)
    if role == "researcher":
        if "web_search" in tool_names and "web_search" not in used:
            return tool_call("web_search", {"query": last_human(messages)[:500]})
        if "publish" in tool_names and "publish" not in used:
            return tool_call(
                "publish",
                {
                    "kind": "finding",
                    "text": "Transformers use attention to weigh tokens. Training uses gradient descent.",
                },
            )
        return AIMessage(content="Research contribution: attention weighs tokens and training uses gradient descent.")
    if role == "analyst":
        if "publish" in tool_names and "publish" not in used:
            return tool_call(
                "publish",
                {
                    "kind": "analysis",
                    "text": "Attention and training are separate steps and both belong in the report.",
                },
            )
        return AIMessage(content="Analysis contribution: keep attention and training as distinct sections.")
    if role == "writer":
        if "read_board" in tool_names and "read_board" not in used:
            return tool_call("read_board", {})
        notes = ""
        if "NOTES:" in text:
            notes = text.split("NOTES:", 1)[-1].splitlines()[0].strip()
        body = "Narrative integrates the research and analysis already on the board."
        if notes:
            body = f"{body} Editor notes: {notes}"
        return AIMessage(content=body)
    return AIMessage(content="No role was assigned.")


register_policy("platform", platform_policy)
