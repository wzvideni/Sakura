from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from app.chat_reply import ChatReply


@dataclass(frozen=True)
class AgentAction:
    """Agent 决策出的外部动作。"""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEvent:
    """运行时主动事件，例如提醒到期。"""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentProgress:
    """Agent 运行中的中间回复，用于前台展示工具调用进度。"""

    reply: ChatReply
    stage: str = "tool_planning"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingToolAction:
    """等待用户确认后才执行的工具动作。"""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    created_at: str
    continuation_messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str = "",
    ) -> "PendingToolAction":
        return cls(
            id=uuid.uuid4().hex[:8],
            tool_name=tool_name,
            arguments=dict(arguments),
            reason=reason,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingToolAction":
        action_id = data.get("id")
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        reason = data.get("reason", "")
        created_at = data.get("created_at")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("待确认动作缺少 id。")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("待确认动作缺少工具名。")
        if not isinstance(arguments, dict):
            raise ValueError("待确认动作参数必须是 JSON object。")
        if not isinstance(reason, str):
            reason = ""
        if not isinstance(created_at, str) or not created_at.strip():
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        continuation_messages = data.get("continuation_messages", [])
        if not isinstance(continuation_messages, list):
            continuation_messages = []
        return cls(
            id=action_id.strip(),
            tool_name=tool_name.strip(),
            arguments=dict(arguments),
            reason=reason.strip(),
            created_at=created_at.strip(),
            continuation_messages=[
                dict(message)
                for message in continuation_messages
                if isinstance(message, dict)
            ],
        )

    def with_continuation_messages(
        self,
        continuation_messages: list[dict[str, Any]],
    ) -> "PendingToolAction":
        """附带确认后继续推理所需的轻量对话上下文。"""
        return PendingToolAction(
            id=self.id,
            tool_name=self.tool_name,
            arguments=dict(self.arguments),
            reason=self.reason,
            created_at=self.created_at,
            continuation_messages=[dict(message) for message in continuation_messages],
        )

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "reason": self.reason,
            "created_at": self.created_at,
        }
        if include_context and self.continuation_messages:
            data["continuation_messages"] = self.continuation_messages
        return data


@dataclass(frozen=True)
class AgentResult:
    """Agent Runtime 的统一输出，供 UI 根据回复和动作分别处理。"""

    reply: ChatReply
    actions: list[AgentAction] = field(default_factory=list)
