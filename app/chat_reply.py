from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DEFAULT_TONE = "中性"


@dataclass(frozen=True)
class ChatSegment:
    text: str
    tone: str = DEFAULT_TONE
    translation: str = ""
    portrait: str = ""

    def display_text(self, subtitle_language: str) -> str:
        """按字幕语言返回气泡显示文本；缺少译文时回退日文原文。"""
        if subtitle_language == "zh" and self.translation.strip():
            return self.translation.strip()
        return self.text


@dataclass(frozen=True)
class ChatReply:
    segments: list[ChatSegment]

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text.strip()).strip()

    @property
    def translation(self) -> str:
        return "\n".join(
            segment.display_text("zh")
            for segment in self.segments
            if segment.display_text("zh").strip()
        ).strip()

    def display_text(self, subtitle_language: str) -> str:
        if subtitle_language == "zh":
            return self.translation or self.text
        return self.text

    @property
    def tone(self) -> str:
        for segment in self.segments:
            if segment.text.strip() and segment.tone.strip():
                return segment.tone.strip()
        return DEFAULT_TONE


def parse_chat_reply(content: str) -> ChatReply:
    """解析模型返回；非 JSON 或旧格式会自动降级成单段中性回复。"""
    content = content.strip()
    if not content:
        return ChatReply([ChatSegment("", DEFAULT_TONE)])

    data = _try_load_json(content)
    if data is None:
        return ChatReply([ChatSegment(content, DEFAULT_TONE)])

    if isinstance(data, dict):
        segments = _parse_segments(data)
        if segments:
            return ChatReply(segments)

    return ChatReply([ChatSegment(content, DEFAULT_TONE)])


def _parse_segments(data: dict[str, Any]) -> list[ChatSegment]:
    raw_segments = data.get("segments")
    if isinstance(raw_segments, list):
        segments = [_parse_segment(item) for item in raw_segments]
        return [segment for segment in segments if segment is not None]

    text = _clean_first_text(data, "ja", "japanese", "reply", "text")
    if text:
        tone = data.get("tone")
        translation = _clean_first_text(data, "zh", "chinese", "translation")
        return [_build_segment(text, tone, translation, data.get("portrait"))]

    return []


def _parse_segment(item: Any) -> ChatSegment | None:
    if isinstance(item, str):
        text = item.strip()
        return ChatSegment(text, DEFAULT_TONE) if text else None
    if not isinstance(item, dict):
        return None

    text = _clean_first_text(item, "ja", "japanese", "text")
    if not text:
        return None
    translation = _clean_first_text(item, "zh", "chinese", "translation")
    return _build_segment(text, item.get("tone"), translation, item.get("portrait"))


def _build_segment(text: str, tone: Any, translation: str, portrait: Any) -> ChatSegment:
    text = text.strip()
    translation = translation.strip()
    if _looks_chinese(text) and _looks_japanese(translation):
        text, translation = translation, text
    return ChatSegment(text, _clean_tone(tone), translation, _clean_portrait(portrait))


def _clean_tone(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_TONE


def _clean_portrait(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _clean_first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _looks_japanese(value: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff" or "\uff66" <= char <= "\uff9f"
        for char in value
    )


def _looks_chinese(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value) and not _looks_japanese(value)


def _try_load_json(content: str) -> Any | None:
    try:
        return json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        return None


def _strip_code_fence(content: str) -> str:
    lines = content.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content
