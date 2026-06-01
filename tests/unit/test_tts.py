from __future__ import annotations

import importlib.util
import sys
import types
import urllib.error
from pathlib import Path

if importlib.util.find_spec("PySide6") is None:
    pyside_module = types.ModuleType("PySide6")
    qtcore_module = types.ModuleType("PySide6.QtCore")
    qtmultimedia_module = types.ModuleType("PySide6.QtMultimedia")

    class QObject:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class QTimer:
        @staticmethod
        def singleShot(*_args: object, **_kwargs: object) -> None:
            pass

    class QUrl:
        @staticmethod
        def fromLocalFile(path: str) -> str:
            return path

    class Signal:
        def __init__(self, *_args: object) -> None:
            pass

        def connect(self, *_args: object, **_kwargs: object) -> None:
            pass

        def emit(self, *_args: object, **_kwargs: object) -> None:
            pass

    def Slot(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        def decorator(function):  # type: ignore[no-untyped-def]
            return function

        return decorator

    class QAudioOutput:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class QMediaPlayer:
        class MediaStatus:
            EndOfMedia = object()

        class PlaybackState:
            PlayingState = object()

        class Error:
            pass

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    qtcore_module.QObject = QObject
    qtcore_module.QTimer = QTimer
    qtcore_module.QUrl = QUrl
    qtcore_module.Signal = Signal
    qtcore_module.Slot = Slot
    qtmultimedia_module.QAudioOutput = QAudioOutput
    qtmultimedia_module.QMediaPlayer = QMediaPlayer
    sys.modules["PySide6"] = pyside_module
    sys.modules["PySide6.QtCore"] = qtcore_module
    sys.modules["PySide6.QtMultimedia"] = qtmultimedia_module

from app.voice.tts import (
    GPTSoVITSTTSProvider,
    GPTSoVITSTTSSettings,
    TTSPreparedAudio,
    _load_tone_references,
    _resolve_request_text_lang,
)
from app.voice import VoicePlaybackController
from app.voice.text_language_guard import should_skip_tts_text


def test_language_guard_allows_japanese_text_for_japanese_tts() -> None:
    assert not should_skip_tts_text("うん。大丈夫。", "ja")


def test_language_guard_skips_obvious_chinese_for_japanese_tts() -> None:
    assert should_skip_tts_text("原因是 Mermaid 语法。", "ja")
    assert should_skip_tts_text("这是中文，不能进 TTS。", "all_ja")


def test_language_guard_keeps_kanji_only_japanese_candidate() -> None:
    assert not should_skip_tts_text("大丈夫", "ja")


def test_language_guard_only_applies_to_japanese_targets() -> None:
    assert not should_skip_tts_text("这是中文，不能进 TTS。", "zh")
    assert not should_skip_tts_text("这是中文，不能进 TTS。", "en")


def test_tts_mixed_japanese_and_english_uses_auto_lang() -> None:
    text = "Steamを開いているんだね。Muse Dash…楽しそうなゲーム。"

    assert _resolve_request_text_lang(text, "ja") == "auto"


def test_tts_plain_japanese_keeps_configured_lang() -> None:
    text = "でも私、初めて君に会った時、思ったよ。"

    assert _resolve_request_text_lang(text, "ja") == "ja"


def test_tts_explicit_english_lang_is_not_overridden() -> None:
    text = "Steam is open."

    assert _resolve_request_text_lang(text, "en") == "en"


def test_tts_yue_mixed_english_uses_auto_yue() -> None:
    text = "Steam 打开咗。"

    assert _resolve_request_text_lang(text, "all_yue") == "auto_yue"


def test_tone_references_load_four_part_rows_only() -> None:
    ref_path = Path("characters/sakura/voice/refs/ref.txt")
    rows = [line for line in ref_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    references = _load_tone_references(ref_path, Path("characters/sakura"))

    assert all(len(row.split("|")) == 4 for row in rows)
    assert references
    assert all("|" not in reference.ref_text for items in references.values() for reference in items)
    assert all(reference.ref_audio_path.exists() for items in references.values() for reference in items)


def test_tts_service_probe_reports_unavailable_service(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = types.SimpleNamespace()
    provider.settings = _minimal_tts_settings()
    provider._service_checked = False
    messages: list[str] = []

    def fake_create_connection(*_args: object, **_kwargs: object) -> object:
        raise OSError("connection refused")

    monkeypatch.setattr("app.voice.tts.socket.create_connection", fake_create_connection)

    assert not GPTSoVITSTTSProvider._ensure_service_available(provider, messages.append)
    assert "服务不可用" in messages[0]
    assert "http://127.0.0.1:9880/tts" in messages[0]


def test_tts_service_probe_uses_tcp_connection_without_get(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = types.SimpleNamespace()
    provider.settings = _minimal_tts_settings()
    provider._service_checked = False
    messages: list[str] = []
    calls: list[tuple[tuple[str, int], int]] = []

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_create_connection(address: tuple[str, int], timeout: int) -> FakeConnection:
        calls.append((address, timeout))
        return FakeConnection()

    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("服务探测不应请求 /tts")

    monkeypatch.setattr("app.voice.tts.socket.create_connection", fake_create_connection)
    monkeypatch.setattr("app.voice.tts.urllib.request.urlopen", fail_urlopen)

    assert GPTSoVITSTTSProvider._ensure_service_available(provider, messages.append)
    assert messages == []
    assert calls == [(("127.0.0.1", 9880), 1)]


def test_tts_weight_switch_error_includes_endpoint_and_path(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = types.SimpleNamespace()
    provider.settings = _minimal_tts_settings()
    messages: list[str] = []

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("bad weights")

    monkeypatch.setattr("app.voice.tts.urllib.request.urlopen", fake_urlopen)

    ok = GPTSoVITSTTSProvider._request_weight_switch(
        provider,
        "set_gpt_weights",
        Path("characters/sakura/voice/models/Sakura-e15.ckpt"),
        messages.append,
    )

    assert not ok
    assert "set_gpt_weights" in messages[0]
    assert "Sakura-e15.ckpt" in messages[0]
    assert "bad weights" in messages[0]


def test_gptsovits_provider_initializes_qt_player_lazily(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.voice.tts as tts_module

    calls: list[str] = []

    class SignalStub:
        def connect(self, *_args: object, **_kwargs: object) -> None:
            pass

    class AudioOutputStub:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("audio")

    class MediaPlayerStub:
        class MediaStatus:
            EndOfMedia = object()

        class PlaybackState:
            PlayingState = object()

        class Error:
            pass

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("player")
            self.mediaStatusChanged = SignalStub()
            self.playbackStateChanged = SignalStub()
            self.errorOccurred = SignalStub()

        def setAudioOutput(self, _output: object) -> None:
            pass

        def setSource(self, _source: object) -> None:
            calls.append("source")

        def play(self) -> None:
            calls.append("play")

        def stop(self) -> None:
            pass

    monkeypatch.setattr(tts_module, "QAudioOutput", AudioOutputStub)
    monkeypatch.setattr(tts_module, "QMediaPlayer", MediaPlayerStub)

    provider = GPTSoVITSTTSProvider(_minimal_tts_settings())

    assert calls == []

    provider._pending_audio.append((Path("dummy.wav"), None, None, None))
    provider._play_next()

    assert calls == ["audio", "player", "source", "play"]


def test_voice_playback_controller_falls_back_to_subtitle_callbacks_on_tts_error() -> None:
    from app.llm.chat_reply import ChatSegment

    class FailingTTS:
        def speak(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("tts down")

    events: list[str] = []
    controller = VoicePlaybackController(FailingTTS(), lambda *_args, **_kwargs: None)  # type: ignore[arg-type]

    controller.speak_segment(
        ChatSegment("こんにちは", "中性"),
        1,
        on_started=lambda: events.append("started"),
        on_finished=lambda: events.append("finished"),
    )

    assert events == ["started", "finished"]


def test_voice_playback_controller_skips_chinese_text_for_japanese_tts() -> None:
    from app.llm.chat_reply import ChatSegment

    class RecordingTTS:
        def __init__(self) -> None:
            self.speak_calls = 0

        def speak(self, *_args: object, **_kwargs: object) -> None:
            self.speak_calls += 1

    events: list[str] = []
    stages: list[str] = []
    tts = RecordingTTS()
    controller = VoicePlaybackController(
        tts,
        lambda stage, _payload=None: stages.append(stage),
        target_text_lang_getter=lambda: "ja",
    )  # type: ignore[arg-type]

    controller.speak_segment(
        ChatSegment("这是中文，不能进 TTS。", "中性"),
        1,
        on_started=lambda: events.append("started"),
        on_finished=lambda: events.append("finished"),
    )

    assert tts.speak_calls == 0
    assert events == ["started", "finished"]
    assert "tts_skipped_language_guard" in stages


def test_voice_playback_controller_skips_prepare_for_chinese_text() -> None:
    from app.llm.chat_reply import ChatSegment

    class RecordingTTS:
        def __init__(self) -> None:
            self.prepare_calls = 0

        def prepare(self, *_args: object, **_kwargs: object) -> TTSPreparedAudio:
            self.prepare_calls += 1
            return TTSPreparedAudio(text="dummy")

        def discard_prepared(self, *_args: object, **_kwargs: object) -> None:
            pass

    tts = RecordingTTS()
    controller = VoicePlaybackController(
        tts,
        lambda *_args, **_kwargs: None,
        target_text_lang_getter=lambda: "ja",
    )  # type: ignore[arg-type]

    controller.prepare_next(ChatSegment("这是中文，不能进 TTS。", "中性"))

    assert tts.prepare_calls == 0


def test_voice_playback_controller_allows_japanese_speak_and_prepare() -> None:
    from app.llm.chat_reply import ChatSegment

    class RecordingTTS:
        def __init__(self) -> None:
            self.speak_calls = 0
            self.prepare_calls = 0

        def speak(
            self,
            *_args: object,
            on_started=None,  # type: ignore[no-untyped-def]
            on_finished=None,  # type: ignore[no-untyped-def]
            **_kwargs: object,
        ) -> None:
            self.speak_calls += 1
            if on_started is not None:
                on_started()
            if on_finished is not None:
                on_finished()

        def prepare(self, text: str, *_args: object, **_kwargs: object) -> TTSPreparedAudio:
            self.prepare_calls += 1
            return TTSPreparedAudio(text=text)

        def discard_prepared(self, *_args: object, **_kwargs: object) -> None:
            pass

    events: list[str] = []
    tts = RecordingTTS()
    controller = VoicePlaybackController(tts, lambda *_args, **_kwargs: None)  # type: ignore[arg-type]

    controller.speak_segment(
        ChatSegment("うん。大丈夫。", "中性"),
        1,
        on_started=lambda: events.append("started"),
        on_finished=lambda: events.append("finished"),
    )
    controller.prepare_next(ChatSegment("次の一段。", "中性"))

    assert tts.speak_calls == 1
    assert tts.prepare_calls == 1
    assert events == ["started", "finished"]


def test_voice_playback_controller_uses_prepared_japanese_audio() -> None:
    from app.llm.chat_reply import ChatSegment

    class RecordingTTS:
        def __init__(self) -> None:
            self.prepared = TTSPreparedAudio(text="次の一段。")
            self.speak_prepared_calls = 0

        def prepare(self, *_args: object, **_kwargs: object) -> TTSPreparedAudio:
            return self.prepared

        def speak_prepared(
            self,
            _handle: TTSPreparedAudio,
            on_started=None,  # type: ignore[no-untyped-def]
            on_finished=None,  # type: ignore[no-untyped-def]
        ) -> None:
            self.speak_prepared_calls += 1
            if on_started is not None:
                on_started()
            if on_finished is not None:
                on_finished()

        def discard_prepared(self, *_args: object, **_kwargs: object) -> None:
            pass

    events: list[str] = []
    segment = ChatSegment("次の一段。", "中性")
    tts = RecordingTTS()
    controller = VoicePlaybackController(tts, lambda *_args, **_kwargs: None)  # type: ignore[arg-type]

    controller.prepare_next(segment)
    controller.speak_segment(
        segment,
        1,
        on_started=lambda: events.append("started"),
        on_finished=lambda: events.append("finished"),
    )

    assert tts.speak_prepared_calls == 1
    assert events == ["started", "finished"]


def test_voice_playback_controller_ignores_prepare_error() -> None:
    from app.llm.chat_reply import ChatSegment

    class FailingPrepareTTS:
        def prepare(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("prepare down")

        def discard_prepared(self, *_args: object, **_kwargs: object) -> None:
            pass

    controller = VoicePlaybackController(FailingPrepareTTS(), lambda *_args, **_kwargs: None)  # type: ignore[arg-type]

    controller.prepare_next(ChatSegment("次の一段", "中性"))


def _minimal_tts_settings() -> GPTSoVITSTTSSettings:
    return GPTSoVITSTTSSettings(
        enabled=True,
        api_url="http://127.0.0.1:9880/tts",
        ref_audio_path=Path("characters/sakura/voice/refs/tone_refs/00_中性_VO01_2785.ogg"),
        ref_text_path=Path("characters/sakura/voice/refs/ref.txt"),
        ref_text="テスト",
        ref_lang="ja",
        text_lang="ja",
        timeout_seconds=1,
    )
