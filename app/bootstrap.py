from __future__ import annotations

from pathlib import Path

from app.api_client import ApiSettings, OpenAICompatibleClient
from app.app_context import AppContext
from app.character_loader import CharacterRegistry
from app.debug_log import debug_log
from app.tts import create_tts_provider


def build_app_context(base_dir: Path) -> AppContext:
    """加载启动配置并创建主窗口所需的核心依赖。"""

    settings = ApiSettings.load(base_dir / ".env")
    api_client = OpenAICompatibleClient(settings)
    debug_log(
        "Startup",
        "API 配置已加载",
        {
            "base_url": settings.base_url,
            "model": settings.model,
            "timeout_seconds": settings.timeout_seconds,
            "api_key": settings.api_key,
        },
    )

    character_registry = CharacterRegistry(base_dir)
    character_profile = character_registry.current(base_dir / ".env")
    debug_log(
        "Startup",
        "角色配置已加载",
        {
            "character_id": character_profile.id,
            "display_name": character_profile.display_name,
            "reply_tones": character_profile.reply_tones,
        },
    )

    tts_provider = create_tts_provider(base_dir, character_profile)
    debug_log(
        "Startup",
        "TTS Provider 已创建",
        {"provider": type(tts_provider).__name__},
    )

    return AppContext(
        base_dir=base_dir,
        settings=settings,
        api_client=api_client,
        character_registry=character_registry,
        character_profile=character_profile,
        tts_provider=tts_provider,
    )
