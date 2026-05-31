from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api_client import ApiSettings, OpenAICompatibleClient
from app.character_loader import CharacterProfile, CharacterRegistry
from app.tts import TTSProvider


@dataclass(frozen=True)
class AppContext:
    """应用启动阶段组装出的核心依赖。"""

    base_dir: Path
    settings: ApiSettings
    api_client: OpenAICompatibleClient
    character_registry: CharacterRegistry
    character_profile: CharacterProfile
    tts_provider: TTSProvider
