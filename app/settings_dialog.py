from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.api_client import ApiSettings, OpenAICompatibleClient
from app.character_loader import CharacterProfile, CharacterRegistry
from app.proactive_care import (
    PROACTIVE_MAX_COOLDOWN_MINUTES,
    PROACTIVE_MAX_CHECK_INTERVAL_MINUTES,
    PROACTIVE_MIN_COOLDOWN_MINUTES,
    PROACTIVE_MIN_CHECK_INTERVAL_MINUTES,
    ProactiveCareSettings,
)
from app.tts import GPTSoVITSTTSSettings, TTSConfigError


class ApiConnectionTestWorker(QObject):
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, settings: ApiSettings) -> None:
        super().__init__()
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            message = OpenAICompatibleClient(self.settings).test_connection()
        except Exception as exc:  # UI 边界统一转成可读错误。
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(message)
        finally:
            self.finished.emit()


class SettingsDialog(QDialog):
    def __init__(
        self,
        api_settings: ApiSettings,
        tts_settings: GPTSoVITSTTSSettings,
        base_dir: Path,
        character_registry: CharacterRegistry | None = None,
        current_character: CharacterProfile | None = None,
        screen_observation_enabled: bool = True,
        proactive_care_settings: ProactiveCareSettings | None = None,
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self.tts_settings = tts_settings
        self.character_registry = character_registry
        self.current_character = current_character
        self.result_api_settings: ApiSettings | None = None
        self.result_tts_settings: GPTSoVITSTTSSettings | None = None
        self.result_character_id: str | None = None
        self.result_screen_observation_enabled: bool | None = None
        self.result_proactive_care_settings: ProactiveCareSettings | None = None
        self._api_test_thread: QThread | None = None
        self._api_test_worker: ApiConnectionTestWorker | None = None

        self.setWindowTitle("设置")
        self.resize(560, 400)

        tabs = QTabWidget(self)
        if character_registry is not None and current_character is not None:
            tabs.addTab(self._build_character_tab(character_registry, current_character), "角色")
        tabs.addTab(self._build_api_tab(api_settings), "API")
        tabs.addTab(self._build_tts_tab(tts_settings), "TTS")
        tabs.addTab(
            self._build_privacy_tab(
                screen_observation_enabled,
                proactive_care_settings or ProactiveCareSettings(),
            ),
            "隐私",
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background: #f4fbfd;
                color: #24343a;
                font-family: "Microsoft YaHei", "Yu Gothic UI", sans-serif;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(120, 176, 188, 0.48);
                border-radius: 8px;
                background: rgba(226, 246, 250, 0.70);
            }
            QTabBar::tab {
                background: rgba(226, 246, 250, 0.75);
                border: 1px solid rgba(120, 176, 188, 0.42);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 7px 18px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #27616d;
                font-weight: 700;
            }
            QLineEdit, QSpinBox {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(120, 176, 188, 0.50);
                border-radius: 7px;
                padding: 6px 8px;
            }
            QPushButton {
                background: #72c7d6;
                border: none;
                border-radius: 8px;
                color: white;
                min-width: 72px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #5eb7c8;
            }
            """
        )

    def _build_character_tab(
        self,
        character_registry: CharacterRegistry,
        current_character: CharacterProfile,
    ) -> QWidget:
        tab = QWidget(self)
        self.character_combo = QComboBox(tab)
        for profile in character_registry.all():
            self.character_combo.addItem(profile.display_name, profile.id)
            if profile.id == current_character.id:
                self.character_combo.setCurrentIndex(self.character_combo.count() - 1)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("当前角色", self.character_combo)
        tab.setLayout(form_layout)
        return tab

    def _build_api_tab(self, settings: ApiSettings) -> QWidget:
        tab = QWidget(self)
        self.base_url_edit = QLineEdit(settings.base_url, tab)
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")

        self.api_key_edit = QLineEdit(settings.api_key, tab)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("请输入 API Key")

        self.model_edit = QLineEdit(settings.model, tab)
        self.model_edit.setPlaceholderText("gpt-4.1-mini")

        self.api_timeout_spin = QSpinBox(tab)
        self.api_timeout_spin.setRange(1, 600)
        self.api_timeout_spin.setSuffix(" 秒")
        self.api_timeout_spin.setValue(settings.timeout_seconds)

        self.api_test_button = QPushButton("测试 API", tab)
        self.api_test_button.clicked.connect(self._test_api_settings)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("Base URL", self.base_url_edit)
        form_layout.addRow("API Key", self.api_key_edit)
        form_layout.addRow("模型", self.model_edit)
        form_layout.addRow("超时", self.api_timeout_spin)
        form_layout.addRow("", self.api_test_button)
        tab.setLayout(form_layout)
        return tab

    def _build_tts_tab(self, settings: GPTSoVITSTTSSettings) -> QWidget:
        tab = QWidget(self)
        self.tts_enabled_check = QCheckBox("启用 GPT-SoVITS 语音", tab)
        self.tts_enabled_check.setChecked(settings.enabled)

        self.tts_api_url_edit = QLineEdit(settings.api_url, tab)
        self.tts_api_url_edit.setPlaceholderText("http://127.0.0.1:9880/tts")

        self.ref_lang_edit = QLineEdit(settings.ref_lang, tab)
        self.text_lang_edit = QLineEdit(settings.text_lang, tab)

        self.tts_timeout_spin = QSpinBox(tab)
        self.tts_timeout_spin.setRange(1, 600)
        self.tts_timeout_spin.setSuffix(" 秒")
        self.tts_timeout_spin.setValue(settings.timeout_seconds)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("", self.tts_enabled_check)
        form_layout.addRow("API URL", self.tts_api_url_edit)
        form_layout.addRow("参考语言", self.ref_lang_edit)
        form_layout.addRow("文本语言", self.text_lang_edit)
        form_layout.addRow("超时", self.tts_timeout_spin)
        tab.setLayout(form_layout)
        return tab

    def _build_privacy_tab(
        self,
        screen_observation_enabled: bool,
        proactive_care_settings: ProactiveCareSettings,
    ) -> QWidget:
        tab = QWidget(self)
        self.screen_observation_enabled_check = QCheckBox("允许按需屏幕观察", tab)
        self.screen_observation_enabled_check.setChecked(screen_observation_enabled)

        self.proactive_care_enabled_check = QCheckBox("启用主动关怀", tab)
        self.proactive_care_enabled_check.setChecked(proactive_care_settings.enabled)

        self.proactive_screen_context_enabled_check = QCheckBox("允许模型主动获取屏幕信息", tab)
        self.proactive_screen_context_enabled_check.setChecked(
            proactive_care_settings.screen_context_enabled
        )
        self.proactive_screen_context_enabled_check.setEnabled(screen_observation_enabled)

        self.proactive_check_interval_spin = QSpinBox(tab)
        self.proactive_check_interval_spin.setRange(
            PROACTIVE_MIN_CHECK_INTERVAL_MINUTES,
            PROACTIVE_MAX_CHECK_INTERVAL_MINUTES,
        )
        self.proactive_check_interval_spin.setSuffix(" 分钟")
        self.proactive_check_interval_spin.setValue(
            proactive_care_settings.normalized().check_interval_minutes
        )

        self.proactive_cooldown_spin = QSpinBox(tab)
        self.proactive_cooldown_spin.setRange(
            PROACTIVE_MIN_COOLDOWN_MINUTES,
            PROACTIVE_MAX_COOLDOWN_MINUTES,
        )
        self.proactive_cooldown_spin.setSuffix(" 分钟")
        self.proactive_cooldown_spin.setValue(
            proactive_care_settings.normalized().cooldown_minutes
        )

        self.screen_observation_enabled_check.toggled.connect(
            self._sync_proactive_screen_context_enabled
        )

        form_layout = QFormLayout()
        form_layout.setContentsMargins(16, 18, 16, 16)
        form_layout.setSpacing(12)
        form_layout.addRow("", self.screen_observation_enabled_check)
        form_layout.addRow("", self.proactive_care_enabled_check)
        form_layout.addRow("", self.proactive_screen_context_enabled_check)
        form_layout.addRow("主动检查间隔", self.proactive_check_interval_spin)
        form_layout.addRow("主动打扰冷却", self.proactive_cooldown_spin)
        tab.setLayout(form_layout)
        return tab

    def accept(self) -> None:
        if self._api_test_thread is not None:
            QMessageBox.information(self, "测试中", "API 测试仍在进行，请等待完成后再保存设置。")
            return

        api_settings = self._validated_api_settings()
        if api_settings is None:
            return
        tts_settings = self._validated_tts_settings()
        if tts_settings is None:
            return

        self.result_api_settings = api_settings
        self.result_tts_settings = tts_settings
        self.result_character_id = self._selected_character_id()
        self.result_screen_observation_enabled = self.screen_observation_enabled_check.isChecked()
        self.result_proactive_care_settings = ProactiveCareSettings(
            enabled=self.proactive_care_enabled_check.isChecked(),
            screen_context_enabled=(
                self.screen_observation_enabled_check.isChecked()
                and self.proactive_screen_context_enabled_check.isChecked()
            ),
            check_interval_minutes=self.proactive_check_interval_spin.value(),
            cooldown_minutes=self.proactive_cooldown_spin.value(),
        )
        super().accept()

    def reject(self) -> None:
        if self._api_test_thread is not None:
            QMessageBox.information(self, "测试中", "API 测试仍在进行，请等待完成后再关闭设置。")
            return
        super().reject()

    def _test_api_settings(self) -> None:
        settings = self._validated_api_settings()
        if settings is None or self._api_test_thread is not None:
            return

        self.api_test_button.setEnabled(False)
        self.api_test_button.setText("测试中...")

        thread = QThread(self)
        worker = ApiConnectionTestWorker(settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._handle_api_test_success)
        worker.failed.connect(self._handle_api_test_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._reset_api_test_state)

        self._api_test_thread = thread
        self._api_test_worker = worker
        thread.start()

    @Slot(str)
    def _handle_api_test_success(self, message: str) -> None:
        QMessageBox.information(self, "测试成功", f"API 连接成功，模型返回：{message}")

    @Slot(str)
    def _handle_api_test_failed(self, message: str) -> None:
        QMessageBox.warning(self, "测试失败", message)

    @Slot()
    def _reset_api_test_state(self) -> None:
        self.api_test_button.setEnabled(True)
        self.api_test_button.setText("测试 API")
        self._api_test_thread = None
        self._api_test_worker = None

    @Slot(bool)
    def _sync_proactive_screen_context_enabled(self, screen_observation_enabled: bool) -> None:
        self.proactive_screen_context_enabled_check.setEnabled(screen_observation_enabled)
        if not screen_observation_enabled:
            self.proactive_screen_context_enabled_check.setChecked(False)

    def _validated_api_settings(self) -> ApiSettings | None:
        base_url = self.base_url_edit.text().strip().rstrip("/")
        api_key = self.api_key_edit.text().strip()
        model = self.model_edit.text().strip()

        if not _is_http_url(base_url):
            QMessageBox.warning(self, "配置无效", "Base URL 必须是有效的 http 或 https 地址。")
            return None
        if not api_key:
            QMessageBox.warning(self, "配置无效", "API Key 不能为空。")
            return None
        if not model:
            QMessageBox.warning(self, "配置无效", "模型不能为空。")
            return None

        return ApiSettings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=self.api_timeout_spin.value(),
        )

    def _validated_tts_settings(self) -> GPTSoVITSTTSSettings | None:
        enabled = self.tts_enabled_check.isChecked()
        api_url = self.tts_api_url_edit.text().strip()
        ref_lang = self.ref_lang_edit.text().strip()
        text_lang = self.text_lang_edit.text().strip()

        if enabled and not _is_http_url(api_url):
            QMessageBox.warning(self, "配置无效", "TTS API URL 必须是有效的 http 或 https 地址。")
            return None

        selected_profile = self._selected_character_profile()
        if selected_profile is not None:
            settings = GPTSoVITSTTSSettings.from_character_profile(
                character_profile=selected_profile,
                enabled=enabled,
                api_url=api_url,
                ref_lang=ref_lang,
                text_lang=text_lang,
                timeout_seconds=self.tts_timeout_spin.value(),
                validate_enabled=False,
            )
        else:
            settings = GPTSoVITSTTSSettings(
                enabled=enabled,
                api_url=api_url,
                ref_audio_path=self.tts_settings.ref_audio_path,
                ref_text_path=self.tts_settings.ref_text_path,
                ref_text=self.tts_settings.ref_text,
                gpt_model_path=self.tts_settings.gpt_model_path,
                sovits_model_path=self.tts_settings.sovits_model_path,
                ref_lang=ref_lang,
                text_lang=text_lang,
                timeout_seconds=self.tts_timeout_spin.value(),
                tone_references=self.tts_settings.tone_references,
            )
        if enabled:
            try:
                settings.validate()
            except TTSConfigError as exc:
                QMessageBox.warning(self, "配置无效", str(exc))
                return None
        return settings

    def _selected_character_id(self) -> str | None:
        if self.character_registry is None or not hasattr(self, "character_combo"):
            return self.current_character.id if self.current_character is not None else None
        character_id = self.character_combo.currentData()
        if isinstance(character_id, str) and character_id.strip():
            return character_id.strip()
        return self.current_character.id if self.current_character is not None else None

    def _selected_character_profile(self) -> CharacterProfile | None:
        character_id = self._selected_character_id()
        if character_id is None or self.character_registry is None:
            return self.current_character
        return self.character_registry.get(character_id)


def _is_http_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)
