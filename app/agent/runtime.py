from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable

from app.agent.actions import AgentAction, AgentEvent, AgentProgress, AgentResult, PendingToolAction
from app.agent.memory import MemoryStore
from app.agent.screen_tools import (
    OBSERVE_SCREEN_TOOL_NAME,
    SCREEN_OBSERVATION_CAPABILITY,
    SCREEN_OBSERVATION_DISABLED_ERROR,
    SCREEN_OBSERVATION_REQUEST_ACTION,
)
from app.agent.tool_registry import ToolExecutionResult, ToolRegistry
from app.api_client import (
    ApiRequestError,
    ChatMessage,
    OpenAICompatibleClient,
    is_vision_unsupported_error,
    messages_contain_image,
)
from app.chat_reply import ChatReply, parse_chat_reply
from app.debug_log import debug_log, summarize_messages
from app.prompt_templates import (
    build_agent_reply_protocol,
    build_context_acquisition_strategy,
    build_event_reply_protocol,
    build_proactive_rules,
)
from app.screen_observation import (
    MANUAL_SCREEN_OBSERVATION_HISTORY_MARKER,
    SCREEN_OBSERVATION_HISTORY_MARKER,
)


MAX_AGENT_STEPS_PER_TURN = 4
MAX_TOOL_CALLS_PER_STEP = 3
MAX_TOOL_CALLS_PER_TURN = 8
MAX_TOOL_RESULT_CHARS = 6000
MAX_PENDING_CONTEXT_MESSAGES = 12
MAX_PENDING_CONTEXT_TEXT_CHARS = 4000
WINDOWS_CLICK_TOOL_NAME = "windows__Click"
WINDOWS_SCREENSHOT_TOOL_NAME = "windows__Screenshot"
WINDOWS_SNAPSHOT_TOOL_NAME = "windows__Snapshot"
WINDOWS_BROWSER_PAGE_CONFLICT_TOOL_NAMES = {
    WINDOWS_CLICK_TOOL_NAME,
    WINDOWS_SCREENSHOT_TOOL_NAME,
    WINDOWS_SNAPSHOT_TOOL_NAME,
    "windows__Type",
    "windows__Scroll",
    "windows__Move",
}
BROWSER_DOM_TOOL_NAMES = {
    "browser__browser_navigate",
    "browser__browser_snapshot",
    "browser__browser_click",
    "browser__browser_type",
    "browser__browser_scroll",
    "browser__browser_wait_for",
}
WEB_BACKGROUND_TOOL_NAMES = {
    "web__web_search",
    "web__fetch_url",
}
ProgressCallback = Callable[[AgentProgress], None]


class AgentRuntime:
    """封装聊天决策链路，为后续工具调用和长期记忆留下扩展点。"""

    def __init__(
        self,
        api_client: OpenAICompatibleClient,
        system_prompt: str,
        reply_tones: list[str] | None = None,
        reply_portraits: list[str] | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.api_client = api_client
        self.system_prompt = system_prompt
        self.reply_tones = [*reply_tones] if reply_tones is not None else []
        self.reply_portraits = [*reply_portraits] if reply_portraits is not None else []
        self.tools = tools or ToolRegistry()
        self.memory = memory or MemoryStore()
        self.model_vision_enabled = True
        self.autonomous_screen_observation_enabled = False

    def update_character(
        self,
        system_prompt: str,
        reply_tones: list[str] | None = None,
        reply_portraits: list[str] | None = None,
    ) -> None:
        """角色切换后同步系统提示词、可用语气和可用立绘列表。"""
        self.system_prompt = system_prompt
        self.reply_tones = [*reply_tones] if reply_tones is not None else []
        self.reply_portraits = [*reply_portraits] if reply_portraits is not None else []

    def set_model_vision_enabled(self, enabled: bool) -> None:
        """允许模型在需要时请求一次当前屏幕截图。"""
        self.model_vision_enabled = enabled

    def set_autonomous_screen_observation_enabled(self, enabled: bool) -> None:
        """允许模型在对话或主动事件中自主决定是否观察屏幕。"""
        self.autonomous_screen_observation_enabled = enabled

    def handle_user_message(
        self,
        messages: list[ChatMessage],
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        turn_started_at = time.perf_counter()
        allow_screen_observation = (
            self.model_vision_enabled
            and self.autonomous_screen_observation_enabled
            and not messages_contain_image(messages)
            and _should_offer_screen_observation(messages)
        )
        debug_log(
            "AgentRuntime",
            "开始处理用户消息",
            {
                "message_count": len(messages),
                "allow_screen_observation": allow_screen_observation,
                "model_vision_enabled": self.model_vision_enabled,
                "autonomous_screen_observation_enabled": self.autonomous_screen_observation_enabled,
                "messages": summarize_messages(messages),
            },
        )
        return self._run_tool_loop(
            messages,
            allow_screen_observation=allow_screen_observation,
            turn_started_at=turn_started_at,
            vision_unsupported_reply=_build_vision_unsupported_reply(),
            progress_callback=progress_callback,
        )

    def _run_tool_loop(
        self,
        messages: list[ChatMessage],
        *,
        allow_screen_observation: bool,
        turn_started_at: float,
        planning_extra_instructions: str = "",
        initial_actions: list[AgentAction] | None = None,
        vision_unsupported_reply: ChatReply | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        """执行受限 Agent 循环：规划工具、执行、回填结果，并允许继续规划。"""
        working_messages: list[ChatMessage] = [*messages]
        execution_results: list[ToolExecutionResult] = []
        emitted_actions: list[AgentAction] = [*(initial_actions or [])]
        total_tool_calls = 0
        for step_index in range(MAX_AGENT_STEPS_PER_TURN):
            browser_page_mode = _should_prefer_browser_page_tools(working_messages)
            browser_page_guard_active = (
                browser_page_mode
                and _browser_dom_tools_available(self.tools)
                and not _recent_browser_tool_failed(working_messages)
                and not _latest_user_explicitly_requests_windows_control(working_messages)
            )
            visible_browser_guard_active = (
                _latest_user_requests_visible_browser(working_messages)
                and _browser_dom_tools_available(self.tools)
                and not _recent_browser_tool_failed(working_messages)
            )
            try:
                planning_started_at = time.perf_counter()
                model_content = self.api_client.complete_raw(
                    self._build_tool_planning_prompt(
                        allow_screen_observation=allow_screen_observation,
                        step_index=step_index,
                        remaining_steps=MAX_AGENT_STEPS_PER_TURN - step_index - 1,
                        extra_instructions=planning_extra_instructions,
                        browser_page_mode=browser_page_guard_active,
                        visible_browser_mode=visible_browser_guard_active,
                    ),
                    working_messages,
                    temperature=0.8,
                )
            except ApiRequestError as exc:
                if messages_contain_image(working_messages) and is_vision_unsupported_error(exc):
                    debug_log("AgentRuntime", "视觉输入不受支持，返回兜底回复", {"error": str(exc)})
                    return AgentResult(
                        reply=vision_unsupported_reply or _build_vision_unsupported_reply(),
                        actions=emitted_actions,
                    )
                raise
            debug_log(
                "AgentRuntime",
                "工具规划模型返回",
                {
                    "step_index": step_index,
                    "content": model_content,
                    "planning_elapsed_ms": int((time.perf_counter() - planning_started_at) * 1000),
                },
            )
            agent_data = _load_json_object(model_content)
            if agent_data is None:
                debug_log(
                    "AgentRuntime",
                    "模型返回非 JSON，按普通回复解析",
                    {
                        "step_index": step_index,
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=parse_chat_reply(model_content),
                    actions=emitted_actions,
                )

            tool_calls = _parse_tool_calls(agent_data.get("tool_calls"))
            debug_log(
                "AgentRuntime",
                "工具规划解析完成",
                {
                    "step_index": step_index,
                    "has_reply": isinstance(agent_data.get("reply"), dict),
                    "tool_calls": tool_calls,
                    "total_tool_calls": total_tool_calls,
                },
            )
            if not tool_calls:
                debug_log(
                    "AgentRuntime",
                    "多步循环完成，返回模型回复",
                    {
                        "step_index": step_index,
                        "tool_result_count": len(execution_results),
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=_parse_agent_reply(agent_data, model_content),
                    actions=emitted_actions,
                )

            _emit_progress(
                progress_callback,
                agent_data,
                stage="tool_planning",
                metadata={
                    "step_index": step_index,
                    "tool_names": [call["name"] for call in tool_calls],
                    "tool_call_count": len(tool_calls),
                },
            )
            step_results: list[ToolExecutionResult] = []
            pending_actions: list[PendingToolAction] = []
            tools_started_at = time.perf_counter()
            allowed_calls = min(
                len(tool_calls),
                MAX_TOOL_CALLS_PER_STEP,
                max(0, MAX_TOOL_CALLS_PER_TURN - total_tool_calls),
            )
            for call in tool_calls[:allowed_calls]:
                total_tool_calls += 1
                debug_log("AgentRuntime", "准备工具调用", {"step_index": step_index, **call})
                if _should_block_windows_tool_for_browser_page(call, browser_page_guard_active):
                    blocked_result = _build_browser_page_windows_tool_block_result(call)
                    debug_log("AgentRuntime", "浏览器页面模式拦截 Windows 工具", blocked_result.to_dict())
                    step_results.append(blocked_result)
                    execution_results.append(blocked_result)
                    emitted_actions.append(
                        AgentAction(
                            type="tool_call",
                            payload=_redact_tool_result_for_model(blocked_result),
                        )
                    )
                    continue
                if _should_block_background_web_tool_for_visible_browser(call, visible_browser_guard_active):
                    blocked_result = _build_visible_browser_web_tool_block_result(call)
                    debug_log("AgentRuntime", "可见浏览器模式拦截后台网页工具", blocked_result.to_dict())
                    step_results.append(blocked_result)
                    execution_results.append(blocked_result)
                    emitted_actions.append(
                        AgentAction(
                            type="tool_call",
                            payload=_redact_tool_result_for_model(blocked_result),
                        )
                    )
                    continue
                prepared = self.tools.prepare_or_execute(
                    call["name"],
                    call["arguments"],
                    call.get("reason", ""),
                )
                if isinstance(prepared, PendingToolAction):
                    prepared = prepared.with_continuation_messages(
                        _build_pending_continuation_messages(working_messages, model_content)
                    )
                    debug_log(
                        "AgentRuntime",
                        "工具调用等待用户确认",
                        {
                            **prepared.to_dict(),
                            "continuation_message_count": len(prepared.continuation_messages),
                        },
                    )
                    pending_actions.append(prepared)
                    continue

                if _is_screen_observation_request(prepared):
                    if allow_screen_observation:
                        screen_action = AgentAction(
                            type=SCREEN_OBSERVATION_REQUEST_ACTION,
                            payload={"reason": call.get("reason", "")},
                        )
                        debug_log(
                            "AgentRuntime",
                            "请求屏幕观察 follow-up",
                            {
                                "step_index": step_index,
                                "reason": call.get("reason", ""),
                                "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                            },
                        )
                        return AgentResult(
                            reply=_build_screen_observation_request_reply(),
                            actions=[*emitted_actions, screen_action],
                        )
                    prepared = ToolExecutionResult(
                        tool_name=OBSERVE_SCREEN_TOOL_NAME,
                        success=False,
                        content="",
                        error=SCREEN_OBSERVATION_DISABLED_ERROR,
                    )

                debug_log("AgentRuntime", "工具调用完成", _redact_tool_result_for_model(prepared))
                step_results.append(prepared)
                execution_results.append(prepared)
                emitted_actions.append(
                    AgentAction(
                        type="tool_call",
                        payload=_redact_tool_result_for_model(prepared),
                    )
                )

            skipped_calls = len(tool_calls) - allowed_calls
            if skipped_calls > 0:
                limit_error = (
                    f"本步骤最多执行 {MAX_TOOL_CALLS_PER_STEP} 个工具调用，"
                    f"整轮最多执行 {MAX_TOOL_CALLS_PER_TURN} 个工具调用，"
                    f"已跳过 {skipped_calls} 个后续调用。"
                )
                limit_result = ToolExecutionResult(
                    tool_name="runtime",
                    success=False,
                    content="",
                    error=limit_error,
                )
                debug_log(
                    "AgentRuntime",
                    "工具调用数量超过上限",
                    {
                        "step_index": step_index,
                        "requested": len(tool_calls),
                        "allowed": allowed_calls,
                        "total_tool_calls": total_tool_calls,
                        "step_limit": MAX_TOOL_CALLS_PER_STEP,
                        "turn_limit": MAX_TOOL_CALLS_PER_TURN,
                    },
                )
                step_results.append(limit_result)
                execution_results.append(limit_result)
                emitted_actions.append(
                    AgentAction(
                        type="tool_call",
                        payload=_redact_tool_result_for_model(limit_result),
                    )
                )

            if pending_actions:
                debug_log(
                    "AgentRuntime",
                    "返回待确认动作",
                    {
                        "step_index": step_index,
                        "pending_actions": [action.to_dict() for action in pending_actions],
                        "tools_elapsed_ms": int((time.perf_counter() - tools_started_at) * 1000),
                        "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
                    },
                )
                return AgentResult(
                    reply=_build_pending_action_reply(pending_actions),
                    actions=[
                        *emitted_actions,
                        *[
                            AgentAction(
                                type="pending_action",
                                payload=action.to_dict(include_context=True),
                            )
                            for action in pending_actions
                        ],
                    ],
                )

            if not step_results:
                break

            working_messages.extend(
                [
                    {"role": "assistant", "content": model_content},
                    _build_tool_results_message(
                        step_results,
                        include_images=self.model_vision_enabled,
                    ),
                ]
            )
            if total_tool_calls >= MAX_TOOL_CALLS_PER_TURN:
                break

        try:
            final_started_at = time.perf_counter()
            final_reply = self.api_client.chat(
                self._build_final_reply_prompt(),
                working_messages,
                self.reply_tones,
                self.reply_portraits,
            )
        except Exception as exc:
            print(f"[AgentRuntime] 工具结果总结失败，使用本地兜底回复：{exc}")
            debug_log("AgentRuntime", "工具结果总结失败，使用本地兜底回复", {"error": str(exc)})
            final_reply = _build_fallback_tool_reply(execution_results)
        debug_log(
            "AgentRuntime",
            "最终回复生成完成",
            {
                "segments": len(final_reply.segments),
                "actions": [_redact_tool_result_for_model(result) for result in execution_results],
                "final_reply_elapsed_ms": int((time.perf_counter() - final_started_at) * 1000),
                "turn_elapsed_ms": int((time.perf_counter() - turn_started_at) * 1000),
            },
        )
        return AgentResult(
            reply=final_reply,
            actions=emitted_actions,
        )

    def handle_confirmed_action(
        self,
        action: PendingToolAction,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        turn_started_at = time.perf_counter()
        debug_log("AgentRuntime", "执行已确认动作", action.to_dict())
        result = self.tools.execute(action.tool_name, action.arguments)
        results = [result]
        verification_result = _verify_confirmed_windows_click(self.tools, action.tool_name)
        if verification_result is not None:
            results.append(verification_result)
        emitted_actions = [
            AgentAction(
                type="tool_call",
                payload=_redact_tool_result_for_model(item),
            )
            for item in results
        ]
        if action.continuation_messages:
            working_messages = [
                *action.continuation_messages,
                _build_confirmed_action_result_message(action, results),
            ]
            allow_screen_observation = (
                self.model_vision_enabled
                and self.autonomous_screen_observation_enabled
                and not messages_contain_image(working_messages)
                and _should_offer_screen_observation(working_messages)
            )
            debug_log(
                "AgentRuntime",
                "已确认动作接回 Agent 循环",
                {
                    "tool_name": action.tool_name,
                    "message_count": len(working_messages),
                    "allow_screen_observation": allow_screen_observation,
                },
            )
            return self._run_tool_loop(
                working_messages,
                allow_screen_observation=allow_screen_observation,
                turn_started_at=turn_started_at,
                planning_extra_instructions=_build_confirmed_action_continuation_rules(action),
                initial_actions=emitted_actions,
                progress_callback=progress_callback,
            )
        try:
            reply = self.api_client.chat(
                self._build_final_reply_prompt(),
                [
                    _build_confirmed_action_result_message(action, results),
                ],
                self.reply_tones,
                self.reply_portraits,
            )
        except Exception as exc:
            print(f"[AgentRuntime] 确认动作总结失败，使用本地兜底回复：{exc}")
            debug_log("AgentRuntime", "确认动作总结失败，使用本地兜底回复", {"error": str(exc)})
            reply = _build_fallback_tool_reply(results)
        debug_log(
            "AgentRuntime",
            "已确认动作处理完成",
            {
                "results": [_redact_tool_result_for_model(item) for item in results],
                "segments": len(reply.segments),
            },
        )
        return AgentResult(
            reply=reply,
            actions=emitted_actions,
        )

    def handle_cancelled_action(self, action: PendingToolAction) -> AgentResult:
        debug_log("AgentRuntime", "用户取消待确认动作", action.to_dict())
        return AgentResult(
            reply=parse_chat_reply(
                json.dumps(
                    {
                        "segments": [
                            {
                                "ja": "わかった。実行しないでおくね。",
                                "zh": "知道了。我不会执行这个动作。",
                                "tone": "中性",
                                "portrait": "站立待机",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            ),
            actions=[
                AgentAction(
                    type="cancelled_action",
                    payload=action.to_dict(),
                )
            ],
        )

    def handle_event(
        self,
        event: AgentEvent,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        if event.type not in {"reminder_due", "proactive_check"}:
            return AgentResult(reply=parse_chat_reply("未対応のイベントだよ。"))

        debug_log("AgentRuntime", "处理主动事件", {"event": {"type": event.type, "payload": event.payload}})
        event_messages = _build_event_messages(event)
        event_action = AgentAction(
            type="event",
            payload={
                "event_type": event.type,
                "event_payload": event.payload,
            },
        )
        if event.type == "proactive_check":
            screen_context_allowed = bool(event.payload.get("screen_context_allowed"))
            allow_screen_observation = (
                screen_context_allowed
                and not messages_contain_image(event_messages)
            )
            return self._run_tool_loop(
                event_messages,
                allow_screen_observation=allow_screen_observation,
                turn_started_at=time.perf_counter(),
                planning_extra_instructions=_build_proactive_tool_loop_rules(),
                initial_actions=[event_action],
                vision_unsupported_reply=_build_proactive_vision_unsupported_reply(),
                progress_callback=progress_callback,
            )

        try:
            reply = self.api_client.chat(
                self._build_event_reply_prompt(event.type),
                event_messages,
                self.reply_tones,
                self.reply_portraits,
            )
        except ApiRequestError as exc:
            if messages_contain_image(event_messages) and is_vision_unsupported_error(exc):
                debug_log("AgentRuntime", "主动事件视觉输入不受支持，返回兜底回复", {"error": str(exc)})
                return AgentResult(reply=_build_proactive_vision_unsupported_reply())
            raise
        return AgentResult(
            reply=reply,
            actions=[event_action],
        )

    def _build_tool_planning_prompt(
        self,
        allow_screen_observation: bool = False,
        step_index: int = 0,
        remaining_steps: int = MAX_AGENT_STEPS_PER_TURN - 1,
        extra_instructions: str = "",
        browser_page_mode: bool = False,
        visible_browser_mode: bool = False,
    ) -> str:
        allowed_capabilities = {SCREEN_OBSERVATION_CAPABILITY} if allow_screen_observation else set()
        tools = _filter_tools_for_browser_routing(
            self.tools.describe_tools(allowed_capabilities=allowed_capabilities),
            browser_page_mode=browser_page_mode,
            visible_browser_mode=visible_browser_mode,
        )
        tool_descriptions = json.dumps(
            tools,
            ensure_ascii=False,
            indent=2,
        )
        memory_summary = self._memory_summary()
        current_time = datetime.now().astimezone().isoformat(timespec="seconds")
        reply_protocol = build_agent_reply_protocol(self.reply_tones, self.reply_portraits)
        context_strategy = build_context_acquisition_strategy(
            allow_screen_observation=allow_screen_observation
        )
        screen_observation_rule = _build_screen_and_desktop_routing_rule(allow_screen_observation)
        browser_page_rule = _build_browser_page_mode_rule(browser_page_mode)
        visible_browser_rule = _build_visible_browser_mode_rule(visible_browser_mode)
        return f"""
{self.system_prompt.strip()}

你现在可以作为桌面陪伴型 Agent 判断是否需要调用内部工具。
如果需要工具，返回 reply 和 tool_calls；如果不需要工具，tool_calls 返回空数组或省略。
输出必须是单个 JSON object；不要添加 Markdown、代码块、反引号、工具名伪代码或 JSON 外的解释文字。

长期记忆摘要：
{memory_summary}

当前本地时间：
{current_time}

当前 Agent 循环：
- 这是第 {step_index + 1} 步，之后最多还可以继续 {remaining_steps} 步。
- 你可以根据已有工具结果继续请求下一批工具；信息足够或已经完成时，tool_calls 必须为空，并在 reply 中给最终答复。
- 每步最多请求 {MAX_TOOL_CALLS_PER_STEP} 个工具，整轮最多 {MAX_TOOL_CALLS_PER_TURN} 个工具；不要为了凑数量而调用工具。
- 不要重复调用刚失败且参数相同的工具；如果受限、需要确认或信息不足，请停止循环并说明当前状态。

可用工具：
{tool_descriptions}

{reply_protocol}

{context_strategy}

工具要求：
- 如果需要调用工具，reply 只写执行前可以直接说给用户听的短句，例如“我先查一下”“我看一下屏幕”；不要提前给最终结论。
- 如果工具可以帮助完成用户请求，优先用 tool_calls 表达要执行的动作。
- 不要臆造工具名；只能使用上面列出的工具。
- requires_confirmation 为 true 的工具只会在用户确认后执行；你仍然可以发起 tool_calls，但必须说明原因。
- 浏览器内部任务优先使用 browser__ 前缀的 Playwright MCP 工具；先用 browser__browser_snapshot 获取页面结构，再基于真实 target 调用点击、输入、表单、等待等工具。
- 桌面窗口、应用切换、鼠标坐标、快捷键等浏览器外部任务才使用 windows__ 前缀的 Windows-MCP 工具；不要用 windows__Click/Move/Type 操作普通网页内部元素。
- 对桌面窗口执行点击、移动、输入前，必须先用 windows__Snapshot 或 windows__Screenshot 获取真实窗口状态；优先使用 Snapshot 返回的 UI label/id 作为 Click 参数。
- 如果 Windows MCP 截图结果显示 Available Displays 有多个显示器，或 Screenshot Original Size 明显大于单屏，禁止直接基于全虚拟桌面截图里的小图标、缩小坐标或标号执行 Click/Move/Type；必须先再次调用 windows__Snapshot / windows__Screenshot 并传入 display=[0]、display=[1] 等限定到目标所在显示器后再选择 label/id。桌面左上角、回收站等桌面图标默认先检查 display=[0]，看不到再检查其他 display。
- 如果只能基于截图坐标点击，必须结合工具结果里的 Screenshot Original Size、Screenshot Region、显示尺寸和缩放比例，把图像坐标换算为真实虚拟桌面坐标；不要使用 Sakura 内置视觉观察的缩放图片坐标。
- 推理出桌面点击目标后，发起 windows__Click 并等待用户确认；确认执行后 Runtime 会自动用 Windows MCP 截图验证结果。
{screen_observation_rule}
{browser_page_rule}
{visible_browser_rule}
- 如果 browser__ 工具不可用，读取 Sakura 受控浏览器内容时再使用 browser_get_content；需要打开受控浏览器网页时使用 browser_open_url。
- 需要网页交互时，只能基于当前页面真实内容选择工具，不要臆造 selector、target 或页面内容。
{extra_instructions.strip()}
- 用户说“几分钟后/几秒后/一会儿后”等相对提醒时，add_reminder 必须使用 delay_minutes 或 delay_seconds，不要自己换算 trigger_at。
- 只有用户给出明确日期或钟点时，add_reminder 才使用 trigger_at。
- 长期记忆由后台整理器自动维护；你不要尝试写入记忆。
- 只有用户明确要求忘掉信息时，才使用 forget_memory。
""".strip()
    def _build_final_reply_prompt(self) -> str:
        return f"""
{self.system_prompt.strip()}

你会收到上一轮工具调用结果。请基于这些结果给用户最终回复。
不要再次请求工具，不要提及内部 JSON、工具协议或实现细节。
""".strip()

    def _build_event_reply_prompt(self, event_type: str = "reminder_due") -> str:
        proactive_rules = ""
        example_tone = "提醒"
        if event_type == "proactive_check":
            example_tone = "中性"
            proactive_rules = build_proactive_rules()
        reply_protocol = build_event_reply_protocol(
            self.reply_tones,
            self.reply_portraits,
            example_tone=example_tone,
        )
        return f"""
{self.system_prompt.strip()}

你正在处理 Sakura 桌宠的主动事件。请用角色语气自然搭话、提问或提醒用户。
{reply_protocol}
- 不要提及内部事件类型、JSON 或工具实现。
{proactive_rules}
""".strip()

    def _memory_summary(self) -> str:
        try:
            return self.memory.summary()
        except Exception as exc:
            return f"长期记忆读取失败：{exc}"


def _load_json_object(content: str) -> dict[str, Any] | None:
    text = _strip_code_fence(content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data

    fallback: dict[str, Any] | None = None
    for candidate in _iter_json_object_candidates(content):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if fallback is None:
            fallback = data
        if _looks_like_agent_payload(data):
            return data
    return fallback


def _looks_like_agent_payload(data: dict[str, Any]) -> bool:
    return "reply" in data or "tool_calls" in data


def _iter_json_object_candidates(content: str):
    """从混杂模型输出中抽取可能的 JSON object，避免泄露规划协议到前台。"""
    text = _strip_code_fence(content.strip())
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _strip_code_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _parse_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments", {})
        reason = item.get("reason", "")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(reason, str):
            reason = ""
        tool_calls.append({"name": name.strip(), "arguments": arguments, "reason": reason.strip()})
    return tool_calls


def _parse_agent_reply(agent_data: dict[str, Any], fallback_content: str) -> ChatReply:
    reply_data = agent_data.get("reply")
    if isinstance(reply_data, dict):
        return parse_chat_reply(json.dumps(reply_data, ensure_ascii=False))
    return parse_chat_reply(fallback_content)


def _parse_progress_reply(agent_data: dict[str, Any]) -> ChatReply | None:
    reply_data = agent_data.get("reply")
    if not isinstance(reply_data, dict):
        return None
    reply = parse_chat_reply(json.dumps(reply_data, ensure_ascii=False))
    return reply if reply.text.strip() else None


def _emit_progress(
    progress_callback: ProgressCallback | None,
    agent_data: dict[str, Any],
    *,
    stage: str,
    metadata: dict[str, Any],
) -> None:
    if progress_callback is None:
        return
    reply = _parse_progress_reply(agent_data)
    if reply is None:
        return
    try:
        progress_callback(AgentProgress(reply=reply, stage=stage, metadata=metadata))
    except Exception as exc:
        debug_log("AgentRuntime", "中间回复回调失败，已忽略", {"error": str(exc), "stage": stage})


def _is_screen_observation_request(result: ToolExecutionResult) -> bool:
    if result.tool_name != OBSERVE_SCREEN_TOOL_NAME or not result.success:
        return False
    if not isinstance(result.content, dict):
        return False
    return result.content.get("action") == SCREEN_OBSERVATION_REQUEST_ACTION


def _verify_confirmed_windows_click(
    tools: ToolRegistry,
    tool_name: str,
) -> ToolExecutionResult | None:
    """Windows 桌面点击后追加一次只读截图验证。"""
    if tool_name != WINDOWS_CLICK_TOOL_NAME:
        return None

    screenshot_tool = tools.get(WINDOWS_SCREENSHOT_TOOL_NAME)
    snapshot_tool = tools.get(WINDOWS_SNAPSHOT_TOOL_NAME)

    screenshot_result: ToolExecutionResult | None = None
    if screenshot_tool is not None:
        screenshot_result = tools.execute(WINDOWS_SCREENSHOT_TOOL_NAME, {})
        if screenshot_result.success or snapshot_tool is None:
            return screenshot_result

    if snapshot_tool is not None:
        snapshot_result = tools.execute(
            WINDOWS_SNAPSHOT_TOOL_NAME,
            {
                "use_vision": True,
                "use_ui_tree": False,
            },
        )
        if snapshot_result.success or screenshot_result is None:
            return snapshot_result
        return ToolExecutionResult(
            tool_name="windows__verification",
            success=False,
            content="",
            error=(
                f"Screenshot 验证失败：{screenshot_result.error or '未知错误'}；"
                f"Snapshot 验证失败：{snapshot_result.error or '未知错误'}"
            ),
        )

    return ToolExecutionResult(
        tool_name="windows__verification",
        success=False,
        content="",
        error="没有可用的 windows__Screenshot 或 windows__Snapshot，无法自动验证点击结果。",
    )


def _filter_tools_for_browser_routing(
    tools: list[dict[str, Any]],
    *,
    browser_page_mode: bool,
    visible_browser_mode: bool,
) -> list[dict[str, Any]]:
    """按浏览器路由模式隐藏容易诱导模型走错路径的工具。"""
    if not browser_page_mode and not visible_browser_mode:
        return tools
    hidden_names: set[str] = set()
    if browser_page_mode:
        hidden_names.update(WINDOWS_BROWSER_PAGE_CONFLICT_TOOL_NAMES)
    if visible_browser_mode:
        hidden_names.update(WEB_BACKGROUND_TOOL_NAMES)
    return [tool for tool in tools if str(tool.get("name", "")) not in hidden_names]


def _should_block_windows_tool_for_browser_page(
    call: dict[str, Any],
    browser_page_mode: bool,
) -> bool:
    if not browser_page_mode:
        return False
    return str(call.get("name", "")) in WINDOWS_BROWSER_PAGE_CONFLICT_TOOL_NAMES


def _should_block_background_web_tool_for_visible_browser(
    call: dict[str, Any],
    visible_browser_mode: bool,
) -> bool:
    if not visible_browser_mode:
        return False
    return str(call.get("name", "")) in WEB_BACKGROUND_TOOL_NAMES


def _build_browser_page_windows_tool_block_result(call: dict[str, Any]) -> ToolExecutionResult:
    tool_name = str(call.get("name", "")).strip() or "unknown"
    return ToolExecutionResult(
        tool_name="runtime",
        success=False,
        content={
            "blocked_tool": tool_name,
            "reason": "当前上下文是浏览器页面内部操作，已阻止 Windows-MCP 坐标/截图工具抢路由。",
            "guidance": (
                "请先使用 browser__browser_snapshot 获取页面结构，"
                "再基于真实 target 调用 browser__browser_click、"
                "browser__browser_type、browser__browser_wait_for 等 Playwright/browser MCP 工具。"
            ),
        },
        error=f"已阻止 {tool_name}：浏览器页面内部操作应优先使用 browser__ 工具。",
    )


def _build_visible_browser_web_tool_block_result(call: dict[str, Any]) -> ToolExecutionResult:
    tool_name = str(call.get("name", "")).strip() or "unknown"
    return ToolExecutionResult(
        tool_name="runtime",
        success=False,
        content={
            "blocked_tool": tool_name,
            "reason": "用户明确要求打开浏览器或看到搜索过程，已阻止后台网页搜索/抓取工具。",
            "guidance": (
                "请使用 browser__browser_navigate 打开搜索引擎或目标网址，"
                "再用 browser__browser_snapshot 获取页面结构，并用 browser__browser_type、"
                "browser__browser_click、browser__browser_wait_for 等工具完成可见浏览器流程。"
            ),
        },
        error=f"已阻止 {tool_name}：显式浏览器任务应使用 browser__ 工具，不要只做后台搜索。",
    )


def _browser_dom_tools_available(tools: ToolRegistry) -> bool:
    return any(tools.get(name) is not None for name in BROWSER_DOM_TOOL_NAMES)


def _should_prefer_browser_page_tools(messages: list[ChatMessage]) -> bool:
    text = _messages_text_for_tool_routing(messages).lower()
    if "browser__" in text:
        return True

    latest_text = (_latest_user_text(messages) or "").lower()
    if not latest_text:
        return False
    browser_keywords = (
        "浏览器",
        "网页",
        "页面",
        "链接",
        "搜索结果",
        "搜索框",
        "输入框",
        "点进",
        "点开",
        "打开网页",
        "标签页",
        "网址",
        "url",
        "http://",
        "https://",
        "百科",
        "必应",
        "bing",
        "百度",
        "google",
    )
    return any(keyword in latest_text for keyword in browser_keywords)


def _latest_user_requests_visible_browser(messages: list[ChatMessage]) -> bool:
    text = (_latest_user_text(messages) or "").lower()
    if not text:
        return False
    visible_browser_keywords = (
        "打开浏览器",
        "用浏览器",
        "浏览器搜索",
        "在浏览器",
        "打开网页",
        "打开页面",
        "看搜索过程",
        "看到搜索过程",
        "让我看到",
        "给我看搜索",
        "搜给我看",
        "可见浏览器",
        "前台浏览器",
    )
    return any(keyword in text for keyword in visible_browser_keywords)


def _recent_browser_tool_failed(messages: list[ChatMessage]) -> bool:
    recent_text = _messages_text_for_tool_routing(messages[-4:]).lower()
    return (
        "browser__" in recent_text
        and (
            '"success": false' in recent_text
            or '"success":false' in recent_text
            or "'success': false" in recent_text
            or "'success':false" in recent_text
            or '"is_error": true' in recent_text
            or '"is_error":true' in recent_text
            or "'is_error': true" in recent_text
            or "'is_error':true" in recent_text
            or "工具执行异常" in recent_text
            or "工具执行失败" in recent_text
        )
    )


def _latest_user_explicitly_requests_windows_control(messages: list[ChatMessage]) -> bool:
    text = (_latest_user_text(messages) or "").lower()
    if not text:
        return False
    explicit_keywords = (
        "真实鼠标",
        "物理鼠标",
        "鼠标",
        "坐标",
        "windows",
        "桌面",
        "窗口",
        "浏览器窗口",
        "地址栏",
        "任务栏",
        "快捷键",
        "键盘",
        "系统界面",
    )
    return any(keyword in text for keyword in explicit_keywords)


def _messages_text_for_tool_routing(messages: list[ChatMessage]) -> str:
    return "\n".join(_compact_pending_context_content(message.get("content")) for message in messages)


def _build_browser_page_mode_rule(browser_page_mode: bool) -> str:
    if not browser_page_mode:
        return ""
    return (
        "- 当前上下文已识别为浏览器页面内部操作模式：Windows-MCP 坐标、截图、输入、滚动工具已从可用工具中隐藏。"
        "继续点击链接、输入搜索词、滚动页面或等待页面变化时，必须使用 browser__ 前缀的 Playwright MCP 工具。"
    )


def _build_visible_browser_mode_rule(visible_browser_mode: bool) -> str:
    if not visible_browser_mode:
        return ""
    return (
        "- 用户明确要求打开浏览器或看到搜索过程：后台 web__ 搜索/抓取工具已从可用工具中隐藏。"
        "必须使用 browser__browser_navigate/type/click/snapshot/wait_for 等工具完成可见浏览器搜索流程。"
    )


def _build_screen_and_desktop_routing_rule(allow_screen_observation: bool) -> str:
    if allow_screen_observation:
        return "\n".join(
            [
                "- 当用户询问当前屏幕内容、可见文字、报错含义、界面状态或“这个是什么意思”时，优先调用 observe_screen；这是 Sakura 内置视觉观察，只用于理解画面和解释，不用于鼠标坐标。",
                "- 当用户要求你点击、移动鼠标、输入、切换窗口或操作桌面应用时，不要用 observe_screen 推理坐标；改用 Windows MCP 的 windows__Snapshot / windows__Screenshot 作为操作前观察。",
            ]
        )
    return "\n".join(
        [
            "- 当前没有可用的 Sakura 内置屏幕理解工具；不要臆造当前屏幕内容。",
            "- 如果用户要求桌面点击、移动鼠标、输入或窗口操作，并且 Windows MCP 截图工具可用，先用 windows__Snapshot / windows__Screenshot 获取真实桌面状态。",
        ]
    )


def _should_offer_screen_observation(messages: list[ChatMessage]) -> bool:
    """只在当前轮仍有可关联用户消息时开放自主屏幕观察。"""
    text = _latest_user_text(messages)
    if text is None:
        return False
    return (
        SCREEN_OBSERVATION_HISTORY_MARKER not in text
        and MANUAL_SCREEN_OBSERVATION_HISTORY_MARKER not in text
    )


def _latest_user_text(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(parts)
        return ""
    return None


def _build_pending_continuation_messages(
    working_messages: list[ChatMessage],
    model_content: str,
) -> list[ChatMessage]:
    """为待确认动作保存轻量上下文，避免确认执行后丢失原任务。"""
    messages = [
        *_compact_messages_for_pending_context(working_messages),
        {
            "role": "assistant",
            "content": _truncate_pending_context_text(model_content),
        },
    ]
    return messages[-MAX_PENDING_CONTEXT_MESSAGES:]


def _compact_messages_for_pending_context(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [_compact_message_for_pending_context(message) for message in messages]


def _compact_message_for_pending_context(message: ChatMessage) -> ChatMessage:
    role = message.get("role")
    return {
        "role": role if isinstance(role, str) and role else "user",
        "content": _compact_pending_context_content(message.get("content")),
    }


def _compact_pending_context_content(content: Any) -> str:
    if isinstance(content, str):
        return _truncate_pending_context_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text", "")
                parts.append(_truncate_pending_context_text(str(text)))
            elif part.get("type") == "image_url":
                parts.append("[图片内容已省略，确认后继续时请根据文本工具结果判断。]")
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    try:
        text = json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        text = str(content)
    return _truncate_pending_context_text(text)


def _truncate_pending_context_text(text: str) -> str:
    if len(text) <= MAX_PENDING_CONTEXT_TEXT_CHARS:
        return text
    head_chars = max(1, MAX_PENDING_CONTEXT_TEXT_CHARS // 2)
    tail_chars = MAX_PENDING_CONTEXT_TEXT_CHARS - head_chars
    return (
        text[:head_chars]
        + f"\n...[已省略 {len(text) - head_chars - tail_chars} 字确认上下文]...\n"
        + text[-tail_chars:]
    )


def _build_tool_results_message(
    results: list[ToolExecutionResult],
    include_images: bool = False,
) -> ChatMessage:
    text = _format_tool_results_for_model(results)
    images = _extract_tool_result_images(results) if include_images else []
    if not images:
        return {"role": "user", "content": text}

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "low",
            },
        }
        for image_url in images
    )
    return {"role": "user", "content": content}


def _build_confirmed_action_result_message(
    action: PendingToolAction,
    results: list[ToolExecutionResult],
) -> ChatMessage:
    text = (
        "用户刚刚确认并执行了一个待确认工具动作。"
        "这不是新的用户任务，请结合此前上下文继续完成原请求；"
        "如果该动作只是中间步骤，不要把当前窗口状态误当成新问题。\n"
        f"已确认动作：{action.tool_name}\n"
        f"动作参数：{json.dumps(action.arguments, ensure_ascii=False, default=str)}\n"
        f"动作原因：{action.reason or '未提供'}\n\n"
        + _format_tool_results_for_model(results)
    )
    return {"role": "user", "content": text}


def _build_confirmed_action_continuation_rules(action: PendingToolAction) -> str:
    rules = [
        "确认动作续接规则：",
        f"- 用户刚刚确认执行了 {action.tool_name}，这只是前一轮任务的一个中间步骤。",
        "- 不要把工具执行后的界面当成用户发起的新闲聊问题；必须回到前文的原始用户目标继续推进。",
        "- 如果动作成功但任务尚未完成，请继续请求下一步必要工具；如果已经完成，再给最终回复。",
        "- 如果刚打开的是 Windows“运行”窗口，且前文已经计划通过命令完成任务，应继续输入/提交对应命令，而不是询问用户想使用什么工具。",
    ]
    if action.tool_name.startswith("browser__"):
        rules.append(
            "- 刚确认执行的是 browser__ 工具，后续网页内点击、输入、滚动、等待页面变化仍应继续使用 browser__ 工具；不要因为页面可见就切换到 windows__ 坐标点击。"
        )
    return "\n".join(rules)


def _format_tool_results_for_model(results: list[ToolExecutionResult]) -> str:
    return (
        "工具执行结果如下，请据此给用户最终回复。"
        "如果工具结果标记已附加浏览器截图，请结合截图兜底判断页面内容，不要臆造看不到的信息：\n"
        + json.dumps(
            [_redact_tool_result_for_model(result) for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )


def _redact_tool_result_for_model(result: ToolExecutionResult) -> dict[str, Any]:
    data = result.to_dict()
    content = data.get("content")
    if isinstance(content, str):
        data["content"] = _truncate_text_for_model(content, MAX_TOOL_RESULT_CHARS)
        return data
    if not isinstance(content, dict):
        return data

    redacted, image_count = _redact_tool_images_from_content(content)
    if image_count:
        redacted["screenshot_attached"] = True
        redacted["screenshot_image_count"] = image_count
    data["content"] = _truncate_value_for_model(redacted, MAX_TOOL_RESULT_CHARS)
    return data


def _truncate_value_for_model(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return value
    head_chars = max(1, max_chars // 2)
    tail_chars = max(0, max_chars - head_chars)
    return {
        "truncated": True,
        "original_chars": len(text),
        "omitted_chars": max(0, len(text) - head_chars - tail_chars),
        "head": text[:head_chars],
        "tail": text[-tail_chars:] if tail_chars else "",
    }


def _truncate_text_for_model(text: str, max_chars: int) -> str | dict[str, Any]:
    if len(text) <= max_chars:
        return text
    head_chars = max(1, max_chars // 2)
    tail_chars = max(0, max_chars - head_chars)
    return {
        "truncated": True,
        "original_chars": len(text),
        "omitted_chars": max(0, len(text) - head_chars - tail_chars),
        "head": text[:head_chars],
        "tail": text[-tail_chars:] if tail_chars else "",
    }


def _extract_tool_result_images(results: list[ToolExecutionResult]) -> list[str]:
    images: list[str] = []
    for result in results:
        if not isinstance(result.content, dict):
            continue
        images.extend(_extract_image_data_urls_from_value(result.content))
    return images[:1]


def _redact_tool_images_from_content(content: dict[str, Any]) -> tuple[dict[str, Any], int]:
    image_count = 0

    def redact(value: Any) -> Any:
        nonlocal image_count
        if isinstance(value, dict):
            if _mcp_image_item_to_data_url(value) is not None:
                image_count += 1
                return {
                    "type": value.get("type", "image"),
                    "image_attached": True,
                    "mime_type": _mcp_image_mime_type(value),
                }
            redacted_dict: dict[str, Any] = {}
            for key, item in value.items():
                if key in {"screenshot_data_url", "mcp_image_data_urls"}:
                    if isinstance(item, str) and item.startswith("data:image/"):
                        image_count += 1
                    elif isinstance(item, list):
                        image_count += len(
                            [
                                image_url
                                for image_url in item
                                if isinstance(image_url, str) and image_url.startswith("data:image/")
                            ]
                        )
                    continue
                redacted_dict[str(key)] = redact(item)
            return redacted_dict
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    redacted = redact(content)
    return redacted if isinstance(redacted, dict) else {}, image_count


def _extract_image_data_urls_from_value(value: Any) -> list[str]:
    images: list[str] = []
    if isinstance(value, dict):
        screenshot = value.get("screenshot_data_url")
        if isinstance(screenshot, str) and screenshot.startswith("data:image/"):
            images.append(screenshot)

        mcp_images = value.get("mcp_image_data_urls")
        if isinstance(mcp_images, list):
            images.extend(
                image_url
                for image_url in mcp_images
                if isinstance(image_url, str) and image_url.startswith("data:image/")
            )

        data_url = _mcp_image_item_to_data_url(value)
        if data_url is not None:
            images.append(data_url)

        for item in value.values():
            images.extend(_extract_image_data_urls_from_value(item))
    elif isinstance(value, list):
        for item in value:
            images.extend(_extract_image_data_urls_from_value(item))
    return _deduplicate_preserving_order(images)


def _mcp_image_item_to_data_url(item: dict[str, Any]) -> str | None:
    if str(item.get("type", "")).lower() != "image":
        return None
    data = item.get("data")
    if not isinstance(data, str) or not data.strip():
        return None
    if data.startswith("data:image/"):
        return data
    mime_type = _mcp_image_mime_type(item)
    if not mime_type.startswith("image/"):
        return None
    return f"data:{mime_type};base64,{data}"


def _mcp_image_mime_type(item: dict[str, Any]) -> str:
    mime_type = item.get("mimeType")
    if not isinstance(mime_type, str) or not mime_type.strip():
        mime_type = item.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.strip():
        mime_type = "image/png"
    return mime_type.strip()


def _deduplicate_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _build_pending_action_reply(actions: list[PendingToolAction]) -> ChatReply:
    if len(actions) == 1:
        action = actions[0]
        text = _describe_pending_action(action)
        return parse_chat_reply(
            json.dumps(
                {
                    "segments": [
                        {
                            "ja": "実行する前に確認させて。",
                            "zh": f"执行前需要你确认：{text}",
                            "tone": "提醒",
                            "portrait": "伸手命令",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "いくつか確認が必要な操作があるよ。",
                        "zh": f"有 {len(actions)} 个动作需要你确认，我会先处理第一个。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _describe_pending_action(action: PendingToolAction) -> str:
    if action.tool_name == "open_url":
        return f"打开网页 {action.arguments.get('url', '')}"
    if action.tool_name == "browser_open_url":
        return f"在受控浏览器中打开网页 {action.arguments.get('url', '')}"
    if action.tool_name == "browser_scroll":
        direction = action.arguments.get("direction", "")
        amount = action.arguments.get("amount", "")
        return f"滚动受控浏览器页面 {direction} {amount}"
    if action.tool_name == "browser_click":
        return f"点击受控浏览器页面元素 {action.arguments.get('selector', '')}"
    if action.tool_name == "open_local_folder":
        return f"打开文件夹 {action.arguments.get('path', '')}"
    if action.tool_name.startswith("browser__"):
        return f"执行浏览器 MCP 操作 {action.tool_name.removeprefix('browser__')}"
    if action.tool_name.startswith("windows__"):
        return f"执行 Windows 桌面 MCP 操作 {action.tool_name.removeprefix('windows__')}"
    return f"执行 {action.tool_name}"


def _build_screen_observation_request_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "画面を確認してから答えるね。",
                        "zh": "我先看一下当前画面再回答。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_fallback_tool_reply(results: list[ToolExecutionResult]) -> ChatReply:
    if not results:
        return parse_chat_reply("ツール結果の確認に失敗したよ。")

    succeeded = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    if succeeded and not failed:
        summary = _summarize_tool_results(succeeded)
        return parse_chat_reply(
            json.dumps(
                {
                    "segments": [
                        {
                            "ja": f"処理は終わったよ。{summary}",
                            "zh": f"已经处理好了。{summary}",
                            "tone": "提醒",
                            "portrait": "自信拍胸",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

    error_text = "；".join(
        f"{result.tool_name}: {result.error or '执行失败'}"
        for result in failed
    )
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "処理中に問題が起きたみたい。設定かネットワークを確認して。",
                        "zh": f"工具执行时出了点问题：{error_text}",
                        "tone": "困惑",
                        "portrait": "张嘴疑问",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_vision_unsupported_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "今のモデルでは画像を見られないみたい。画面の内容は勝手に想像しないでおくね。",
                        "zh": "当前模型或接口似乎不支持图片输入。我不会猜屏幕内容，请换成支持视觉的模型后再试。",
                        "tone": "困惑",
                        "portrait": "张嘴疑问",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _summarize_tool_results(results: list[ToolExecutionResult]) -> str:
    parts: list[str] = []
    for result in results:
        if isinstance(result.content, dict):
            if isinstance(result.content.get("reminder"), dict):
                reminder = result.content["reminder"]
                text = reminder.get("text", "")
                trigger_at = reminder.get("trigger_at", "")
                parts.append(f"提醒「{text}」已设置在 {trigger_at}。")
            elif isinstance(result.content.get("task"), dict):
                task = result.content["task"]
                parts.append(f"待办「{task.get('text', '')}」已更新。")
            elif isinstance(result.content.get("memory"), dict):
                memory = result.content["memory"]
                parts.append(f"记忆「{memory.get('content', '')}」已更新。")
            elif result.tool_name == "open_url":
                parts.append(f"网页已打开：{result.content.get('url', '')}。")
            elif result.tool_name == "browser_open_url":
                parts.append(f"受控浏览器已打开：{result.content.get('url', '')}。")
            elif result.tool_name == "browser_get_content":
                title = result.content.get("title", "")
                text = result.content.get("text", "")
                parts.append(f"网页内容已读取：{title}。{str(text)[:120]}")
            elif result.tool_name == "browser_scroll":
                parts.append(f"页面已滚动到 Y={result.content.get('scroll_y', '')}。")
            elif result.tool_name == "browser_click":
                parts.append(f"页面元素已点击：{result.content.get('selector', '')}。")
            elif result.tool_name == "browser_get_state":
                parts.append(
                    f"当前网页：{result.content.get('title', '')}，"
                    f"滚动位置 Y={result.content.get('scroll_y', '')}。"
                )
            elif result.tool_name == "open_local_folder":
                parts.append(f"文件夹已打开：{result.content.get('path', '')}。")
            elif result.tool_name == "read_note":
                parts.append(f"笔记「{result.content.get('name', '')}」已读取。")
            elif result.tool_name == "write_note":
                parts.append(f"笔记「{result.content.get('name', '')}」已保存。")
            else:
                parts.append(f"{result.tool_name} 已完成。")
        else:
            parts.append(f"{result.tool_name} 已完成。")
    return " ".join(part for part in parts if part).strip()


def _build_event_messages(event: AgentEvent) -> list[ChatMessage]:
    text = _format_event_for_model(event)
    image_parts = _build_event_screen_context_image_parts(event.payload)
    if not image_parts:
        return [{"role": "user", "content": text}]

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
                *image_parts,
            ],
        }
    ]


def _build_event_screen_context_image_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    screen_contexts = payload.get("screen_contexts")
    image_parts: list[dict[str, Any]] = []
    if isinstance(screen_contexts, list):
        for screen_context in screen_contexts:
            if isinstance(screen_context, dict):
                image_part = _build_screen_context_image_part(screen_context)
                if image_part is not None:
                    image_parts.append(image_part)
    if image_parts:
        return image_parts

    screen_context = payload.get("screen_context")
    if isinstance(screen_context, dict):
        image_part = _build_screen_context_image_part(screen_context)
        if image_part is not None:
            return [image_part]
    return []


def _build_screen_context_image_part(screen_context: dict[str, Any]) -> dict[str, Any] | None:
    data_url = screen_context.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
            "detail": "low",
        },
    }


def _format_event_for_model(event: AgentEvent) -> str:
    instruction = (
        "主动事件如下，请基于屏幕或事件内容生成要直接说给用户听的自然搭话："
        if event.type == "proactive_check"
        else "主动事件如下，请生成要直接说给用户听的提醒："
    )
    return instruction + "\n" + json.dumps(
        _redact_event_for_model(event),
        ensure_ascii=False,
        indent=2,
    )


def _redact_event_for_model(event: AgentEvent) -> dict[str, Any]:
    payload = dict(event.payload)
    screen_context = payload.get("screen_context")
    if isinstance(screen_context, dict):
        payload["screen_context"] = _redact_screen_context_for_model(screen_context)
    screen_contexts = payload.get("screen_contexts")
    if isinstance(screen_contexts, list):
        payload["screen_contexts"] = [
            _redact_screen_context_for_model(screen_context)
            if isinstance(screen_context, dict)
            else screen_context
            for screen_context in screen_contexts
        ]
    return {
        "type": event.type,
        "payload": payload,
    }


def _redact_screen_context_for_model(screen_context: dict[str, Any]) -> dict[str, Any]:
    redacted_context = dict(screen_context)
    if redacted_context.pop("data_url", None):
        redacted_context["image_attached"] = True
    return redacted_context


def _build_proactive_vision_unsupported_reply() -> ChatReply:
    return parse_chat_reply(
        json.dumps(
            {
                "segments": [
                    {
                        "ja": "今のモデルでは画面までは見られないみたい。勝手に想像しないで、少しだけ休憩の合図にしておくね。",
                        "zh": "当前模型似乎还不能看屏幕。我不会乱猜，就先轻轻提醒你休息一下。",
                        "tone": "提醒",
                        "portrait": "伸手命令",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _build_proactive_tool_loop_rules() -> str:
    return "\n".join(
        [
            "- 这是主动检查事件，不是用户直接发来的请求；整体保持低打扰。",
            "- 请用角色语气自然搭话、提问或提醒用户。",
            build_proactive_rules(include_tool_rules=True),
        ]
    )
