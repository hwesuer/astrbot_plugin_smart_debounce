"""
智能消息防抖动插件 - SmartDebounce
作用：用户连续发多条短消息时，调用大模型判断是否说完，
      没说完就合并等待，说完了再一起发给 LLM 处理。
      超时后自动发送，避免用户等太久。

参考：astrbot_plugin_debounce（原版 debounce）的设计思路，
     但改用外部 LLM API 替代本地 ONNX 模型做判断。
"""

# --- 标准库导入 ---
import json          # 解析 LLM 返回的 JSON
import asyncio       # 异步任务，用于超时等待
import re            # 正则，从 LLM 回复里提取 JSON
from typing import Dict  # 类型提示

# --- 第三方库 ---
import httpx         # 异步 HTTP 客户端，调 LLM API 用

# --- AstrBot 插件 API ---
from astrbot.api import AstrBotConfig           # 插件的配置对象
from astrbot.api.event import filter, AstrMessageEvent  # 事件过滤器和消息事件
from astrbot.api.provider import ProviderRequest # LLM 请求对象，可以修改 prompt
from astrbot.api.star import Context, Star, register  # 插件注册和基类
from astrbot.api import logger  # AstrBot 的日志系统


# @register 告诉 AstrBot 这是一个插件
# 参数：(插件名, 作者, 描述, 版本号)
@register("astrbot_plugin_smart_debounce", "plugin_author", "智能消息防抖动插件", "1.0.0")
class SmartDebounce(Star):
    """继承 Star 基类，AstrBot 会自动加载"""

    def __init__(self, context: Context, config: AstrBotConfig):
        """
        初始化插件。
        context: AstrBot 的上下文对象，可以调用平台功能
        config:  插件配置，对应 _conf_schema.json 里定义的字段
        """
        super().__init__(context)
        self.config = config

        # buffer: 按 session_id（每个用户/群聊一个 ID）暂存未说完的消息
        self.buffer: Dict[str, list] = {}

        # timeout_tasks: 每个会话的超时后台任务（超过 timeout 后强制发送）
        self.timeout_tasks: Dict[str, asyncio.Task] = {}

        # judge_tasks: 每个会话的判断延迟任务（等 judge_delay 秒后再调 API 判断）
        self.judge_tasks: Dict[str, asyncio.Task] = {}

        # saved_events: 保存每个会话的原始事件对象，超时后伪造事件要用
        self.saved_events: Dict[str, AstrMessageEvent] = {}

        # 异步锁，防止多个协程同时修改 buffer
        self._lock = asyncio.Lock()

        # 记录需要跳过防抖的消息 ID（超时后伪造的消息不能再被自己拦截）
        self.skip_debounce_msg_ids: set = set()

    @filter.on_llm_request(priority=80)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        核心入口。on_llm_request 在消息即将发送给 LLM 时触发。
        priority=80 低于 blockwords 插件的默认优先级（好像是 50 左右），
        所以 blockwords 会先拦截屏蔽词，然后才到我们这里。

        流程：
        1. 立即停止事件，把消息加入 buffer
        2. 取消旧的 judge_delay 任务和 timeout 任务
        3. 启动新的 judge_delay 任务（等待 judge_delay 秒后再判断）
        """
        # 如果插件被禁用了，直接放行所有消息
        if not self.config.get("enabled", True):
            return

        # 提取会话 ID、消息文本、消息 ID
        session_id = event.message_obj.session_id
        message = event.message_str
        msg_id = event.message_obj.message_id

        # 空消息不管
        if not message:
            return

        # 如果是超时后伪造的消息，跳过防抖
        if msg_id in self.skip_debounce_msg_ids:
            self.skip_debounce_msg_ids.remove(msg_id)
            return

        # 加锁，把消息存入 buffer，替换保存的事件
        async with self._lock:
            if session_id not in self.buffer:
                self.buffer[session_id] = []
            self.buffer[session_id].append(message)
            self.saved_events[session_id] = event

        # 立即阻止当前消息继续传递
        event.stop_event()

        # 取消旧的 judge_delay 任务和 timeout 任务（新消息来了，重新计时）
        old_judge = self.judge_tasks.pop(session_id, None)
        if old_judge:
            old_judge.cancel()
        old_timeout = self.timeout_tasks.pop(session_id, None)
        if old_timeout:
            old_timeout.cancel()

        # 启动新的 judge_delay 任务
        judge_delay = float(self.config.get("judge_delay", 1.5))
        task = asyncio.create_task(self._judge_handler(session_id, judge_delay))
        self.judge_tasks[session_id] = task
        logger.debug(f"[SmartDebounce] 消息已缓存，等待 {judge_delay} 秒后判断: session={session_id}")

    async def _judge_handler(self, session_id: str, delay: float):
        """
        判断延迟处理：等待 delay 秒后，调 LLM API 判断当前累积的消息是否说完。
        如果说完，伪造事件发送；如果没说完，启动超时任务继续等待。
        """
        try:
            await asyncio.sleep(delay)

            # 加锁读取 buffer
            async with self._lock:
                messages = list(self.buffer.get(session_id, []))
                event = self.saved_events.get(session_id)

            if not messages or not event:
                return

            # 调 LLM API 判断当前累积的消息是否完整
            is_complete = await self._check_completeness(messages)

            if is_complete:
                # 说完了一从 buffer 取出合并，伪造事件发送
                async with self._lock:
                    merged = " ".join(self.buffer.pop(session_id, []))
                    self.saved_events.pop(session_id, None)
                self.judge_tasks.pop(session_id, None)

                logger.info(f'[SmartDebounce] 判断完成，合并发送: "{merged[:50]}..."')
                await self._forge_and_send(event, merged)
            else:
                # 没说完，启动超时任务
                logger.debug(f"[SmartDebounce] 判断为未说完，启动超时等待: session={session_id}")
                self.judge_tasks.pop(session_id, None)
                timeout = int(self.config.get("timeout_seconds", 20))
                task = asyncio.create_task(self._timeout_handler(session_id))
                self.timeout_tasks[session_id] = task

        except asyncio.CancelledError:
            logger.debug(f"[SmartDebounce] 判断任务被取消: session={session_id}")
        except Exception as e:
            logger.error(f"[SmartDebounce] 判断处理出错: {e}")
            # 出错时保守处理，发出去
            async with self._lock:
                messages = self.buffer.pop(session_id, None)
                event = self.saved_events.pop(session_id, None)
            if messages and event:
                merged = " ".join(messages)
                await self._forge_and_send(event, merged)

    async def _check_completeness(self, messages: list) -> bool:
        """
        调用外部 LLM API 判断用户消息是否说完。
        返回 True = 说完了，可以发送
        返回 False = 还没说完，继续等

        API 配置要求（在 WebUI 里填）：
        - api_base:  例如 https://api.deepseek.com
        - api_key:   DeepSeek 或其他兼容 API 的密钥
        - model:     例如 deepseek-chat
        - prompt_template: 发给 LLM 的提示词，{messages} 会被替换成实际消息
        """
        # 读取配置
        api_base = str(self.config.get("api_base", "")).strip()
        api_key = str(self.config.get("api_key", "")).strip()
        model = str(self.config.get("model", "")).strip()
        prompt_template = str(self.config.get(
            "prompt_template",
            "你是一个判断用户是否把话说完的助手。\n"
            "以下是用户连续发送的消息片段，请判断用户是否已经完整表达了他的意思。\n"
            "考虑以下情况：\n"
            "- 结尾有句号、问号、感叹号等结束标点 -> 可能说完了\n"
            "- 以「然后」「但是」「不过」「还有」等词结尾 -> 可能没说完\n"
            "- 只说了一个词或短语 -> 可能没说完\n"
            "- 明显是一句话的中间部分 -> 没说完\n"
            "- 一句话结构完整、意思清晰 -> 说完了\n\n"
            "仅返回JSON格式，不要包含其他内容：\n"
            '{"is_complete": true} 或 {"is_complete": false}\n\n'
            "用户消息片段：\n{messages}"
        ))

        # 没配 API 就当成完整，直接放行（降级为普通模式）
        if not api_base or not api_key or not model:
            return True

        # 把用户消息拼进去提示词
        user_content = prompt_template.replace("{messages}", " | ".join(messages))

        # 构造 OpenAI 兼容的 API 请求体
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.1,   # 低温度，让模型输出更确定
            "max_tokens": 100,    # 只需要返回一个简短 JSON
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            # 用 httpx 异步请求，超时 10 秒
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{api_base.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                # 提取模型回复的文本
                content = data["choices"][0]["message"]["content"].strip()

                # 用正则从回复里提取 JSON 对象（防止模型多说了废话）
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    content = json_match.group(0)

                # 解析 JSON，取 is_complete 字段
                result = json.loads(content)
                return bool(result.get("is_complete", True))

        except Exception as e:
            # API 调用失败时，保守地返回 True（放行消息，别卡死用户）
            logger.error(f"[SmartDebounce] LLM API error: {e}")
            return True

    async def _forge_and_send(self, event: AstrMessageEvent, merged: str):
        """
        伪造一条消息事件发送合并后的文本。
        这样消息会重新走一遍消息管道，blockwords 等插件也能正常处理。
        """
        try:
            from astrbot.core.message.components import Plain
            from astrbot.core.star.star_tools import StarTools

            # 保留原始消息里的非文本组件（图片、表情等）
            original_message = event.message_obj.message
            new_components = []
            for component in original_message:
                if not isinstance(component, Plain):
                    new_components.append(component)
            new_components.insert(0, Plain(merged))

            # 创建新消息对象
            new_message = await StarTools.create_message(
                type=str(event.message_obj.type.value),
                self_id=event.get_self_id(),
                session_id=event.session_id,
                sender=event.message_obj.sender,
                message=new_components,
                message_str=merged,
                group_id=event.get_group_id() or ""
            )

            # 标记跳过防抖
            self.skip_debounce_msg_ids.add(new_message.message_id)

            # 提交到事件总线
            await StarTools.create_event(
                abm=new_message,
                platform=event.get_platform_name(),
                is_wake=True
            )
            logger.debug(f"[SmartDebounce] 已伪造事件发送: \"{merged[:50]}...\"")
        except Exception as e:
            logger.error(f"[SmartDebounce] 伪造事件失败: {e}")

    async def _timeout_handler(self, session_id: str):
        """
        超时处理：用户说话说到一半停了，超时后自动把累积的消息发出去。
        """
        timeout = int(self.config.get("timeout_seconds", 20))
        await asyncio.sleep(timeout)

        async with self._lock:
            messages = self.buffer.pop(session_id, None)
            self.timeout_tasks.pop(session_id, None)
            event = self.saved_events.pop(session_id, None)

        if not messages or not event:
            return

        merged = " ".join(messages)
        logger.info(f'[SmartDebounce] 超时，发送合并消息: "{merged[:50]}..."')
        await self._forge_and_send(event, merged)

    async def terminate(self):
        """
        插件卸载时自动调用，清理所有资源。
        卸载前先把缓冲区里积压的消息发出去，避免用户说的话丢失。
        """
        # 取消所有判断任务和超时任务
        for task in self.judge_tasks.values():
            task.cancel()
        self.judge_tasks.clear()
        for task in self.timeout_tasks.values():
            task.cancel()
        self.timeout_tasks.clear()

        # 把缓冲区里还没发出的消息发送出去
        for session_id, messages in list(self.buffer.items()):
            if not messages:
                continue
            event = self.saved_events.get(session_id)
            if not event:
                continue
            merged = " ".join(messages)
            logger.info(f'[SmartDebounce] 插件卸载，清空缓冲区发送: "{merged[:50]}..."')
            await self._forge_and_send(event, merged)

        # 清空缓冲区
        self.buffer.clear()
        self.saved_events.clear()
        self.skip_debounce_msg_ids.clear()
        logger.info("[SmartDebounce] 插件已卸载，资源已清理")
