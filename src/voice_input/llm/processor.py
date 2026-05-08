from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum

import httpx


SYSTEM_PROMPT = """你是一个语音输入助手。用户通过语音输入文字，语音识别后的原始文本会包含口误、自我纠正、犹豫、重复等情况。

你的任务是：
1. 理解用户最终想要输入的文本
2. 去掉所有口误、纠正性话语、犹豫词、重复
3. 只保留用户最终想表达的干净文本

规则：
- 如果用户说了某句话然后纠正了自己（如"不对"、"我的意思是"、"应该是"），只保留纠正后的版本
- 去掉口头禅和填充词（嗯、啊、那个、就是说、basically、like等）
- 如果用户说"删掉"、"撤回"等，理解为要删除前面说的内容
- 如果是标点/格式命令（换行、句号等），输出对应符号
- 保持用户的语言（中文说中文，英文说英文，混合就混合）

输出格式为JSON：
{"text": "最终干净文本", "intent": "input|correction|command"}

- input: 正常输入文本
- correction: 用户纠正了之前的输入（text为纠正后的完整内容）
- command: 标点或格式命令（text为对应符号如\\n、。、，等）

只输出JSON，不要任何其他内容。"""


class LLMBackend(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    backend: LLMBackend = LLMBackend.OLLAMA
    model: str = ""  # auto-select based on backend if empty
    api_key: str = ""  # reads from env if empty
    ollama_base_url: str = "http://localhost:11434"

    def get_model(self) -> str:
        if self.model:
            return self.model
        env_model = os.environ.get("LLM_MODEL", "")
        if env_model:
            return env_model
        match self.backend:
            case LLMBackend.CLAUDE:
                return "claude-3-5-sonnet-20241022"
            case LLMBackend.OPENAI:
                return "gpt-4o-mini"
            case LLMBackend.OLLAMA:
                return "qwen2.5:3b"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        match self.backend:
            case LLMBackend.CLAUDE:
                return os.environ.get("ANTHROPIC_API_KEY", "")
            case LLMBackend.OPENAI:
                return os.environ.get("OPENAI_API_KEY", "")
            case _:
                return ""


@dataclass
class LLMResult:
    text: str
    intent: str  # "input", "correction", "command"
    raw_response: str = ""


class LLMProcessor:
    """Uses LLM to understand speech intent and clean text."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    async def process(self, raw_text: str, context: str = "") -> LLMResult:
        """Process raw transcription through LLM to get clean intended text."""
        user_msg = raw_text
        if context:
            user_msg = f"[之前已输入的文本: {context}]\n[新的语音输入: {raw_text}]"

        match self.config.backend:
            case LLMBackend.CLAUDE:
                return await self._call_claude(user_msg)
            case LLMBackend.OPENAI:
                return await self._call_openai(user_msg)
            case LLMBackend.OLLAMA:
                return await self._call_ollama(user_msg)

    def process_sync(self, raw_text: str, context: str = "") -> LLMResult:
        """Synchronous version of process."""
        import asyncio
        return asyncio.run(self.process(raw_text, context))

    async def _call_claude(self, user_msg: str) -> LLMResult:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=self.config.get_api_key())
        response = await client.messages.create(
            model=self.config.get_model(),
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return self._parse_response(response.content[0].text)

    async def _call_openai(self, user_msg: str) -> LLMResult:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.config.get_api_key())
        response = await client.chat.completions.create(
            model=self.config.get_model(),
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return self._parse_response(response.choices[0].message.content)

    async def _call_ollama(self, user_msg: str) -> LLMResult:
        payload = {
            "model": self.config.get_model(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.config.ollama_base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return self._parse_response(data["message"]["content"])

    def _parse_response(self, raw: str) -> LLMResult:
        raw = raw.strip()
        # Try to extract JSON from response
        try:
            # Handle cases where LLM wraps in ```json
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            data = json.loads(raw)
            return LLMResult(
                text=data.get("text", ""),
                intent=data.get("intent", "input"),
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError):
            return LLMResult(text=raw, intent="input", raw_response=raw)
