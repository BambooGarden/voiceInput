from __future__ import annotations

import re


FILLER_WORDS_ZH = [
    "嗯", "啊", "呃", "那个", "就是", "然后", "这个",
    "哦", "额", "唔", "诶",
]

FILLER_WORDS_EN = [
    "um", "uh", "er", "like", "you know", "i mean",
    "basically", "actually", "literally",
]

COMMAND_MAP = {
    "换行": "\n",
    "回车": "\n",
    "句号": "。",
    "逗号": "，",
    "问号": "？",
    "感叹号": "！",
    "冒号": "：",
    "分号": "；",
    "newline": "\n",
    "enter": "\n",
    "period": ".",
    "comma": ",",
    "question mark": "?",
}


class TextCleaner:
    """Cleans and normalizes transcribed text."""

    def __init__(self, remove_fillers: bool = True):
        self.remove_fillers = remove_fillers
        self._filler_pattern_zh = re.compile(
            "|".join(re.escape(w) for w in FILLER_WORDS_ZH)
        )
        self._filler_pattern_en = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS_EN) + r")\b",
            re.IGNORECASE,
        )

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        if not text:
            return text
        text = self._remove_fillers(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def apply_command(self, text: str) -> str | None:
        """Convert command text to its output. Returns None if not a known command."""
        text_lower = text.lower().strip()
        for cmd, output in COMMAND_MAP.items():
            if cmd in text_lower:
                return output
        return None

    def _remove_fillers(self, text: str) -> str:
        if not self.remove_fillers:
            return text
        text = self._filler_pattern_zh.sub("", text)
        text = self._filler_pattern_en.sub("", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text
