from __future__ import annotations

from enum import Enum

from voice_input.config import IntentConfig


class Intent(Enum):
    INPUT = "input"
    CORRECTION = "correction"
    COMMAND = "command"
    NOISE = "noise"


class IntentClassifier:
    """Classifies transcribed text into intent categories.

    - INPUT: regular text meant to be typed
    - CORRECTION: user wants to fix/delete previous input
    - COMMAND: punctuation or formatting commands
    - NOISE: irrelevant speech (not from target speaker, background)
    """

    def __init__(self, config: IntentConfig):
        self.config = config

    def classify(self, text: str, is_target_speaker: bool = True) -> Intent:
        if not is_target_speaker:
            return Intent.NOISE

        text_lower = text.lower().strip()

        if not text_lower:
            return Intent.NOISE

        for keyword in self.config.correction_keywords:
            if keyword in text_lower:
                return Intent.CORRECTION

        for keyword in self.config.command_keywords:
            if keyword in text_lower:
                return Intent.COMMAND

        return Intent.INPUT
