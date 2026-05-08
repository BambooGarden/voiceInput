"""Basic tests for the voice input pipeline components."""

import numpy as np
import pytest

from voice_input.config import PipelineConfig, IntentConfig
from voice_input.intent.classifier import IntentClassifier, Intent
from voice_input.cleaner.text_cleaner import TextCleaner
from voice_input.audio.processor import AudioProcessor


class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier(IntentConfig())

    def test_input_intent(self):
        assert self.classifier.classify("今天天气不错") == Intent.INPUT

    def test_correction_intent(self):
        assert self.classifier.classify("不对不对，删掉") == Intent.CORRECTION

    def test_command_intent(self):
        assert self.classifier.classify("换行") == Intent.COMMAND

    def test_noise_non_target(self):
        assert self.classifier.classify("hello", is_target_speaker=False) == Intent.NOISE

    def test_empty_is_noise(self):
        assert self.classifier.classify("") == Intent.NOISE


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_remove_chinese_fillers(self):
        result = self.cleaner.clean("嗯那个我想说的是")
        assert "嗯" not in result
        assert "那个" not in result

    def test_remove_english_fillers(self):
        result = self.cleaner.clean("um so like I think this is good")
        assert "um" not in result
        assert "like" not in result

    def test_command_map(self):
        assert self.cleaner.apply_command("换行") == "\n"
        assert self.cleaner.apply_command("句号") == "。"

    def test_unknown_command(self):
        assert self.cleaner.apply_command("跳舞") is None

    def test_normalize_whitespace(self):
        result = self.cleaner.clean("hello    world")
        assert result == "hello world"


class TestAudioProcessor:
    def setup_method(self):
        from voice_input.config import AudioConfig
        self.processor = AudioProcessor(AudioConfig())

    def test_normalize(self):
        audio = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        result = self.processor.normalize(audio)
        assert np.abs(result).max() == pytest.approx(1.0)

    def test_concatenate(self):
        chunks = [np.ones(100), np.zeros(100)]
        result = self.processor.concatenate(chunks)
        assert len(result) == 200

    def test_concatenate_empty(self):
        result = self.processor.concatenate([])
        assert len(result) == 0
