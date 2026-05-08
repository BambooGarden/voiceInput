from __future__ import annotations

import numpy as np
import torch

from voice_input.config import AudioConfig


class VoiceActivityDetector:
    """Detects speech segments using Silero VAD."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._model = None
        self._utils = None

    def load(self) -> None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._utils = utils

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech by scanning for speech timestamps."""
        if self._model is None:
            self.load()
        timestamps = self.get_speech_timestamps(audio)
        return len(timestamps) > 0

    def is_speech_chunk(self, audio: np.ndarray) -> bool:
        """Check a single 512-sample chunk (32ms at 16kHz) for speech."""
        if self._model is None:
            self.load()
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        confidence = self._model(tensor, self.config.sample_rate).item()
        return confidence > self.config.vad_threshold

    def get_speech_timestamps(self, audio: np.ndarray) -> list[dict]:
        """Get start/end sample indices of speech segments in audio."""
        if self._model is None:
            self.load()
        tensor = torch.from_numpy(audio).float()
        get_ts_fn = self._utils[0]
        return get_ts_fn(tensor, self._model, sampling_rate=self.config.sample_rate)

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()
