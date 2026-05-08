from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pyannote.audio import Model, Inference

from voice_input.config import SpeakerConfig


@dataclass
class IdentificationResult:
    speaker: str | None
    confidence: float
    is_target: bool


class SpeakerIdentifier:
    """Identifies speakers by comparing embeddings against enrolled profiles."""

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._inference: Inference | None = None
        self._profiles: dict[str, np.ndarray] = {}

    def load(self) -> None:
        model = Model.from_pretrained(
            self.config.embedding_model,
            use_auth_token=True,
        )
        self._inference = Inference(model, window="whole")
        self._load_profiles()

    def _load_profiles(self) -> None:
        self._profiles = {}
        for path in self.config.profiles_dir.glob("*.npy"):
            self._profiles[path.stem] = np.load(path)

    def identify(self, audio: np.ndarray, sample_rate: int, target: str | None = None) -> IdentificationResult:
        """Identify speaker from audio segment."""
        if self._inference is None:
            self.load()

        if not self._profiles:
            return IdentificationResult(speaker=None, confidence=0.0, is_target=False)

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}
        embedding = self._inference(input_data)
        embedding = embedding / np.linalg.norm(embedding)

        best_speaker = None
        best_score = -1.0

        for name, profile in self._profiles.items():
            score = float(np.dot(embedding.flatten(), profile.flatten()))
            if score > best_score:
                best_score = score
                best_speaker = name

        is_target = (
            best_speaker == target and best_score >= self.config.similarity_threshold
            if target
            else best_score >= self.config.similarity_threshold
        )

        return IdentificationResult(
            speaker=best_speaker if best_score >= self.config.similarity_threshold else None,
            confidence=best_score,
            is_target=is_target,
        )
