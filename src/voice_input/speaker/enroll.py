from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Model, Inference

from voice_input.config import SpeakerConfig


class SpeakerEnroller:
    """Enrolls speakers by computing and storing voice embeddings."""

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._inference: Inference | None = None
        self.config.profiles_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        model = Model.from_pretrained(
            self.config.embedding_model,
            use_auth_token=True,
        )
        self._inference = Inference(model, window="whole")

    def enroll(self, name: str, audio_samples: list[np.ndarray], sample_rate: int) -> Path:
        """Enroll a speaker from multiple audio samples. Returns path to saved profile."""
        if self._inference is None:
            self.load()

        embeddings = []
        for audio in audio_samples:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            input_data = {"waveform": waveform, "sample_rate": sample_rate}
            embedding = self._inference(input_data)
            embeddings.append(embedding)

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        profile_path = self.config.profiles_dir / f"{name}.npy"
        np.save(profile_path, mean_embedding)

        meta_path = self.config.profiles_dir / f"{name}.json"
        meta_path.write_text(json.dumps({
            "name": name,
            "num_samples": len(audio_samples),
            "embedding_dim": mean_embedding.shape[-1],
        }))

        return profile_path

    def list_enrolled(self) -> list[str]:
        """List all enrolled speaker names."""
        return [p.stem for p in self.config.profiles_dir.glob("*.npy")]
