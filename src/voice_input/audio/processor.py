from __future__ import annotations

import numpy as np

from voice_input.config import AudioConfig


class AudioProcessor:
    """Preprocesses audio: resample, normalize, and segment."""

    def __init__(self, config: AudioConfig):
        self.config = config

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            return audio / peak
        return audio

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.config.sample_rate:
            return audio
        import torchaudio
        import torch
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, orig_sr, self.config.sample_rate)
        return resampled.squeeze(0).numpy()

    def concatenate(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
