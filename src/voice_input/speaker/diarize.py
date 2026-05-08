from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pyannote.audio import Pipeline as PyannotePipeline

from voice_input.config import SpeakerConfig


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


class SpeakerDiarizer:
    """Segments audio into speaker turns using pyannote."""

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._pipeline: PyannotePipeline | None = None

    def load(self) -> None:
        self._pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True,
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._pipeline.to(device)

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        """Segment audio into speaker turns."""
        if self._pipeline is None:
            self.load()

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self._pipeline(input_data)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))
        return segments
