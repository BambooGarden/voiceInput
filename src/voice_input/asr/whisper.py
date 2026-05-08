from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voice_input.config import ASRConfig


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    confidence: float
    language: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment]
    language: str


class WhisperASR:
    """Speech-to-text using faster-whisper (CTranslate2 backend)."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device = self._resolve_device()
        self._model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=self.config.compute_type,
        )

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        import torch
        if torch.backends.mps.is_available():
            return "auto"  # faster-whisper handles MPS via ctranslate2
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio to text."""
        if self._model is None:
            self.load()

        lang = language or self.config.language
        segments_iter, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=self.config.beam_size,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []

        for seg in segments_iter:
            segments.append(TranscriptionSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=seg.avg_logprob,
                language=info.language,
            ))
            full_text_parts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
        )
