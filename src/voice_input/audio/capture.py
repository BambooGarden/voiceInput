from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from voice_input.config import AudioConfig


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float


class AudioCapture:
    """Captures audio from microphone in streaming chunks."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._queue: queue.Queue[AudioChunk] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._running = False
        self._start_time = 0.0

    def start(self) -> None:
        self._running = True
        self._start_time = 0.0
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=int(self.config.sample_rate * self.config.chunk_duration_s),
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 1.0) -> AudioChunk | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if not self._running:
            return
        chunk = AudioChunk(
            data=indata.copy().flatten(),
            sample_rate=self.config.sample_rate,
            timestamp=time_info.inputBufferAdcTime,
        )
        self._queue.put(chunk)
