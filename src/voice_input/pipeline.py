from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from voice_input.config import PipelineConfig
from voice_input.audio import AudioCapture, VoiceActivityDetector, AudioProcessor
from voice_input.speaker import SpeakerDiarizer, SpeakerIdentifier
from voice_input.asr import WhisperASR
from voice_input.intent import IntentClassifier, Intent
from voice_input.cleaner import TextCleaner

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    raw_text: str
    clean_text: str
    intent: Intent
    speaker: str | None
    is_target_speaker: bool
    confidence: float


class VoiceInputPipeline:
    """Orchestrates the full voice input processing pipeline.

    Flow: Audio → VAD → Speaker ID → ASR → Intent → Clean → Output
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._capture = AudioCapture(self.config.audio)
        self._vad = VoiceActivityDetector(self.config.audio)
        self._processor = AudioProcessor(self.config.audio)
        self._diarizer = SpeakerDiarizer(self.config.speaker)
        self._identifier = SpeakerIdentifier(self.config.speaker)
        self._asr = WhisperASR(self.config.asr)
        self._intent_classifier = IntentClassifier(self.config.intent)
        self._cleaner = TextCleaner()
        self._speech_buffer: list[np.ndarray] = []
        self._loaded = False

    def load(self) -> None:
        """Load all ML models. Call once before processing."""
        logger.info("Loading models...")
        self._vad.load()
        self._asr.load()
        self._identifier.load()
        self._loaded = True
        logger.info("All models loaded.")

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> ProcessingResult | None:
        """Process a chunk of audio through the full pipeline.

        Returns None if audio contains no target-speaker speech.
        """
        if not self._loaded:
            self.load()

        audio = self._processor.normalize(audio)
        if sample_rate != self.config.audio.sample_rate:
            audio = self._processor.resample(audio, sample_rate)

        if not self._vad.is_speech(audio):
            return None

        id_result = self._identifier.identify(
            audio, self.config.audio.sample_rate, target=self.config.target_speaker
        )

        if self.config.target_speaker and not id_result.is_target:
            return ProcessingResult(
                raw_text="",
                clean_text="",
                intent=Intent.NOISE,
                speaker=id_result.speaker,
                is_target_speaker=False,
                confidence=id_result.confidence,
            )

        transcription = self._asr.transcribe(audio)
        intent = self._intent_classifier.classify(
            transcription.text, is_target_speaker=id_result.is_target
        )

        if intent == Intent.COMMAND:
            clean_text = self._cleaner.apply_command(transcription.text) or ""
        elif intent == Intent.INPUT:
            clean_text = self._cleaner.clean(transcription.text)
        else:
            clean_text = ""

        return ProcessingResult(
            raw_text=transcription.text,
            clean_text=clean_text,
            intent=intent,
            speaker=id_result.speaker,
            is_target_speaker=id_result.is_target,
            confidence=id_result.confidence,
        )

    def run_streaming(self, callback=None) -> None:
        """Run the pipeline in streaming mode from microphone.

        Args:
            callback: function(ProcessingResult) called for each processed segment
        """
        if not self._loaded:
            self.load()

        self._capture.start()
        logger.info("Listening... (Ctrl+C to stop)")

        try:
            while True:
                chunk = self._capture.get_chunk(timeout=1.0)
                if chunk is None:
                    continue

                if self._vad.is_speech(chunk.data):
                    self._speech_buffer.append(chunk.data)
                else:
                    if self._speech_buffer:
                        audio = self._processor.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()
                        self._vad.reset()

                        result = self.process_audio(audio, self.config.audio.sample_rate)
                        if result and result.intent != Intent.NOISE:
                            if callback:
                                callback(result)
                            else:
                                self._default_output(result)
        except KeyboardInterrupt:
            pass
        finally:
            self._capture.stop()
            logger.info("Stopped.")

    def _default_output(self, result: ProcessingResult) -> None:
        if result.intent == Intent.INPUT:
            print(result.clean_text, end="", flush=True)
        elif result.intent == Intent.CORRECTION:
            print(f"\n[CORRECTION: {result.raw_text}]")
        elif result.intent == Intent.COMMAND:
            print(result.clean_text, end="", flush=True)
