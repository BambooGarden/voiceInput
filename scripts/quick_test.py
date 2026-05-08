"""Quick integration test: record 5 seconds from mic, run VAD + Whisper."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import sounddevice as sd

from voice_input.config import AudioConfig, ASRConfig
from voice_input.audio.vad import VoiceActivityDetector
from voice_input.asr.whisper import WhisperASR
from voice_input.intent.classifier import IntentClassifier, Intent
from voice_input.cleaner.text_cleaner import TextCleaner
from voice_input.config import IntentConfig


def main():
    duration = 5.0
    sample_rate = 16000

    print("=" * 50)
    print("Smart Voice Input - Quick Test")
    print("=" * 50)

    # Load models
    print("\n[1/3] Loading Whisper (small) model...")
    asr = WhisperASR(ASRConfig(model_size="small", device="cpu", compute_type="int8", language="zh"))
    asr.load()
    print("      Whisper loaded.")

    print("[2/3] Loading Silero VAD...")
    vad = VoiceActivityDetector(AudioConfig())
    vad.load()
    print("      VAD loaded.")

    intent_classifier = IntentClassifier(IntentConfig())
    cleaner = TextCleaner()
    print("[3/3] Intent classifier and cleaner ready.")

    # Record
    print(f"\nRecording {duration}s from microphone... SPEAK NOW!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print(f"Recorded {len(audio)} samples ({len(audio)/sample_rate:.1f}s)")

    # VAD
    print("\n--- VAD ---")
    has_speech = vad.is_speech(audio)
    print(f"Speech detected: {has_speech}")

    if not has_speech:
        print("No speech detected. Try speaking louder or closer to the mic.")
        return

    # ASR
    print("\n--- ASR (Whisper) ---")
    result = asr.transcribe(audio)
    print(f"Raw text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")

    # Intent
    print("\n--- Intent ---")
    intent = intent_classifier.classify(result.text)
    print(f"Intent: {intent.value}")

    # Clean
    print("\n--- Cleaned Output ---")
    if intent == Intent.INPUT:
        clean = cleaner.clean(result.text)
        print(f"Clean text: {clean}")
    elif intent == Intent.COMMAND:
        cmd = cleaner.apply_command(result.text)
        print(f"Command output: {repr(cmd)}")
    elif intent == Intent.CORRECTION:
        print(f"Correction detected: {result.text}")
    else:
        print("(noise/filtered)")

    print("\n" + "=" * 50)
    print("Pipeline test complete!")


if __name__ == "__main__":
    main()
