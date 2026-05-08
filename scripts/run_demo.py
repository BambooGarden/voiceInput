"""Demo script: run the voice input pipeline from microphone."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_input.config import PipelineConfig, ASRConfig, SpeakerConfig
from voice_input.pipeline import VoiceInputPipeline, ProcessingResult


def on_result(result: ProcessingResult) -> None:
    prefix = f"[{result.intent.value}]"
    if result.speaker:
        prefix += f" ({result.speaker})"
    print(f"{prefix} {result.clean_text or result.raw_text}")


def main():
    parser = argparse.ArgumentParser(description="Smart Voice Input Demo")
    parser.add_argument(
        "--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--language", default="zh",
        help="Language code for ASR (default: zh)",
    )
    parser.add_argument(
        "--speaker", default=None,
        help="Target speaker name (must be enrolled first)",
    )
    parser.add_argument(
        "--profiles-dir", default="data/speaker_profiles",
        help="Directory for speaker profiles",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PipelineConfig(
        asr=ASRConfig(model_size=args.model, language=args.language),
        speaker=SpeakerConfig(profiles_dir=Path(args.profiles_dir)),
        target_speaker=args.speaker,
    )

    pipeline = VoiceInputPipeline(config)
    print(f"Loading models (whisper-{args.model})... this may take a moment.")
    pipeline.load()
    print("Ready! Speak into your microphone. Press Ctrl+C to stop.\n")
    pipeline.run_streaming(callback=on_result)


if __name__ == "__main__":
    main()
