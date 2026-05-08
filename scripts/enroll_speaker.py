"""Enroll a speaker by recording voice samples."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_input.config import SpeakerConfig, AudioConfig
from voice_input.speaker import SpeakerEnroller


def record_sample(duration: float, sample_rate: int) -> np.ndarray:
    print(f"  Recording for {duration}s... speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Done.")
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker for voice identification")
    parser.add_argument("name", help="Name for this speaker profile")
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of voice samples to record (default: 3)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration of each sample in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--profiles-dir", default="data/speaker_profiles",
        help="Directory to store speaker profiles",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    audio_config = AudioConfig()
    speaker_config = SpeakerConfig(profiles_dir=Path(args.profiles_dir))

    enroller = SpeakerEnroller(speaker_config)
    print(f"Loading speaker embedding model...")
    enroller.load()

    print(f"\nEnrolling speaker: {args.name}")
    print(f"Will record {args.samples} samples of {args.duration}s each.\n")

    samples = []
    for i in range(args.samples):
        input(f"Sample {i+1}/{args.samples} - Press Enter when ready...")
        audio = record_sample(args.duration, audio_config.sample_rate)
        samples.append(audio)
        if i < args.samples - 1:
            time.sleep(0.5)

    print(f"\nProcessing {len(samples)} samples...")
    profile_path = enroller.enroll(args.name, samples, audio_config.sample_rate)
    print(f"Speaker '{args.name}' enrolled successfully!")
    print(f"Profile saved to: {profile_path}")


if __name__ == "__main__":
    main()
