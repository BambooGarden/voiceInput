# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart voice input system for macOS that processes speech through a multi-stage pipeline: Audio Capture → VAD (Silero) → Speaker ID (pyannote) → ASR (faster-whisper) → Intent Classification → Text Cleanup → Output. Bilingual Chinese/English, optimized for Apple Silicon.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_pipeline.py::TestIntentClassifier::test_input_intent

# Web demo (FastAPI on :8000)
python scripts/web_server.py

# Terminal streaming demo
python scripts/run_demo.py

# Enroll a speaker
python scripts/enroll_speaker.py --name "name" --samples 5
```

## Architecture

The pipeline is orchestrated by `VoiceInputPipeline` (`src/voice_input/pipeline.py`). Each stage is a separate module with its own class:

- **audio/** — `AudioCapture` (mic streaming), `VoiceActivityDetector` (Silero VAD), `AudioProcessor` (normalize/resample)
- **speaker/** — `SpeakerIdentifier` (cosine similarity against enrolled `.npy` profiles in `data/speaker_profiles/`), `SpeakerDiarizer`, enrollment via `enroll.py`
- **asr/** — `WhisperASR` wraps faster-whisper with auto device detection (MPS/CUDA/CPU)
- **intent/** — `IntentClassifier` does keyword-based classification into `INPUT`, `CORRECTION`, `COMMAND`, `NOISE`
- **cleaner/** — `TextCleaner` strips filler words (regex) and maps voice commands to symbols via `COMMAND_MAP`
- **llm/** — `LLMProcessor` sends raw transcription to an LLM (Ollama/OpenAI/Claude) for deeper intent understanding and text cleanup; returns structured JSON with `text` + `intent`

All ML models lazy-load on first use. The pipeline's `load()` pre-warms VAD, ASR, and speaker models.

## Key Design Decisions

- **Two-tier intent classification**: `IntentClassifier` does fast keyword matching; `LLMProcessor` provides deeper understanding (self-corrections, mid-sentence fixes). The web server uses the LLM path; the pipeline class uses the rule-based path.
- **LLM backend auto-detection**: Ollama (local) → OpenAI → Claude, checked in priority order. Override with `LLM_MODEL` env var.
- **Speaker profiles**: stored as `.npy` files (512-dim pyannote embeddings). Cosine similarity threshold at 0.65.
- **Config is all dataclasses** in `config.py` — `PipelineConfig` composes `AudioConfig`, `SpeakerConfig`, `ASRConfig`, `IntentConfig`.
- **Web server** (`scripts/web_server.py`) uses ffmpeg subprocess to decode browser audio (webm) to raw PCM. It maintains a `conversation_context` string for multi-turn correction.

## Environment

- Requires pyannote HuggingFace auth token for speaker models
- ffmpeg required for web server audio decoding
- For local LLM: `ollama pull qwen2.5:3b`
