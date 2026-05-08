"""Web server for testing voice input pipeline via browser."""

import io
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import subprocess
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from voice_input.config import ASRConfig, AudioConfig
from voice_input.audio.vad import VoiceActivityDetector
from voice_input.asr.whisper import WhisperASR
from voice_input.llm.processor import LLMProcessor, LLMConfig, LLMBackend

app = FastAPI()

# Global model instances
asr: WhisperASR | None = None
vad: VoiceActivityDetector | None = None
llm: LLMProcessor | None = None

# Conversation context (accumulated clean text)
conversation_context: str = ""


@app.on_event("startup")
def load_models():
    global asr, vad, llm
    print("Loading models...")
    asr = WhisperASR(ASRConfig(model_size="small", device="cpu", compute_type="int8", language="zh"))
    asr.load()
    print("  Whisper loaded.")
    vad = VoiceActivityDetector(AudioConfig())
    vad.load()
    print("  VAD loaded.")

    # Auto-detect LLM backend
    llm_config = _detect_llm_backend()
    llm = LLMProcessor(llm_config)
    print(f"  LLM: {llm_config.backend.value} ({llm_config.get_model()})")
    print("All models ready!")


def _detect_llm_backend() -> LLMConfig:
    # Prefer Ollama (free, local) if available
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            return LLMConfig(backend=LLMBackend.OLLAMA)
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return LLMConfig(backend=LLMBackend.OPENAI)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMConfig(backend=LLMBackend.CLAUDE)
    return LLMConfig(backend=LLMBackend.OLLAMA)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    global conversation_context
    audio_bytes = await audio.read()

    # Use ffmpeg to decode any browser audio format to 16kHz mono PCM
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            return {"status": "error", "message": "Failed to decode audio"}
        audio_np = np.frombuffer(result.stdout, dtype=np.float32)
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)

    if len(audio_np) < 1600:
        return {"status": "too_short", "message": "Recording too short"}

    # VAD
    has_speech = vad.is_speech(audio_np)
    if not has_speech:
        return {"status": "no_speech", "message": "No speech detected"}

    # ASR
    asr_result = asr.transcribe(audio_np)
    if not asr_result.text.strip():
        return {"status": "empty", "message": "Speech detected but no text transcribed"}

    # LLM: understand intent and clean text
    try:
        llm_result = await llm.process(asr_result.text, context=conversation_context)
        clean_text = llm_result.text
        intent = llm_result.intent

        # Update context
        if intent == "input":
            conversation_context += clean_text
        elif intent == "correction":
            conversation_context = clean_text
    except Exception as e:
        # Fallback: return raw text if LLM fails
        clean_text = asr_result.text
        intent = "input"
        conversation_context += clean_text
        print(f"LLM error (using raw text): {e}")

    return {
        "status": "ok",
        "raw_text": asr_result.text,
        "clean_text": clean_text,
        "intent": intent,
        "language": asr_result.language,
        "context": conversation_context,
    }


@app.post("/clear")
async def clear_context():
    global conversation_context
    conversation_context = ""
    return {"status": "ok"}


@app.get("/")
async def index():
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
