from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_s: float = 0.5
    vad_threshold: float = 0.5
    min_speech_duration_s: float = 0.25
    max_silence_duration_s: float = 0.8


@dataclass
class SpeakerConfig:
    embedding_model: str = "pyannote/embedding"
    segmentation_model: str = "pyannote/segmentation-3.0"
    similarity_threshold: float = 0.65
    profiles_dir: Path = field(default_factory=lambda: Path("data/speaker_profiles"))


@dataclass
class ASRConfig:
    model_size: str = "medium"
    device: str = "auto"  # auto, cpu, cuda, mps
    compute_type: str = "float16"
    language: str = "zh"
    beam_size: int = 5


@dataclass
class IntentConfig:
    correction_keywords: list[str] = field(default_factory=lambda: [
        "不对", "错了", "删掉", "撤回", "重来", "取消",
        "no", "wrong", "delete", "undo", "cancel",
    ])
    command_keywords: list[str] = field(default_factory=lambda: [
        "换行", "回车", "句号", "逗号", "问号",
        "newline", "enter", "period", "comma",
    ])


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    target_speaker: str | None = None  # enrolled speaker name to filter for
