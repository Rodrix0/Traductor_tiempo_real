"""
Módulo de pipeline desacoplado asíncrono para captura, VAD, ASR, traducción y subtítulos.
"""

from .models import (
    AudioSourceMode,
    VADProfile,
    AudioChunk,
    SpeechSegment,
    TranscriptResult,
    TranslationResult,
    SubtitleItem,
    PipelineMetrics,
)
from .workers import (
    VADWorker,
    ASRWorker,
    TranslationWorker,
    SubtitleDispatcherWorker,
)
from .controller import PipelineController

__all__ = [
    "AudioSourceMode",
    "VADProfile",
    "AudioChunk",
    "SpeechSegment",
    "TranscriptResult",
    "TranslationResult",
    "SubtitleItem",
    "PipelineMetrics",
    "VADWorker",
    "ASRWorker",
    "TranslationWorker",
    "SubtitleDispatcherWorker",
    "PipelineController",
]
