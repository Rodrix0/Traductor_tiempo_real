"""
speakers package
Módulos para detección de habla superpuesta (overlap), extracción de embeddings,
seguimiento de hablantes (speaker tracking) y buffers de enunciados.
"""

from src.speakers.models import (
    OverlapResult,
    SeparatedSource,
    SpeakerAssignment,
    SpeakerProfile,
    SeparationValidationResult,
)
from src.speakers.overlap_detector import OverlapDetector
from src.speakers.embeddings import SpeakerEmbeddingExtractor, SpeakerEmbeddingProvider, SpectralEmbeddingProvider
from src.speakers.tracker import SpeakerTracker
from src.speakers.utterance_buffer import UtteranceBuffer, MultiSpeakerUtteranceManager
from src.speakers.duplicate_resolver import DuplicateTranscriptResolver

__all__ = [
    "OverlapResult",
    "SeparatedSource",
    "SpeakerAssignment",
    "SpeakerProfile",
    "SeparationValidationResult",
    "OverlapDetector",
    "SpeakerEmbeddingExtractor",
    "SpeakerEmbeddingProvider",
    "SpectralEmbeddingProvider",
    "SpeakerTracker",
    "UtteranceBuffer",
    "MultiSpeakerUtteranceManager",
    "DuplicateTranscriptResolver",
]