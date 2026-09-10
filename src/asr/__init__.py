"""Módulo de reconocimiento automático de voz (ASR)."""
from .base import ASREngine
from .whisper_engine import WhisperEngine
from .manager import ASRManager, detect_hardware, sanitize_model_for_language

__all__ = [
    "ASREngine",
    "WhisperEngine",
    "ASRManager",
    "detect_hardware",
    "sanitize_model_for_language",
]
