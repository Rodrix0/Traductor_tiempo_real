"""Módulo de captura y detección de audio."""
from .recorder import AudioRecorder
from .vad import VoiceActivityDetector

__all__ = ["AudioRecorder", "VoiceActivityDetector"]
