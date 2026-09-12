"""
separation package
Módulos para separación de fuentes de voz en habla simultánea.
"""

from src.separation.base import SpeechSeparator
from src.separation.spectral_separator import SpectralSpeechSeparator
from src.separation.mossformer import MossFormerSeparator
from src.separation.sepformer import SepFormerSeparator
from src.separation.manager import SpeechSeparatorManager

__all__ = [
    "SpeechSeparator",
    "SpectralSpeechSeparator",
    "MossFormerSeparator",
    "SepFormerSeparator",
    "SpeechSeparatorManager",
]