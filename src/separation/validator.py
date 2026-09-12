"""
validator.py
Validador acústico de separación de fuentes vocales (SeparationValidator).
Compara las salidas del separador físico utilizando correlación de forma de onda,
similitud espectral, ratio de energía y similitud de huella vocal (embedding)
para evitar la duplicación de una voz dominante como SPEAKER_01 y SPEAKER_02.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any
from scipy import signal

from src.speakers.models import SeparationValidationResult
from src.speakers.embeddings import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)


class SeparationValidator:
    """
    Validador acústico que determina si dos señales separadas representan dos personas reales
    o si una única voz dominante se filtró en ambos canales de salida.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        correlation_threshold: float = 0.60,
        spectral_similarity_threshold: float = 0.86,
        embedding_similarity_threshold: float = 0.68,
        min_energy_ratio: float = 0.08,
    ):
        self.sample_rate = sample_rate
        self.correlation_threshold = correlation_threshold
        self.spectral_similarity_threshold = spectral_similarity_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.min_energy_ratio = min_energy_ratio
        self.extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)

    def validate(
        self,
        source_0: np.ndarray,
        source_1: np.ndarray,
        original_audio: Optional[np.ndarray] = None,
    ) -> SeparationValidationResult:
        """
        Analiza las dos fuentes y dictamina si la separación es válida o si debe realizarse fallback.
        """
        if source_0 is None or source_1 is None or len(source_0) < 512 or len(source_1) < 512:
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=0.0,
                reason="INSUFFICIENT_AUDIO_LENGTH",
                recommended_action="USE_ORIGINAL_AUDIO",
            )

        # 1. Análisis de energía RMS y ratio de energía
        rms_0 = float(np.sqrt(np.mean(source_0 ** 2)))
        rms_1 = float(np.sqrt(np.mean(source_1 ** 2)))

        if rms_0 < 0.005 and rms_1 < 0.005:
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=0.0,
                reason="SILENCE_OR_TOO_QUIET",
                recommended_action="USE_ORIGINAL_AUDIO",
            )

        max_rms = max(rms_0, rms_1)
        min_rms = min(rms_0, rms_1)
        energy_ratio = min_rms / (max_rms + 1e-7)

        # Si una fuente tiene menos del 8% de energía, es fuga/residuo de la otra
        if energy_ratio < self.min_energy_ratio:
            logger.debug("Separación inválida: ratio de energía %.3f < %.3f (fuga residual)", energy_ratio, self.min_energy_ratio)
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=0.90,
                reason="WEAK_RESIDUAL_LEAKAGE",
                energy_ratio=round(energy_ratio, 3),
                recommended_action="USE_ORIGINAL_AUDIO",
            )

        # Alinear longitudes para comparación directa
        min_len = min(len(source_0), len(source_1))
        s0 = source_0[:min_len].astype(np.float32)
        s1 = source_1[:min_len].astype(np.float32)

        # 2. Correlación de forma de onda (Pearson correlation)
        s0_centered = s0 - np.mean(s0)
        s1_centered = s1 - np.mean(s1)
        denom = (np.linalg.norm(s0_centered) * np.linalg.norm(s1_centered)) + 1e-7
        waveform_corr = float(np.dot(s0_centered, s1_centered) / denom)
        waveform_corr = max(-1.0, min(1.0, waveform_corr))

        # 3. Similitud espectral (magnitud STFT media)
        n_fft = 512
        _, _, Z0 = signal.stft(s0, fs=self.sample_rate, nperseg=n_fft)
        _, _, Z1 = signal.stft(s1, fs=self.sample_rate, nperseg=n_fft)
        mag0 = np.mean(np.abs(Z0), axis=1)
        mag1 = np.mean(np.abs(Z1), axis=1)
        spec_norm0 = np.linalg.norm(mag0) + 1e-7
        spec_norm1 = np.linalg.norm(mag1) + 1e-7
        spectral_sim = float(np.dot(mag0, mag1) / (spec_norm0 * spec_norm1))

        # 4. Similitud de huellas vocales (Speaker Embeddings)
        emb0 = self.extractor.extract(source_0)
        emb1 = self.extractor.extract(source_1)
        emb_sim = self.extractor.similarity(emb0, emb1) if (emb0 is not None and emb1 is not None) else 0.50

        details = {
            "rms_0": round(rms_0, 4),
            "rms_1": round(rms_1, 4),
            "energy_ratio": round(energy_ratio, 3),
            "waveform_correlation": round(waveform_corr, 3),
            "spectral_similarity": round(spectral_sim, 3),
            "embedding_similarity": round(emb_sim, 3),
        }

        # Dictamen: ¿Es la misma voz dominante filtrándose a ambos canales?
        # A) Correlación temporal directa muy alta (> 0.60)
        if waveform_corr >= self.correlation_threshold:
            logger.info("Separación rechazada: alta correlación de forma de onda (r=%.2f). Misma voz dominante.", waveform_corr)
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=round(waveform_corr, 3),
                reason="DUPLICATED_DOMINANT_SPEAKER",
                waveform_correlation=round(waveform_corr, 3),
                spectral_similarity=round(spectral_sim, 3),
                embedding_similarity=round(emb_sim, 3),
                energy_ratio=round(energy_ratio, 3),
                recommended_action="USE_ORIGINAL_AUDIO",
                details=details,
            )

        # B) Similitud espectral extrema (> 0.86) Y embedding muy similar (> 0.78)
        if spectral_sim >= self.spectral_similarity_threshold and emb_sim >= self.embedding_similarity_threshold:
            logger.info("Separación rechazada: espectro (%.2f) y voz (%.2f) casi idénticos. Misma persona.", spectral_sim, emb_sim)
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=round(emb_sim, 3),
                reason="SAME_SPEAKER_EMBEDDING",
                waveform_correlation=round(waveform_corr, 3),
                spectral_similarity=round(spectral_sim, 3),
                embedding_similarity=round(emb_sim, 3),
                energy_ratio=round(energy_ratio, 3),
                recommended_action="USE_ORIGINAL_AUDIO",
                details=details,
            )

        # C) Embedding similarity marcadamente alta (> 0.72) independientemente de la correlación de fase
        if emb_sim >= 0.72:
            logger.info("Separación rechazada: embedding similarity muy alta (%.2f). Misma persona.", emb_sim)
            return SeparationValidationResult(
                is_valid_two_speakers=False,
                confidence=round(emb_sim, 3),
                reason="SAME_SPEAKER_EMBEDDING",
                waveform_correlation=round(waveform_corr, 3),
                spectral_similarity=round(spectral_sim, 3),
                embedding_similarity=round(emb_sim, 3),
                energy_ratio=round(energy_ratio, 3),
                recommended_action="USE_ORIGINAL_AUDIO",
                details=details,
            )

        # Separación válida: dos voces con características acústicas claramente distintas
        logger.debug("Separación validada como 2 voces distintas: corr=%.2f, spec=%.2f, emb=%.2f", waveform_corr, spectral_sim, emb_sim)
        return SeparationValidationResult(
            is_valid_two_speakers=True,
            confidence=round(1.0 - max(waveform_corr, emb_sim * 0.8), 3),
            reason="VALID_TWO_SPEAKERS",
            waveform_correlation=round(waveform_corr, 3),
            spectral_similarity=round(spectral_sim, 3),
            embedding_similarity=round(emb_sim, 3),
            energy_ratio=round(energy_ratio, 3),
            recommended_action="USE_TWO_SPEAKERS",
            details=details,
        )
