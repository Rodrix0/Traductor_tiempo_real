"""
src/diagnostics/loss_detector.py
Detector y clasificador de clases de pérdida de diálogo:
  A) AUDIO_MISSING: El WAV entregado a Whisper ya perdió parte de la frase (corte de VAD, buffer o pre/post-roll).
  B) STT_MISSING: El WAV contiene energía y fonemas claros, pero Whisper no transcribió el texto o lo truncó.
  C) PIPELINE_MISSING: Whisper sí transcribió el texto, pero se perdió o descartó posteriormente en colas o filtros.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LossClassification:
    loss_type: str  # "NONE", "AUDIO_MISSING", "STT_MISSING", "PIPELINE_MISSING"
    confidence: float
    reason: str
    details: Dict[str, Any]


class LossDetector:
    """
    Analiza la señal acústica, el resultado STT y el texto final del pipeline
    para aislar exactamente en qué etapa se produjo la pérdida de diálogo.
    """

    def __init__(
        self,
        energy_threshold: float = 0.015,
        min_speech_samples: int = 2880,  # ~180 ms a 16kHz
        edge_clipping_threshold: float = 0.06,
    ):
        self.energy_threshold = energy_threshold
        self.min_speech_samples = min_speech_samples
        self.edge_clipping_threshold = edge_clipping_threshold

    def classify(
        self,
        audio: np.ndarray,
        raw_whisper_text: str,
        final_source_text: str,
        ground_truth_text: Optional[str] = None,
        whisper_confidence: float = 1.0,
        no_speech_prob: float = 0.0,
    ) -> LossClassification:
        """
        Determina si hubo pérdida y clasifica su origen exacto.
        """
        raw_clean = (raw_whisper_text or "").strip()
        final_clean = (final_source_text or "").strip()
        gt_clean = (ground_truth_text or "").strip() if ground_truth_text else None

        if audio is None or len(audio) == 0:
            return LossClassification(
                loss_type="AUDIO_MISSING",
                confidence=1.0,
                reason="audio_buffer_empty",
                details={"audio_len": 0},
            )

        # 1. Comprobar si hubo PIPELINE_MISSING:
        # Whisper transcribió texto legítimo pero el texto final entregado está vacío o severamente truncado
        if raw_clean and not final_clean:
            return LossClassification(
                loss_type="PIPELINE_MISSING",
                confidence=1.0,
                reason="whisper_produced_text_but_pipeline_dropped_it",
                details={"raw_whisper": raw_clean, "final_source": final_clean},
            )

        if raw_clean and len(final_clean) < len(raw_clean) * 0.4 and len(raw_clean) > 8:
            return LossClassification(
                loss_type="PIPELINE_MISSING",
                confidence=0.85,
                reason="text_truncated_downstream_in_pipeline",
                details={"raw_whisper": raw_clean, "final_source": final_clean},
            )

        # 2. Análisis acústico de la señal entregada a Whisper
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        # Comprobar recorte en bordes (onset clipping o offset clipping)
        # Si la señal en los primeros 10-20ms ya tiene alta energía sin rampa suave, el VAD cortó el fonema inicial
        first_10ms = audio[: int(16000 * 0.015)] if len(audio) >= int(16000 * 0.015) else audio
        last_10ms = audio[-int(16000 * 0.015):] if len(audio) >= int(16000 * 0.015) else audio

        onset_energy = float(np.sqrt(np.mean(first_10ms ** 2))) if len(first_10ms) > 0 else 0.0
        offset_energy = float(np.sqrt(np.mean(last_10ms ** 2))) if len(last_10ms) > 0 else 0.0

        is_onset_clipped = onset_energy > self.edge_clipping_threshold
        is_offset_clipped = offset_energy > self.edge_clipping_threshold

        # 3. Comparación con Ground Truth (si se suministra)
        if gt_clean:
            gt_words = gt_clean.lower().split()
            raw_words = raw_clean.lower().split()

            # Si el ground truth tiene palabras y Whisper no sacó nada
            if gt_words and not raw_clean:
                if is_onset_clipped or len(audio) < self.min_speech_samples:
                    return LossClassification(
                        loss_type="AUDIO_MISSING",
                        confidence=0.90,
                        reason="ground_truth_has_words_but_audio_was_clipped_or_too_short",
                        details={"gt": gt_clean, "rms": rms, "samples": len(audio)},
                    )
                elif rms > self.energy_threshold:
                    return LossClassification(
                        loss_type="STT_MISSING",
                        confidence=0.90,
                        reason="audio_has_voice_energy_but_whisper_returned_empty",
                        details={"gt": gt_clean, "rms": rms, "no_speech_prob": no_speech_prob},
                    )

            # Comprobar palabras iniciales faltantes respecto a ground truth
            if len(gt_words) > len(raw_words) and len(raw_words) > 0:
                first_gt = gt_words[0]
                first_raw = raw_words[0]
                if first_gt != first_raw and is_onset_clipped:
                    return LossClassification(
                        loss_type="AUDIO_MISSING",
                        confidence=0.85,
                        reason=f"initial_word_missing_due_to_onset_clipping ('{first_gt}' vs '{first_raw}')",
                        details={"first_gt": first_gt, "first_raw": first_raw, "onset_energy": onset_energy},
                    )

        # 4. Evaluación heurística sin ground truth
        if not raw_clean and rms > self.energy_threshold and len(audio) >= self.min_speech_samples:
            if is_onset_clipped:
                return LossClassification(
                    loss_type="AUDIO_MISSING",
                    confidence=0.80,
                    reason="energy_present_but_abrupt_onset_clipped",
                    details={"rms": rms, "onset_energy": onset_energy},
                )
            else:
                return LossClassification(
                    loss_type="STT_MISSING",
                    confidence=0.80,
                    reason="energy_present_but_whisper_transcribed_empty",
                    details={"rms": rms, "no_speech_prob": no_speech_prob},
                )

        return LossClassification(
            loss_type="NONE",
            confidence=1.0,
            reason="dialogue_captured_normally",
            details={"raw_len": len(raw_clean), "rms": rms},
        )
