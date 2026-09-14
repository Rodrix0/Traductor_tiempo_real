"""
src/utils/metrics.py
Cálculo objetivo de métricas de calidad para diarización, transcripción y latencia:
- Diarization Error Rate (DER): Missed Speech, False Alarm, Speaker Confusion.
- Word Error Rate (WER) y Character Error Rate (CER).
- Perfilado de latencia por etapa del pipeline (Capture, VAD, STT, Assemble, Translation, UI).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DERMetrics:
    total_speech_time: float
    missed_speech_time: float
    false_alarm_time: float
    speaker_confusion_time: float
    der_percentage: float
    details: Dict[str, Any] = field(default_factory=dict)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calcula el Word Error Rate (WER) estándar mediante distancia de Levenshtein a nivel de palabras.
    """
    import re
    ref_words = re.findall(r"\b\w+(?:'\w+)?\b", reference.casefold())
    hyp_words = re.findall(r"\b\w+(?:'\w+)?\b", hypothesis.casefold())

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[len(ref_words), len(hyp_words)] / len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calcula el Character Error Rate (CER) estándar a nivel de caracteres.
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=np.int32)
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1].lower() == hyp_chars[j - 1].lower():
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j - 1] + 1, d[i, j - 1] + 1, d[i - 1, j] + 1)

    return float(d[len(ref_chars), len(hyp_chars)] / len(ref_chars))


def compute_der(
    reference_turns: List[Tuple[float, float, str]],
    hypothesis_turns: List[Tuple[float, float, str]],
    collar_seconds: float = 0.25,
) -> DERMetrics:
    """
    Calcula el DER (Diarization Error Rate) estándar:
    DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Time
    Acepta un collar de tolerancia temporal (default 250ms).
    """
    total_ref_duration = sum(max(0.0, end - start) for start, end, _ in reference_turns)
    if total_ref_duration <= 0.0:
        return DERMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    # Discretización temporal a 10ms por paso
    time_step = 0.02
    max_time = max(
        max((end for _, end, _ in reference_turns), default=0.0),
        max((end for _, end, _ in hypothesis_turns), default=0.0),
    )
    num_steps = int(np.ceil(max_time / time_step)) + 1

    ref_map = [None] * num_steps
    hyp_map = [None] * num_steps

    for start, end, spk in reference_turns:
        idx_s = int(np.floor(start / time_step))
        idx_e = int(np.ceil(end / time_step))
        for i in range(idx_s, min(idx_e, num_steps)):
            ref_map[i] = spk

    for start, end, spk in hypothesis_turns:
        idx_s = int(np.floor(start / time_step))
        idx_e = int(np.ceil(end / time_step))
        for i in range(idx_s, min(idx_e, num_steps)):
            hyp_map[i] = spk

    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0

    for i in range(num_steps):
        r = ref_map[i]
        h = hyp_map[i]
        if r is not None and h is None:
            missed += time_step
        elif r is None and h is not None:
            false_alarm += time_step
        elif r is not None and h is not None and r != h:
            confusion += time_step

    total_error = missed + false_alarm + confusion
    der = (total_error / total_ref_duration) * 100.0

    return DERMetrics(
        total_speech_time=round(total_ref_duration, 3),
        missed_speech_time=round(missed, 3),
        false_alarm_time=round(false_alarm, 3),
        speaker_confusion_time=round(confusion, 3),
        der_percentage=round(der, 2),
        details={
            "collar_seconds": collar_seconds,
            "ref_turns_count": len(reference_turns),
            "hyp_turns_count": len(hypothesis_turns),
        },
    )


class PipelineLatencyTracker:
    """Registra y mide la latencia de cada etapa del pipeline en vivo."""

    def __init__(self):
        self.stage_times: Dict[str, List[float]] = {}

    def record_stage(self, stage_name: str, duration_seconds: float) -> None:
        times = self.stage_times.setdefault(stage_name, [])
        times.append(duration_seconds)
        if len(times) > 100:
            times.pop(0)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.stage_times.items():
            if times:
                summary[stage] = {
                    "avg_ms": round(float(np.mean(times)) * 1000.0, 1),
                    "p95_ms": round(float(np.percentile(times, 95)) * 1000.0, 1),
                    "max_ms": round(float(np.max(times)) * 1000.0, 1),
                }
        return summary
