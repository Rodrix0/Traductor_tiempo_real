"""
src/diagnostics/ab_vad_benchmark.py
Prueba A/B controlada para auditar el impacto del Doble VAD:
  Configuración A: External VAD = ON, Whisper vad_filter = False (recomendado para no recortar fonemas)
  Configuración B: External VAD = ON, Whisper vad_filter = True (Doble VAD activo)
Ejecuta exactamente el mismo conjunto de audios y compara objetivamente métricas de fidelidad y corte.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from src.utils.metrics import compute_wer, compute_cer
from src.diagnostics.loss_detector import LossDetector

logger = logging.getLogger(__name__)


@dataclass
class ABVADBenchmarkResult:
    config_name: str
    vad_filter_enabled: bool
    wer: float
    cer: float
    missed_words_count: int
    missed_utterances_count: int
    clipped_beginnings_count: int
    clipped_endings_count: int
    avg_latency_ms: float
    total_audio_seconds: float
    transcripts: List[Dict[str, Any]] = field(default_factory=list)


class ABVADBenchmark:
    """
    Compara de forma aislada y reproducible ambas configuraciones sobre un banco de audio real.
    """

    def __init__(self, asr_engine, loss_detector: Optional[LossDetector] = None):
        self.asr_engine = asr_engine
        self.loss_detector = loss_detector or LossDetector()

    def run_benchmark(
        self,
        audio_items: List[Dict[str, Any]],
        language: str = "en",
    ) -> Dict[str, ABVADBenchmarkResult]:
        """
        Ejecuta Config A (vad_filter=False) y Config B (vad_filter=True) sobre audio_items.
        audio_items: lista de dicts con:
          - 'audio': np.ndarray float32 mono
          - 'reference_text': str
          - 'name': str
        """
        results = {}
        for config_name, vad_filter in [("CONFIG_A_VAD_OFF", False), ("CONFIG_B_VAD_ON", True)]:
            res = self._evaluate_single_config(config_name, vad_filter, audio_items, language)
            results[config_name] = res
        return results

    def _evaluate_single_config(
        self,
        config_name: str,
        vad_filter: bool,
        audio_items: List[Dict[str, Any]],
        language: str,
    ) -> ABVADBenchmarkResult:
        all_refs = []
        all_hyps = []
        latencies = []
        total_seconds = 0.0
        missed_words = 0
        missed_utterances = 0
        clipped_beginnings = 0
        clipped_endings = 0
        transcripts = []

        for item in audio_items:
            audio = item["audio"]
            ref = item.get("reference_text", "").strip()
            total_seconds += len(audio) / 16000.0

            t0 = time.perf_counter()
            stt_res = self.asr_engine.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=vad_filter,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            hyp = stt_res.get("text", "").strip()
            all_refs.append(ref)
            all_hyps.append(hyp)

            # Evaluar palabras faltantes
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            missing = [w for w in ref_words if w not in set(hyp_words)]
            missed_words += len(missing)

            if ref and not hyp:
                missed_utterances += 1

            # Detectar si el comienzo fue cortado (primera palabra de ref no está al inicio de hyp)
            if ref_words and hyp_words and ref_words[0] != hyp_words[0]:
                clipped_beginnings += 1

            # Detectar si el final fue cortado
            if ref_words and hyp_words and ref_words[-1] != hyp_words[-1]:
                clipped_endings += 1

            transcripts.append({
                "name": item.get("name", "audio"),
                "reference": ref,
                "hypothesis": hyp,
                "latency_ms": round(lat_ms, 1),
                "words": stt_res.get("words", []),
            })

        full_ref = " ".join(all_refs)
        full_hyp = " ".join(all_hyps)

        wer = compute_wer(full_ref, full_hyp) * 100.0
        cer = compute_cer(full_ref, full_hyp) * 100.0

        return ABVADBenchmarkResult(
            config_name=config_name,
            vad_filter_enabled=vad_filter,
            wer=round(wer, 2),
            cer=round(cer, 2),
            missed_words_count=missed_words,
            missed_utterances_count=missed_utterances,
            clipped_beginnings_count=clipped_beginnings,
            clipped_endings_count=clipped_endings,
            avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else 0.0,
            total_audio_seconds=round(total_seconds, 2),
            transcripts=transcripts,
        )
