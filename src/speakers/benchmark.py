"""
benchmark.py
Herramienta de comparación y benchmark para habla simultánea (Phase 40).
Permite evaluar un mismo audio de dos personas hablando con:
- Modo A: Separación desactivada (pipeline estándar).
- Modo B: Separación activada (detección de overlap + separación física + Whisper dual + SpeakerTracker).
Mide latencia, consumo y calidad comparativa de las transcripciones.
"""

import time
import logging
from typing import Dict, Any, Optional
import numpy as np

from src.speakers.overlap_detector import OverlapDetector
from src.separation.manager import SpeechSeparatorManager
from src.speakers.tracker import SpeakerTracker

logger = logging.getLogger(__name__)


class OverlapBenchmark:
    """Ejecutor de benchmarks comparativos de separación de habla superpuesta."""

    @staticmethod
    def compare_audio(
        audio: np.ndarray,
        asr_engine: Any,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Ejecuta la comparación entre Modo A (Separación OFF) y Modo B (Separación ON).
        """
        results: Dict[str, Any] = {
            "duration_seconds": round(len(audio) / sample_rate, 2),
            "mode_a_no_separation": {},
            "mode_b_with_separation": {},
        }

        # --- MODO A: Sin separación ---
        t0_a = time.perf_counter()
        raw_res = asr_engine.transcribe(audio, language=language)
        t_a_ms = (time.perf_counter() - t0_a) * 1000.0

        results["mode_a_no_separation"] = {
            "text": raw_res.get("text", "").strip(),
            "latency_ms": round(t_a_ms, 1),
        }

        # --- MODO B: Con separación y seguimiento de hablantes ---
        t0_b = time.perf_counter()
        overlap_det = OverlapDetector(sample_rate=sample_rate)
        sep_mgr = SpeechSeparatorManager(mode="auto", sample_rate=sample_rate)
        sep_mgr.load()
        tracker = SpeakerTracker(sample_rate=sample_rate)

        # 1. Detección
        t0_det = time.perf_counter()
        ov_res = overlap_det.detect(audio)
        det_ms = (time.perf_counter() - t0_det) * 1000.0

        # 2. Separación
        t0_sep = time.perf_counter()
        sources, failed = sep_mgr.separate(audio, sample_rate=sample_rate)
        sep_ms = (time.perf_counter() - t0_sep) * 1000.0

        # 3. Asignación de hablantes
        assignments = tracker.assign_sources(sources)

        # 4. Transcripción dual independiente
        speaker_transcripts = {}
        asr_dual_ms = 0.0
        for src, assign in zip(sources, assignments):
            t0_asr = time.perf_counter()
            spk_res = asr_engine.transcribe(src.audio, language=language)
            asr_dual_ms += (time.perf_counter() - t0_asr) * 1000.0
            speaker_transcripts[assign.speaker_id] = spk_res.get("text", "").strip()

        t_b_total_ms = (time.perf_counter() - t0_b) * 1000.0

        results["mode_b_with_separation"] = {
            "overlap_detected": ov_res.has_overlap,
            "overlap_confidence": ov_res.confidence,
            "speaker_transcripts": speaker_transcripts,
            "latency_breakdown": {
                "detection_ms": round(det_ms, 1),
                "separation_ms": round(sep_ms, 1),
                "asr_total_ms": round(asr_dual_ms, 1),
                "total_pipeline_ms": round(t_b_total_ms, 1),
            },
        }

        return results

    @staticmethod
    def format_report(results: Dict[str, Any]) -> str:
        """Formatea los resultados en un informe legible por consola."""
        mode_a = results["mode_a_no_separation"]
        mode_b = results["mode_b_with_separation"]
        bd = mode_b["latency_breakdown"]

        lines = [
            "===============================================================",
            "        BENCHMARK DE HABLA SUPERPUESTA: COMPARATIVA REAL",
            "===============================================================",
            f"Duración de audio: {results['duration_seconds']} segundos",
            "",
            "[ MODO A: SIN SEPARACIÓN (Pipeline Estándar) ]",
            f"Transcripción: \"{mode_a['text']}\"",
            f"Latencia:      {mode_a['latency_ms']} ms",
            "",
            "[ MODO B: CON DETECCIÓN, SEPARACIÓN Y SPEAKER TRACKING ]",
            f"Overlap detectado:  {'SÍ' if mode_b['overlap_detected'] else 'NO'} (Confianza: {mode_b['overlap_confidence']})",
        ]

        for spk_id, text in mode_b["speaker_transcripts"].items():
            lines.append(f"  {spk_id}: \"{text}\"")

        lines.extend([
            "",
            "Desglose de latencia Modo B:",
            f"  - Detección de overlap: {bd['detection_ms']} ms",
            f"  - Separación de audio:  {bd['separation_ms']} ms",
            f"  - ASR dual:             {bd['asr_total_ms']} ms",
            f"  - Total Modo B:         {bd['total_pipeline_ms']} ms",
            "===============================================================",
        ])

        return "\n".join(lines)
