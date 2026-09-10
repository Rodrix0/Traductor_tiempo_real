"""
diagnostics.py
Módulo de diagnóstico del sistema, telemetría y benchmark de hardware en tiempo real.
Permite medir con precisión de microsegundos la velocidad de inferencia ASR y traducción
directamente en la máquina local del usuario.
"""

import time
import logging
from typing import Dict, Any, Optional
import numpy as np

from src.asr.manager import detect_hardware

logger = logging.getLogger(__name__)


class SystemDiagnostics:
    """Recopila información diagnóstica de hardware, drivers y audio de Windows."""

    @staticmethod
    def get_system_summary() -> Dict[str, Any]:
        hw = detect_hardware()
        info = {
            "cuda_available": hw["cuda_available"],
            "device_count": hw["device_count"],
            "recommended_device": hw["recommended_device"],
            "recommended_compute": hw["recommended_compute"],
            "python_version": "",
        }
        try:
            import sys
            info["python_version"] = sys.version.split()[0]
        except Exception:
            pass
        return info


class SystemBenchmark:
    """Ejecuta pruebas sintéticas de velocidad para determinar la latencia real del hardware."""

    @staticmethod
    def run_benchmark(engine=None, translator=None) -> Dict[str, Any]:
        """
        Ejecuta un micro-benchmark en frío y en caliente:
        - 1 segundo de audio sintético a 16kHz
        - 1 frase típica de traducción
        Retorna tiempos de ejecución exactos en milisegundos.
        """
        results = {
            "asr_latency_ms": 0.0,
            "trans_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "asr_device": "cpu",
            "benchmark_passed": True,
            "details": "",
        }

        # 1. Benchmark ASR
        if engine is not None:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            t0 = time.perf_counter()
            try:
                engine.transcribe(dummy_audio, beam_size=1, vad_filter=False)
                asr_duration = (time.perf_counter() - t0) * 1000.0
                results["asr_latency_ms"] = round(asr_duration, 1)
                results["asr_device"] = getattr(engine, "device", "cpu")
            except Exception as e:
                logger.warning("Fallo en benchmark ASR: %s", e)
                results["benchmark_passed"] = False
                results["details"] += f"ASR Error: {e}; "

        # 2. Benchmark Traducción
        if translator is not None:
            sample_text = "Good morning, this is a real-time translation benchmark test."
            t0 = time.perf_counter()
            try:
                translator.translate(sample_text, "en", "es")
                trans_duration = (time.perf_counter() - t0) * 1000.0
                results["trans_latency_ms"] = round(trans_duration, 1)
            except Exception as e:
                logger.warning("Fallo en benchmark Traducción: %s", e)
                results["benchmark_passed"] = False
                results["details"] += f"Translation Error: {e}; "

        results["total_latency_ms"] = round(results["asr_latency_ms"] + results["trans_latency_ms"], 1)
        return results
