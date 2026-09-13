"""
src/diagnostics/benchmark_runner.py
Herramienta de barrido y benchmarking paramétrico para pre-roll, post-roll y ventanas de contexto:
  - PRE-ROLL: 200ms, 300ms, 400ms, 500ms
  - POST-ROLL: 250ms, 400ms, 500ms, 700ms
  - CONTEXT WINDOW: 2s, 4s, 6s
Mide el impacto en WER, palabras omitidas iniciales ("I've", "Well", "And"), finales y latencia.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

from src.audio.ring_buffer import AudioRingBuffer
from src.utils.metrics import compute_wer, compute_cer

logger = logging.getLogger(__name__)


@dataclass
class SweepConfigResult:
    config_id: str
    pre_roll_ms: int
    post_roll_ms: int
    window_seconds: float
    wer: float
    cer: float
    missed_initial_words: int
    missed_short_utterances: int
    avg_latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class AcousticBenchmarkRunner:
    """
    Ejecuta simulaciones de captura con buffer rodante y diferentes parámetros
    de pre-roll, post-roll y tamaño de ventana acústica.
    """

    def __init__(self, asr_engine, sample_rate: int = 16000):
        self.asr_engine = asr_engine
        self.sample_rate = sample_rate

    def run_pre_post_roll_sweep(
        self,
        test_items: List[Dict[str, Any]],
        pre_rolls: List[int] = [200, 300, 400, 500],
        post_rolls: List[int] = [250, 400, 500, 700],
    ) -> List[SweepConfigResult]:
        """
        Evalúa combinaciones seleccionadas de pre-roll y post-roll.
        """
        results = []
        # Seleccionar combinaciones representativas
        combinations = [
            (200, 250),
            (300, 400),
            (400, 500),
            (500, 700),
        ]

        for pre_ms, post_ms in combinations:
            config_id = f"pre_{pre_ms}ms_post_{post_ms}ms"
            res = self._evaluate_roll_config(config_id, pre_ms, post_ms, test_items)
            results.append(res)

        return results

    def run_window_size_sweep(
        self,
        test_items: List[Dict[str, Any]],
        windows_seconds: List[float] = [2.0, 4.0, 6.0],
    ) -> List[SweepConfigResult]:
        """
        Evalúa el impacto del tamaño de ventana acústica en la fidelidad y latencia de Whisper.
        """
        results = []
        for win_sec in windows_seconds:
            config_id = f"window_{win_sec:.1f}s"
            res = self._evaluate_window_config(config_id, win_sec, test_items)
            results.append(res)
        return results

    def _evaluate_roll_config(
        self,
        config_id: str,
        pre_ms: int,
        post_ms: int,
        test_items: List[Dict[str, Any]],
    ) -> SweepConfigResult:
        all_refs = []
        all_hyps = []
        latencies = []
        missed_initial = 0
        missed_short = 0

        for item in test_items:
            audio = item["audio"]
            ref = item.get("reference_text", "")
            all_refs.append(ref)

            # Simular ring buffer con pre-roll y post-roll
            ring = AudioRingBuffer(capacity_seconds=8.0, sample_rate=self.sample_rate)
            # Rellenar con audio previo si existe en el fixture o ambiente
            pre_ambient = item.get("pre_audio", np.zeros(int(self.sample_rate * 0.8), dtype=np.float32))
            ring.write(pre_ambient)
            ring.write(audio)

            padded = ring.get_padded_audio(current_speech=audio, pre_ms=pre_ms, post_ms=post_ms)

            t0 = time.perf_counter()
            stt_res = self.asr_engine.transcribe(padded, language="en", vad_filter=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            hyp = stt_res.get("text", "").strip()
            all_hyps.append(hyp)

            # Evaluar palabras iniciales
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            if ref_words and hyp_words:
                if ref_words[0] != hyp_words[0]:
                    missed_initial += 1
            elif ref_words and not hyp_words:
                missed_initial += 1

            # Respuestas cortas
            if len(ref_words) <= 3 and not hyp:
                missed_short += 1

        wer = compute_wer(" ".join(all_refs), " ".join(all_hyps)) * 100.0
        cer = compute_cer(" ".join(all_refs), " ".join(all_hyps)) * 100.0

        return SweepConfigResult(
            config_id=config_id,
            pre_roll_ms=pre_ms,
            post_roll_ms=post_ms,
            window_seconds=0.0,
            wer=round(wer, 2),
            cer=round(cer, 2),
            missed_initial_words=missed_initial,
            missed_short_utterances=missed_short,
            avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else 0.0,
        )

    def _evaluate_window_config(
        self,
        config_id: str,
        window_seconds: float,
        test_items: List[Dict[str, Any]],
    ) -> SweepConfigResult:
        all_refs = []
        all_hyps = []
        latencies = []
        missed_initial = 0
        missed_short = 0

        target_samples = int(window_seconds * self.sample_rate)

        for item in test_items:
            audio = item["audio"]
            ref = item.get("reference_text", "")
            all_refs.append(ref)

            # Ajustar audio al tamaño de ventana (rellenar con ceros o truncar si excede)
            if len(audio) < target_samples:
                pad_len = target_samples - len(audio)
                win_audio = np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])
            else:
                win_audio = audio[:target_samples]

            t0 = time.perf_counter()
            stt_res = self.asr_engine.transcribe(win_audio, language="en", vad_filter=False)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            hyp = stt_res.get("text", "").strip()
            all_hyps.append(hyp)

            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            if ref_words and hyp_words:
                if ref_words[0] != hyp_words[0]:
                    missed_initial += 1
            elif ref_words and not hyp_words:
                missed_initial += 1

            if len(ref_words) <= 3 and not hyp:
                missed_short += 1

        wer = compute_wer(" ".join(all_refs), " ".join(all_hyps)) * 100.0
        cer = compute_cer(" ".join(all_refs), " ".join(all_hyps)) * 100.0

        return SweepConfigResult(
            config_id=config_id,
            pre_roll_ms=0,
            post_roll_ms=0,
            window_seconds=window_seconds,
            wer=round(wer, 2),
            cer=round(cer, 2),
            missed_initial_words=missed_initial,
            missed_short_utterances=missed_short,
            avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else 0.0,
        )
