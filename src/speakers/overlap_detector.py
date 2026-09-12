"""
overlap_detector.py
Módulo de alta eficiencia para la detección de habla simultánea (overlapped speech).
Determina cuándo dos o más personas están hablando al mismo tiempo sobre la misma pista de audio,
evitando activar la separación de fuentes por música, ruido o artefactos aislados.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from scipy.signal import find_peaks
from src.speakers.models import OverlapResult

logger = logging.getLogger(__name__)


class OverlapDetector:
    """
    Detector acústico de habla simultánea optimizado para baja latencia (<5ms).
    Combina análisis espectral armónico de doble tono fundamental (dual-pitch F0),
    relación armónica cruzada y persistencia temporal.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.65,
        min_overlap_duration: float = 0.15,  # 150 ms mínimos de persistencia
        pre_roll_ms: int = 500,
        post_roll_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.min_overlap_duration = min_overlap_duration
        self.pre_roll_ms = pre_roll_ms
        self.post_roll_ms = post_roll_ms

        # Tamaño de frame para resolución espectral F0 (50ms = 800 muestras)
        self.frame_len = int(0.050 * sample_rate)
        self.hop_len = int(0.025 * sample_rate)    # Salto de 25ms

    def detect(self, audio: np.ndarray, base_start_time: float = 0.0) -> OverlapResult:
        """
        Analiza un segmento de audio y determina si contiene habla solapada de dos personas.
        Retorna OverlapResult con confianza, tiempos relativos/absolutos y estimación de hablantes.
        """
        if audio is None or len(audio) < self.frame_len:
            return OverlapResult(
                has_overlap=False,
                speaker_count=1,
                confidence=0.0,
                start_time=base_start_time,
                end_time=base_start_time + (len(audio) / self.sample_rate if audio is not None else 0.0),
                details={"reason": "audio_too_short"},
            )

        # Asegurar formato float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        num_frames = max(1, (len(audio) - self.frame_len) // self.hop_len + 1)
        frame_confidences: List[float] = []
        window = np.hamming(self.frame_len)

        for i in range(num_frames):
            start = i * self.hop_len
            frame = audio[start:start + self.frame_len]
            if len(frame) < self.frame_len:
                break

            # Umbral de energía RMS para descartar silencio o susurros inaudibles
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < 0.015:
                frame_confidences.append(0.0)
                continue

            frame_w = (frame - np.mean(frame)) * window
            conf, _ = self._analyze_frame_pitch(frame_w)
            frame_confidences.append(conf)

        if not frame_confidences:
            return OverlapResult(
                has_overlap=False,
                speaker_count=1,
                confidence=0.0,
                start_time=base_start_time,
                end_time=base_start_time + (len(audio) / self.sample_rate),
            )

        max_conf = max(frame_confidences) if frame_confidences else 0.0
        avg_conf = float(np.mean(frame_confidences))

        # Encontrar frames que superan el umbral
        overlap_frame_indices = [
            idx for idx, c in enumerate(frame_confidences) if c >= (self.confidence_threshold * 0.90)
        ]

        has_overlap = False
        start_t = base_start_time
        end_t = base_start_time + (len(audio) / self.sample_rate)
        confidence = 0.0
        speaker_count = 1

        # Verificar duración contigua mínima para descartar transitorios aislados
        min_required_frames = max(2, int(self.min_overlap_duration / (self.hop_len / self.sample_rate)))
        if len(overlap_frame_indices) >= min_required_frames:
            first_idx = overlap_frame_indices[0]
            last_idx = overlap_frame_indices[-1]
            overlap_duration = (last_idx - first_idx + 1) * (self.hop_len / self.sample_rate)

            if overlap_duration >= self.min_overlap_duration:
                region_confs = [frame_confidences[idx] for idx in overlap_frame_indices]
                confidence = float(np.mean(region_confs))

                if confidence >= self.confidence_threshold:
                    has_overlap = True
                    speaker_count = 2
                    start_t = base_start_time + (first_idx * self.hop_len / self.sample_rate)
                    end_t = base_start_time + ((last_idx * self.hop_len + self.frame_len) / self.sample_rate)

        return OverlapResult(
            has_overlap=has_overlap,
            speaker_count=speaker_count,
            confidence=round(confidence if has_overlap else max_conf, 3),
            start_time=round(start_t, 3),
            end_time=round(end_t, 3),
            details={
                "max_confidence": round(max_conf, 3),
                "avg_confidence": round(avg_conf, 3),
                "overlap_frames_count": len(overlap_frame_indices),
                "total_frames": num_frames,
            },
        )

    def _analyze_frame_pitch(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Analiza el espectro armónico en el rango vocal humano (75Hz a 450Hz).
        Detecta si coexisten dos frecuencias fundamentales independientes que
        no sean múltiplos armónicos entre sí.
        """
        n_fft = 2048
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        max_val = np.max(mag)
        if max_val <= 1e-6:
            return 0.0, False

        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        # Rango vocal fundamental humano (75Hz a 450Hz)
        min_bin = int(n_fft * 75 / self.sample_rate)
        max_bin = int(n_fft * 450 / self.sample_rate)
        sub_mag = mag[min_bin:max_bin]
        sub_freqs = freqs[min_bin:max_bin]

        # Encontrar picos significativos (mínimo 25Hz de distancia entre picos)
        min_distance = int(n_fft * 25 / self.sample_rate)
        peak_indices, _ = find_peaks(sub_mag, height=max_val * 0.20, distance=min_distance)

        if len(peak_indices) < 2:
            return (0.10 if len(peak_indices) == 1 else 0.0), False

        # Extraer frecuencias y magnitudes ordenadas por potencia
        peaks = [(float(sub_freqs[idx]), float(sub_mag[idx])) for idx in peak_indices]
        peaks.sort(key=lambda p: p[1], reverse=True)

        f1, m1 = peaks[0]
        f2, m2 = peaks[1]

        f_low, m_low = (f1, m1) if f1 < f2 else (f2, m2)
        f_high, m_high = (f2, m2) if f1 < f2 else (f1, m1)

        # Comprobar si f_high es múltiplo armónico de f_low (margen de tolerancia 9%)
        ratio = f_high / f_low if f_low > 0 else 0
        nearest_harmonic = round(ratio)
        is_harmonic = False
        if nearest_harmonic >= 2:
            harmonic_diff = abs(ratio - nearest_harmonic)
            if harmonic_diff <= 0.09:
                is_harmonic = True

        if is_harmonic:
            # Es un armónico natural de la misma voz
            return 0.15, False

        # Dos picos fundamentales independientes (VOZ + VOZ)
        peak_ratio = min(m1, m2) / max(m1, m2)
        if peak_ratio >= 0.35:
            conf = min(0.98, float(0.50 + 0.45 * peak_ratio))
            return conf, True

        return float(0.20 + 0.30 * peak_ratio), False

    def get_overlap_region_with_margins(
        self,
        full_audio: np.ndarray,
        result: OverlapResult,
        audio_start_time: float = 0.0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Extrae la región de solapamiento añadiendo márgenes seguros de pre-roll y post-roll (300-700ms)
        para que el separador de fuentes reciba el contexto fonético completo.
        Retorna: (audio_recortado, start_time_absoluto, end_time_absoluto).
        """
        if not result.has_overlap or full_audio is None or len(full_audio) == 0:
            return full_audio, audio_start_time, audio_start_time + (len(full_audio) / self.sample_rate)

        pre_roll_samples = int((self.pre_roll_ms / 1000.0) * self.sample_rate)
        post_roll_samples = int((self.post_roll_ms / 1000.0) * self.sample_rate)

        rel_start_s = max(0.0, result.start_time - audio_start_time)
        rel_end_s = max(rel_start_s, result.end_time - audio_start_time)

        start_idx = max(0, int(rel_start_s * self.sample_rate) - pre_roll_samples)
        end_idx = min(len(full_audio), int(rel_end_s * self.sample_rate) + post_roll_samples)

        cropped = full_audio[start_idx:end_idx]
        abs_start = audio_start_time + (start_idx / self.sample_rate)
        abs_end = audio_start_time + (end_idx / self.sample_rate)

        return cropped, abs_start, abs_end
