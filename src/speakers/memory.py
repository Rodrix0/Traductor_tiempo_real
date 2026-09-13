"""
src/speakers/memory.py
Memoria de identidades de hablantes a largo plazo (SpeakerMemory).
Preserva perfiles vocales, centroides y multi-ventanas de embeddings a lo largo de toda
la sesión, garantizando que transiciones A -> B -> A mantengan siempre las identidades originales.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.speakers.models import SpeakerProfile

logger = logging.getLogger(__name__)


def estimate_pitch_f0(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimación ligera y robusta de la frecuencia fundamental F0 (pitch)
    mediante autocorrelación normalizada sobre las ventanas de mayor energía vocal (70 Hz - 400 Hz).
    """
    if audio is None or len(audio) < int(sample_rate * 0.05):
        return 0.0

    frame_len = int(sample_rate * 0.10)  # 100ms
    if len(audio) < frame_len:
        return 0.0

    # Extraer ventanas candidatas con su energía RMS
    step = frame_len // 2
    candidates = []
    for i in range(0, len(audio) - frame_len + 1, step):
        chunk = audio[i : i + frame_len]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms >= 0.02:
            candidates.append((rms, i))

    if not candidates:
        return 0.0

    # Ordenar por energía decreciente y evaluar hasta los mejores 5 frames de voz
    candidates.sort(key=lambda x: x[0], reverse=True)
    min_lag = int(sample_rate / 400)  # 400 Hz
    max_lag = int(sample_rate / 70)   # 70 Hz
    pitches = []

    for _, start_idx in candidates[:5]:
        slice_audio = audio[start_idx : start_idx + frame_len]
        slice_audio = slice_audio - np.mean(slice_audio)
        norm = np.sum(slice_audio ** 2)
        if norm < 1e-6:
            continue

        corr = np.correlate(slice_audio, slice_audio, mode='full')
        corr = corr[len(corr) // 2 :]
        if len(corr) <= max_lag:
            continue

        segment = corr[min_lag:max_lag]
        peak_offset = int(np.argmax(segment))
        peak_val = segment[peak_offset]

        if peak_val / norm > 0.25:
            peak_lag = min_lag + peak_offset
            pitches.append(float(sample_rate / peak_lag))

    if pitches:
        return float(np.median(pitches))
    return 0.0


class SpeakerMemory:
    """
    Almacén de perfiles de hablantes con retención persistente a lo largo de la sesión.
    Evita la degradación o reciclado de IDs cuando un hablante deja de hablar temporalmente.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.58,
        max_profiles: int = 8,
        history_window: int = 10,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_profiles = max_profiles
        self.history_window = history_window
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.pitch_history: Dict[str, List[float]] = {}
        self.last_active_speaker: Optional[str] = None

    def reset(self) -> None:
        """Limpia todos los perfiles de la memoria de sesión."""
        self.profiles.clear()
        self.pitch_history.clear()
        self.last_active_speaker = None
        logger.info("SpeakerMemory reiniciada por completo.")

    def has_speakers(self) -> bool:
        return len(self.profiles) > 0

    def speaker_count(self) -> int:
        return len(self.profiles)

    def get_known_speakers(self) -> List[str]:
        return list(self.profiles.keys())

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        return self.profiles.get(speaker_id)

    def register_new_speaker(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
        pitch: float = 0.0,
    ) -> SpeakerProfile:
        """Crea y almacena un nuevo perfil con su vector inicial."""
        prof = SpeakerProfile(
            speaker_id=speaker_id,
            embedding=embedding.copy(),
            sample_count=1,
            total_duration=duration,
            last_seen=timestamp,
            is_confirmed=True,
            recent_embeddings=[embedding.copy()],
        )
        self.profiles[speaker_id] = prof
        if pitch > 0:
            self.pitch_history[speaker_id] = [pitch]
        self.last_active_speaker = speaker_id
        logger.info(
            "SpeakerMemory: Nuevo hablante registrado [%s] (duración=%.2fs, pitch=%.1f Hz)",
            speaker_id,
            duration,
            pitch,
        )
        return prof

    def update_profile(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
        is_clean: bool = True,
        pitch: float = 0.0,
    ) -> None:
        """Actualiza el centroide y la ventana de embeddings recientes de un hablante."""
        prof = self.profiles.get(speaker_id)
        if prof is None:
            return

        prof.last_seen = timestamp
        prof.total_duration += duration

        if is_clean and duration >= 0.5:
            # Actualización ponderada del centroide vocal
            weight = min(0.20, 1.0 / (prof.sample_count + 1))
            new_centroid = (1.0 - weight) * prof.embedding + weight * embedding
            norm = np.linalg.norm(new_centroid)
            if norm > 1e-6:
                prof.embedding = new_centroid / norm
            prof.sample_count += 1

            # Mantener ventana de embeddings recientes
            prof.recent_embeddings.append(embedding.copy())
            if len(prof.recent_embeddings) > self.history_window:
                prof.recent_embeddings.pop(0)

        if pitch > 0:
            p_list = self.pitch_history.setdefault(speaker_id, [])
            p_list.append(pitch)
            if len(p_list) > 20:
                p_list.pop(0)

        self.last_active_speaker = speaker_id

    def get_median_pitch(self, speaker_id: str) -> Optional[float]:
        p_list = self.pitch_history.get(speaker_id)
        if p_list:
            return float(np.median(p_list))
        return None

    def match_against_all(
        self,
        embedding: np.ndarray,
        extractor_sim_fn,
        pitch: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula la similitud contra todos los perfiles existentes usando:
        1. Similitud de coseno contra el centroide principal.
        2. Similitud máxima contra los embeddings recientes (nearest neighbor).
        3. Puntuación combinada robusta.
        """
        results = {}
        for spk_id, prof in self.profiles.items():
            centroid_sim = float(extractor_sim_fn(embedding, prof.embedding_centroid))
            
            # Comparar con embeddings recientes para capturar variaciones acústicas
            recent_sims = [float(extractor_sim_fn(embedding, e)) for e in prof.recent_embeddings]
            max_recent_sim = max(recent_sims) if recent_sims else centroid_sim
            
            # Puntuación acústica compuesta
            best_acoustic = 0.70 * centroid_sim + 0.30 * max_recent_sim

            # Ponderación acústica por pitch fundamental (F0)
            pitch_bonus = 0.0
            if pitch > 0:
                med_pitch = self.get_median_pitch(spk_id)
                if med_pitch and med_pitch > 0:
                    diff_pct = abs(pitch - med_pitch) / med_pitch
                    if diff_pct < 0.12:
                        pitch_bonus = 0.05
                    elif diff_pct > 0.35:
                        pitch_bonus = -0.28
                    elif diff_pct > 0.22:
                        pitch_bonus = -0.15

            combined_sim = max(0.0, min(1.0, best_acoustic + pitch_bonus))

            results[spk_id] = {
                "centroid_sim": centroid_sim,
                "recent_sim": max_recent_sim,
                "acoustic_sim": best_acoustic,
                "combined_sim": combined_sim,
            }
        return results
