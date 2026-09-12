"""
tracker.py
Seguimiento y coherencia de identidades de hablantes (SpeakerTracker).
Resuelve el problema de permutación de canales, garantiza estabilidad temporal
mediante prior de continuidad con histéresis, confirmación de nuevos hablantes (PENDING_SPEAKER)
y protección estricta contra contaminación de perfiles por audio solapado o ruidoso.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.speakers.models import SeparatedSource, SpeakerAssignment, SpeakerProfile
from src.speakers.embeddings import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)


class SpeakerTracker:
    """
    Rastreador de hablantes en tiempo real para sesiones de traducción.
    Mantiene identidades vocales coherentes a lo largo de toda la sesión sin alternancias espurias.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        similarity_threshold: float = 0.58,
        continuity_bonus: float = 0.08,
        min_clean_duration_for_profile: float = 0.70,
        min_interjection_duration: float = 0.40,
    ):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.continuity_bonus = continuity_bonus
        self.min_clean_duration_for_profile = min_clean_duration_for_profile
        self.min_interjection_duration = min_interjection_duration

        self.extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.last_active_speaker: Optional[str] = None
        self._speaker_counter = 0
        self._speaker_swaps_corrected = 0

        # Estado para hablantes candidatos (PENDING_SPEAKER)
        self._pending_embedding: Optional[np.ndarray] = None
        self._pending_observations: int = 0
        self._pending_first_seen: float = 0.0

    def reset(self) -> None:
        """Limpia los perfiles y el estado al finalizar la sesión (privacidad total)."""
        self.profiles.clear()
        self.last_active_speaker = None
        self._speaker_counter = 0
        self._speaker_swaps_corrected = 0
        self._pending_embedding = None
        self._pending_observations = 0
        self._pending_first_seen = 0.0
        logger.info("SpeakerTracker reiniciado. Memoria de sesión vaciada.")

    @property
    def speaker_swaps_count(self) -> int:
        return self._speaker_swaps_corrected

    def _next_speaker_id(self) -> str:
        self._speaker_counter += 1
        return f"SPEAKER_{self._speaker_counter:02d}"

    def register_solo_speech(
        self,
        audio: np.ndarray,
        duration: float = 0.0,
        is_clean: bool = True,
    ) -> str:
        """
        Registra la voz de una persona hablando sola.
        Aplica prior de continuidad hacia el último hablante activo para evitar oscilaciones.
        """
        emb = self.extractor.extract(audio)
        now = time.monotonic()
        dur = duration if duration > 0 else (len(audio) / self.sample_rate if audio is not None else 0.0)

        # Si el audio es silencio, ruido puro o demasiado corto
        if emb is None:
            if self.last_active_speaker and self.last_active_speaker in self.profiles:
                return self.last_active_speaker
            if self.profiles:
                return list(self.profiles.keys())[0]
            new_id = self._next_speaker_id()
            return new_id

        # Si no hay perfiles registrados aún, inicializar el primer hablante principal
        if not self.profiles:
            spk_01 = self._next_speaker_id()
            self._create_profile(spk_01, emb, dur, now)
            self.last_active_speaker = spk_01
            logger.info("Hablante inicial registrado: %s (duración: %.2fs)", spk_01, dur)
            return spk_01

        # Comparar con los perfiles existentes aplicando prior de continuidad
        best_id = None
        best_effective_sim = -1.0
        raw_sims = {}

        for spk_id, prof in self.profiles.items():
            # Comparar con el centroide del perfil
            raw_sim = self.extractor.similarity(emb, prof.embedding_centroid)
            raw_sims[spk_id] = raw_sim

            # Sesgo de continuidad: bono si este hablante fue el último en hablar
            bonus = self.continuity_bonus if spk_id == self.last_active_speaker else 0.0
            effective_sim = raw_sim + bonus

            if effective_sim > best_effective_sim:
                best_effective_sim = effective_sim
                best_id = spk_id

        logger.debug(
            "Comparación de voz: raw=%s, best=%s (eff_sim=%.2f, umbral=%.2f)",
            {k: round(v, 3) for k, v in raw_sims.items()},
            best_id,
            best_effective_sim,
            self.similarity_threshold,
        )

        # Si supera el umbral de similitud (con continuidad), mantener este hablante
        if best_id and best_effective_sim >= self.similarity_threshold:
            self.last_active_speaker = best_id
            # Solo actualizar el perfil con audio limpio de duración suficiente
            if is_clean and dur >= self.min_clean_duration_for_profile:
                self._update_profile(best_id, emb, dur, now)
            else:
                self.profiles[best_id].last_seen = now
                self.profiles[best_id].total_duration += dur
            return best_id

        # La voz no coincide con ningún perfil existente.
        # Caso especial: Interjección extremadamente corta (< 0.40s: "yeah", "mhm")
        if dur < self.min_interjection_duration:
            logger.debug("Interjección corta (%.2fs): no se crea nuevo perfil.", dur)
            if self.last_active_speaker:
                return self.last_active_speaker
            return list(self.profiles.keys())[0]

        # Evaluación de PENDING_SPEAKER: Se requiere evidencia sólida antes de crear SPEAKER_02
        if self._pending_embedding is None:
            # Primera observación de la nueva voz
            self._pending_embedding = emb
            self._pending_observations = 1
            self._pending_first_seen = now
            logger.info("Voz candidata detectada (observación 1/2). En espera de confirmación...")
            # Si el segmento tiene duración suficiente (>= 0.95s de voz limpia), confirmar inmediatamente
            if is_clean and dur >= 0.95:
                new_id = self._next_speaker_id()
                self._create_profile(new_id, emb, dur, now)
                self.last_active_speaker = new_id
                self._pending_embedding = None
                self._pending_observations = 0
                logger.info("Nuevo hablante confirmado por segmento de voz limpia (%.2fs): %s", dur, new_id)
                return new_id
            # Devolver temporalmente el hablante activo para no inventar uno nuevo ante ruidos breves
            return self.last_active_speaker or list(self.profiles.keys())[0]

        # Comprobar si coincide con la voz candidata pendiente previa
        pending_sim = self.extractor.similarity(emb, self._pending_embedding)
        if pending_sim >= 0.65:
            # Segunda observación consistente: confirmar nuevo hablante!
            new_id = self._next_speaker_id()
            self._create_profile(new_id, emb, dur, now)
            self.last_active_speaker = new_id
            self._pending_embedding = None
            self._pending_observations = 0
            logger.info("Nuevo hablante CONFIRMADO tras observaciones repetidas: %s", new_id)
            return new_id
        else:
            # Muestra inconsistente con la pendiente previa: actualizar pendiente
            self._pending_embedding = emb
            self._pending_observations = 1
            self._pending_first_seen = now
            return self.last_active_speaker or list(self.profiles.keys())[0]

    def assign_sources(
        self,
        sources: List[SeparatedSource],
    ) -> List[SpeakerAssignment]:
        """
        Asigna las fuentes separadas a identidades de hablante coherentes.
        Verifica primero si ambas fuentes pertenecen a la misma persona para no inventar identidades.
        Resuelve la permutación de canales entre chunks consecutivos.
        """
        if not sources:
            return []

        if len(sources) == 1:
            dur = len(sources[0].audio) / self.sample_rate if sources[0].audio is not None else 0.0
            spk_id = self.register_solo_speech(sources[0].audio, duration=dur, is_clean=False)
            return [SpeakerAssignment(source_id=sources[0].source_id, speaker_id=spk_id, similarity=1.0, confidence=1.0)]

        emb0 = self.extractor.extract(sources[0].audio)
        emb1 = self.extractor.extract(sources[1].audio)
        now = time.monotonic()

        # Comprobar si ambas fuentes separadas corresponden acústicamente a la MISMA persona
        if emb0 is not None and emb1 is not None:
            cross_source_sim = self.extractor.similarity(emb0, emb1)
            if cross_source_sim >= 0.80:
                logger.info(
                    "assign_sources: Ambas fuentes tienen huellas casi idénticas (sim=%.2f). Asignando a un solo hablante.",
                    cross_source_sim,
                )
                # Asignar ambas a la misma identidad activa
                target_id = self.last_active_speaker or (list(self.profiles.keys())[0] if self.profiles else self._next_speaker_id())
                if target_id not in self.profiles:
                    self._create_profile(target_id, emb0, 1.0, now)
                return [
                    SpeakerAssignment(source_id=sources[0].source_id, speaker_id=target_id, similarity=round(cross_source_sim, 3), confidence=0.95),
                    SpeakerAssignment(source_id=sources[1].source_id, speaker_id=target_id, similarity=round(cross_source_sim, 3), confidence=0.95),
                ]

        active_ids = list(self.profiles.keys())

        # Si aún no hay perfiles registrados, creamos SPEAKER_01 para la fuente más fuerte y evaluamos la segunda
        if not active_ids:
            spk_1 = self._next_speaker_id()
            if emb0 is not None:
                self._create_profile(spk_1, emb0, 1.0, now)
            spk_2 = self._next_speaker_id()
            if emb1 is not None:
                self._create_profile(spk_2, emb1, 1.0, now)
            self.last_active_speaker = spk_1
            return [
                SpeakerAssignment(source_id=sources[0].source_id, speaker_id=spk_1, similarity=1.0, confidence=1.0, is_new_speaker=True),
                SpeakerAssignment(source_id=sources[1].source_id, speaker_id=spk_2, similarity=1.0, confidence=1.0, is_new_speaker=True),
            ]

        # Si solo tenemos 1 perfil confirmado (ej. SPEAKER_01)
        if len(active_ids) == 1:
            prof_1 = self.profiles[active_ids[0]].embedding_centroid
            sim_0 = self.extractor.similarity(emb0, prof_1) if emb0 is not None else 0.5
            sim_1 = self.extractor.similarity(emb1, prof_1) if emb1 is not None else 0.5

            # La fuente más parecida a SPEAKER_01 es SPEAKER_01
            if sim_0 >= sim_1:
                # source_0 -> SPEAKER_01, source_1 -> candidato a SPEAKER_02
                spk_2 = self._next_speaker_id()
                if emb1 is not None:
                    self._create_profile(spk_2, emb1, 1.0, now)
                return [
                    SpeakerAssignment(source_id=sources[0].source_id, speaker_id=active_ids[0], similarity=round(sim_0, 3), confidence=0.90),
                    SpeakerAssignment(source_id=sources[1].source_id, speaker_id=spk_2, similarity=round(sim_1, 3), confidence=0.85, is_new_speaker=True),
                ]
            else:
                # Permutación en primer bloque: source_1 -> SPEAKER_01, source_0 -> candidato a SPEAKER_02
                spk_2 = self._next_speaker_id()
                if emb0 is not None:
                    self._create_profile(spk_2, emb0, 1.0, now)
                return [
                    SpeakerAssignment(source_id=sources[0].source_id, speaker_id=spk_2, similarity=round(sim_0, 3), confidence=0.85, is_new_speaker=True),
                    SpeakerAssignment(source_id=sources[1].source_id, speaker_id=active_ids[0], similarity=round(sim_1, 3), confidence=0.90),
                ]

        # Si tenemos 2 o más perfiles, resolver permutación evaluando ambas hipótesis
        spk_a = active_ids[0]
        spk_b = active_ids[1]

        prof_a = self.profiles[spk_a].embedding_centroid
        prof_b = self.profiles[spk_b].embedding_centroid

        sim_0_a = self.extractor.similarity(emb0, prof_a) if emb0 is not None else 0.5
        sim_0_b = self.extractor.similarity(emb0, prof_b) if emb0 is not None else 0.5
        sim_1_a = self.extractor.similarity(emb1, prof_a) if emb1 is not None else 0.5
        sim_1_b = self.extractor.similarity(emb1, prof_b) if emb1 is not None else 0.5

        score_direct = sim_0_a + sim_1_b  # source_0 -> A, source_1 -> B
        score_swapped = sim_0_b + sim_1_a # source_0 -> B, source_1 -> A

        if score_swapped > score_direct:
            self._speaker_swaps_corrected += 1
            logger.debug("Permutación corregida: source_0 -> %s, source_1 -> %s", spk_b, spk_a)
            self.last_active_speaker = spk_b
            return [
                SpeakerAssignment(source_id=sources[0].source_id, speaker_id=spk_b, similarity=round(sim_0_b, 3), confidence=0.90),
                SpeakerAssignment(source_id=sources[1].source_id, speaker_id=spk_a, similarity=round(sim_1_a, 3), confidence=0.90),
            ]
        else:
            self.last_active_speaker = spk_a
            return [
                SpeakerAssignment(source_id=sources[0].source_id, speaker_id=spk_a, similarity=round(sim_0_a, 3), confidence=0.90),
                SpeakerAssignment(source_id=sources[1].source_id, speaker_id=spk_b, similarity=round(sim_1_b, 3), confidence=0.90),
            ]

    def _create_profile(self, speaker_id: str, emb: np.ndarray, duration: float, now: float) -> None:
        """Crea un nuevo perfil con historial de embeddings limpios."""
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            embedding=emb.copy(),
            sample_count=1,
            total_duration=duration,
            last_seen=now,
            is_confirmed=True,
            recent_embeddings=[emb.copy()],
        )

    def _update_profile(self, speaker_id: str, new_emb: np.ndarray, duration: float, now: float) -> None:
        """
        Actualiza el perfil únicamente con muestras de voz limpia (sin solapamiento).
        Mantiene una ventana móvil de embeddings limpios para recalcular el centroide.
        """
        if speaker_id not in self.profiles or new_emb is None:
            return

        prof = self.profiles[speaker_id]
        prof.recent_embeddings.append(new_emb.copy())
        if len(prof.recent_embeddings) > 8:
            prof.recent_embeddings.pop(0)

        # Recalcular centroide normalizado
        mean_vec = np.mean(prof.recent_embeddings, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            prof.embedding = (mean_vec / norm).astype(np.float32)

        prof.sample_count += 1
        prof.total_duration += duration
        prof.last_seen = now
