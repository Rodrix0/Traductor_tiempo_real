"""
tracker.py
Seguimiento y coherencia de identidades de hablantes (SpeakerTracker).
Resuelve el problema de permutación de canales, garantiza estabilidad temporal
mediante prior de continuidad con histéresis, confirmación de nuevos hablantes (PENDING_SPEAKER)
y protección estricta contra contaminación de perfiles por audio solapado o ruidoso.
Integrado con SpeakerMemory (persistencia A -> B -> A) y SpeakerTurnDetector (detección de turnos y respuestas cortas).
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.speakers.models import SeparatedSource, SpeakerAssignment, SpeakerProfile
from src.speakers.embeddings import SpeakerEmbeddingExtractor
from src.speakers.memory import SpeakerMemory, estimate_pitch_f0
from src.speakers.turn_detector import SpeakerTurnDetector, TurnTransition

logger = logging.getLogger(__name__)


class SpeakerTracker:
    """
    Rastreador de hablantes en tiempo real para sesiones de traducción.
    Mantiene identidades vocales coherentes a lo largo de toda la sesión sin alternancias espurias
    ni colapso de hablantes en conversaciones cruzadas (A -> B -> A).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        similarity_threshold: float = 0.58,
        continuity_bonus: float = 0.08,
        min_clean_duration_for_profile: float = 0.50,
        min_interjection_duration: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.continuity_bonus = continuity_bonus
        self.min_clean_duration_for_profile = min_clean_duration_for_profile
        self.min_interjection_duration = min_interjection_duration

        self.extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
        self.memory = SpeakerMemory(similarity_threshold=similarity_threshold)
        self.turn_detector = SpeakerTurnDetector()
        self.profiles: Dict[str, SpeakerProfile] = self.memory.profiles

        self.last_active_speaker: Optional[str] = None
        self._speaker_counter = 0
        self._speaker_swaps_corrected = 0

        # Estado para hablantes candidatos (PENDING_SPEAKER) y suavizado temporal
        self._pending_embedding: Optional[np.ndarray] = None
        self._pending_observations: int = 0
        self._pending_first_seen: float = 0.0

        # Suavizado de candidatos y estabilidad
        self.speaker_candidate: Optional[str] = None
        self.speaker_stability_frames: int = 0
        self.speaker_confidence: float = 1.0
        self.last_speech_time: float = 0.0

    def reset(self) -> None:
        """Limpia los perfiles y el estado al finalizar la sesión (privacidad total)."""
        self.memory.reset()
        self.turn_detector.reset()
        self.profiles = self.memory.profiles
        self.last_active_speaker = None
        self._speaker_counter = 0
        self._speaker_swaps_corrected = 0
        self._pending_embedding = None
        self._pending_observations = 0
        self._pending_first_seen = 0.0
        self.speaker_candidate = None
        self.speaker_stability_frames = 0
        self.speaker_confidence = 1.0
        self.last_speech_time = 0.0
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
        Aplica prior de continuidad hacia el último hablante activo para evitar oscilaciones,
        pero permite transiciones nítidas ante pausas o respuestas cortas.
        """
        emb = self.extractor.extract(audio)
        now = time.monotonic()
        dur = duration if duration > 0 else (len(audio) / self.sample_rate if audio is not None else 0.0)
        pitch = estimate_pitch_f0(audio, self.sample_rate)

        # Si el audio es silencio, ruido puro o no se pudo extraer embedding
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
            self._create_profile(spk_01, emb, dur, now, pitch=pitch)
            self.last_active_speaker = spk_01
            self.speaker_stability_frames = 1
            self.turn_detector.evaluate_turn(spk_01, now, now + dur, acoustic_similarity=1.0, timestamp=now)
            logger.info("Hablante inicial registrado: %s (duración: %.2fs, pitch: %.1f Hz)", spk_01, dur, pitch)
            return spk_01

        # Decaimiento dinámico del bono de continuidad tras pausas de silencio
        silence_gap = max(0.0, now - self.last_speech_time) if self.last_speech_time > 0 else 0.0
        self.last_speech_time = now + dur

        continuity_weight = self.turn_detector.get_continuity_weight(silence_gap)
        actual_bonus = self.continuity_bonus * continuity_weight

        # Comparar con todos los perfiles en memoria (centroides + ventana de embeddings + pitch)
        matches = self.memory.match_against_all(emb, self.extractor.similarity, pitch=pitch)

        best_id = None
        best_effective_sim = -1.0
        raw_sims = {}

        for spk_id, metrics in matches.items():
            raw_sim = metrics["acoustic_sim"]
            raw_sims[spk_id] = raw_sim

            bonus = actual_bonus if spk_id == self.last_active_speaker else 0.0
            effective_sim = metrics["combined_sim"] + bonus

            if effective_sim > best_effective_sim:
                best_effective_sim = effective_sim
                best_id = spk_id

        # Si un perfil alternativo tiene mejor similitud acústica directa, permitir que gane
        if len(self.profiles) > 1 and self.last_active_speaker:
            sorted_raw = sorted(raw_sims.items(), key=lambda x: x[1], reverse=True)
            top_spk, top_raw = sorted_raw[0]
            if top_spk != self.last_active_speaker and top_raw >= self.similarity_threshold:
                best_id = top_spk
                best_effective_sim = top_raw

        logger.debug(
            "Comparación de voz: raw=%s, best=%s (eff_sim=%.2f, umbral=%.2f, gap=%.2fs)",
            {k: round(v, 3) for k, v in raw_sims.items()},
            best_id,
            best_effective_sim,
            self.similarity_threshold,
            silence_gap,
        )

        # 1. Coincidencia sólida con perfil existente
        if best_id and best_effective_sim >= self.similarity_threshold:
            if best_id == self.last_active_speaker:
                self.speaker_stability_frames = min(10, self.speaker_stability_frames + 1)
                self.speaker_confidence = min(1.0, 0.7 + 0.05 * self.speaker_stability_frames)
            else:
                self.last_active_speaker = best_id
                self.speaker_candidate = best_id
                self.speaker_stability_frames = 1
                self.speaker_confidence = 0.85

            if is_clean and dur >= self.min_clean_duration_for_profile:
                self._update_profile(best_id, emb, dur, now, is_clean=True, pitch=pitch)
            else:
                self.profiles[best_id].last_seen = now
                self.profiles[best_id].total_duration += dur

            self.turn_detector.evaluate_turn(
                best_id,
                now - dur,
                now,
                acoustic_similarity=best_effective_sim,
                timestamp=now,
            )
            return best_id

        # 2. Si ya hay 2 o más hablantes y uno tiene ventaja clara sin llegar al umbral estricto
        if len(self.profiles) >= 2 and best_id:
            sorted_raw = sorted(raw_sims.items(), key=lambda x: x[1], reverse=True)
            top_spk, top_val = sorted_raw[0]
            second_val = sorted_raw[1][1] if len(sorted_raw) > 1 else 0.0
            if top_val >= 0.48 and (top_val - second_val >= 0.06):
                self.last_active_speaker = top_spk
                self.turn_detector.evaluate_turn(top_spk, now - dur, now, acoustic_similarity=top_val, timestamp=now)
                return top_spk

        # 3. Respuestas muy cortas (interjecciones 0.18s - 0.40s: "yeah", "right", "sure")
        if dur < 0.45:
            # Si se parece moderadamente a algún perfil existente, asignarlo a ese perfil
            if best_id and raw_sims.get(best_id, 0.0) >= 0.48:
                self.last_active_speaker = best_id
                return best_id
            if self.last_active_speaker:
                return self.last_active_speaker
            return list(self.profiles.keys())[0]

        # 4. Evaluación de nuevo hablante (duración suficiente >= 0.45s o candidato repetido)
        if self._pending_embedding is None:
            self._pending_embedding = emb
            self._pending_observations = 1
            self._pending_first_seen = now

            # Si el segmento es suficientemente largo y limpio (>= 0.50s), confirmar nuevo hablante inmediatamente
            if is_clean and dur >= 0.50:
                new_id = self._next_speaker_id()
                self._create_profile(new_id, emb, dur, now, pitch=pitch)
                self.last_active_speaker = new_id
                self._pending_embedding = None
                self._pending_observations = 0
                self.turn_detector.evaluate_turn(new_id, now - dur, now, acoustic_similarity=0.90, timestamp=now)
                logger.info("Nuevo hablante confirmado por segmento limpio (%.2fs): %s", dur, new_id)
                return new_id

            return self.last_active_speaker or list(self.profiles.keys())[0]

        # Comprobar si coincide con la voz candidata pendiente previa
        pending_sim = self.extractor.similarity(emb, self._pending_embedding)
        if pending_sim >= 0.60:
            new_id = self._next_speaker_id()
            self._create_profile(new_id, emb, dur, now, pitch=pitch)
            self.last_active_speaker = new_id
            self._pending_embedding = None
            self._pending_observations = 0
            self.turn_detector.evaluate_turn(new_id, now - dur, now, acoustic_similarity=0.90, timestamp=now)
            logger.info("Nuevo hablante CONFIRMADO tras observaciones repetidas: %s", new_id)
            return new_id
        else:
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
                target_id = self.last_active_speaker or (list(self.profiles.keys())[0] if self.profiles else self._next_speaker_id())
                if target_id not in self.profiles:
                    self._create_profile(target_id, emb0, 1.0, now)
                return [
                    SpeakerAssignment(source_id=sources[0].source_id, speaker_id=target_id, similarity=round(cross_source_sim, 3), confidence=0.95),
                    SpeakerAssignment(source_id=sources[1].source_id, speaker_id=target_id, similarity=round(cross_source_sim, 3), confidence=0.95),
                ]

        active_ids = list(self.profiles.keys())

        # Si aún no hay perfiles registrados, creamos SPEAKER_01 para la primera y SPEAKER_02 para la segunda
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

            if sim_0 >= sim_1:
                spk_2 = self._next_speaker_id()
                if emb1 is not None:
                    self._create_profile(spk_2, emb1, 1.0, now)
                return [
                    SpeakerAssignment(source_id=sources[0].source_id, speaker_id=active_ids[0], similarity=round(sim_0, 3), confidence=0.90),
                    SpeakerAssignment(source_id=sources[1].source_id, speaker_id=spk_2, similarity=round(sim_1, 3), confidence=0.85, is_new_speaker=True),
                ]
            else:
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

    def _create_profile(self, speaker_id: str, emb: np.ndarray, duration: float, now: float, pitch: float = 0.0) -> None:
        """Crea un nuevo perfil a través de SpeakerMemory."""
        self.memory.register_new_speaker(speaker_id, emb, duration, now, pitch=pitch)

    def _update_profile(self, speaker_id: str, new_emb: np.ndarray, duration: float, now: float, is_clean: bool = True, pitch: float = 0.0) -> None:
        """Actualiza el perfil existente a través de SpeakerMemory."""
        self.memory.update_profile(speaker_id, new_emb, duration, now, is_clean=is_clean, pitch=pitch)
