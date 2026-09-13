"""
src/speakers/intra_segment_diarizer.py
Diarización intra-segmento para regiones continuas de voz (IntraSegmentDiarizer).
Resuelve el problema estructural de VAD donde múltiples personas hablan en una sola emisión continua
(ej: Persona A "I want to help" -> Persona B "Absolutely" sin silencio grande entre ambos).
Utiliza los word timestamps de Whisper y ventanas acústicas rodantes para desacoplar hablantes
dentro del mismo bloque de audio.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from src.speakers.tracker import SpeakerTracker
from src.speakers.memory import estimate_pitch_f0

logger = logging.getLogger(__name__)


@dataclass
class AttributedSubsegment:
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    audio: np.ndarray
    confidence: float = 1.0
    is_intra_split: bool = False
    words: List[Dict[str, Any]] = field(default_factory=list)


class IntraSegmentDiarizer:
    """
    Inspecciona segmentos continuos de voz y analiza si contienen más de un hablante,
    dividiéndolos en subsegmentos independientes antes del reensamblado y traducción.
    """

    def __init__(
        self,
        speaker_tracker: SpeakerTracker,
        sample_rate: int = 16000,
        min_split_duration_seconds: float = 1.0,
        min_word_pause_split_seconds: float = 0.15,
    ):
        self.speaker_tracker = speaker_tracker
        self.sample_rate = sample_rate
        self.min_split_duration = min_split_duration_seconds
        self.min_word_pause_split = min_word_pause_split_seconds

    def diarize_segment(
        self,
        audio: np.ndarray,
        base_start_time: float,
        base_end_time: float,
        words: List[Dict[str, Any]],
        fallback_text: str = "",
    ) -> List[AttributedSubsegment]:
        """
        Diariza el segmento continuo.
        Si detecta cambio de voz en una pausa entre palabras o salto tímbrico,
        retorna múltiples AttributedSubsegment con su respectivo speaker_id.
        """
        if audio is None or len(audio) == 0:
            return []

        dur = len(audio) / self.sample_rate

        # 1. Si no hay timestamps de palabras o el audio es muy breve (< 1.0s),
        # asignar con el SpeakerTracker estándar
        if not words or dur < self.min_split_duration or len(words) < 2:
            spk_id = self.speaker_tracker.register_solo_speech(audio, duration=dur)
            return [
                AttributedSubsegment(
                    text=fallback_text or " ".join(w.get("word", "") for w in words).strip(),
                    speaker_id=spk_id,
                    start_time=base_start_time,
                    end_time=base_end_time,
                    audio=audio,
                    confidence=1.0,
                    is_intra_split=False,
                    words=words,
                )
            ]

        # 2. Buscar puntos candidatos de división entre palabras
        # Criterio A: Mini-pausa entre palabras (>= 150 ms)
        split_idx = -1
        max_pause = 0.0

        for i in range(len(words) - 1):
            w_curr = words[i]
            w_next = words[i + 1]
            end_curr = float(w_curr.get("end", 0.0))
            start_next = float(w_next.get("start", 0.0))
            pause = start_next - end_curr

            if pause >= self.min_word_pause_split and pause > max_pause:
                # Asegurar que ambos lados tengan suficiente audio (>= 0.35s)
                if end_curr >= 0.35 and (dur - start_next) >= 0.35:
                    max_pause = pause
                    split_idx = i

        # Si se encontró una pausa inter-palabras significativa, evaluar si hay cambio de voz
        if split_idx >= 0:
            split_time = float(words[split_idx].get("end", 0.0))
            sample_split = int(split_time * self.sample_rate)

            part1_audio = audio[:sample_split]
            part2_audio = audio[sample_split:]

            if len(part1_audio) >= int(0.35 * self.sample_rate) and len(part2_audio) >= int(0.35 * self.sample_rate):
                emb1 = self.speaker_tracker.extractor.extract(part1_audio)
                emb2 = self.speaker_tracker.extractor.extract(part2_audio)
                pitch1 = estimate_pitch_f0(part1_audio, self.sample_rate)
                pitch2 = estimate_pitch_f0(part2_audio, self.sample_rate)

                is_different_speaker = False
                if emb1 is not None and emb2 is not None:
                    sim = self.speaker_tracker.extractor.similarity(emb1, emb2)
                    pitch_diff = abs(pitch1 - pitch2) / max(1.0, pitch1) if pitch1 > 0 and pitch2 > 0 else 0.0

                    if sim < 0.65 or (sim < 0.85 and pitch_diff > 0.25) or (pitch_diff > 0.30):
                        is_different_speaker = True
                        logger.info(
                            "IntraSegmentDiarizer: Cambio de hablante detectado dentro del segmento "
                            "(pausa=%.2fs, sim=%.2f, pitch1=%.1f, pitch2=%.1f)",
                            max_pause,
                            sim,
                            pitch1,
                            pitch2,
                        )

                if is_different_speaker:
                    import time
                    spk1 = self.speaker_tracker.register_solo_speech(part1_audio, duration=split_time)
                    spk2 = self.speaker_tracker.register_solo_speech(part2_audio, duration=dur - split_time)
                    if spk2 == spk1:
                        other_profiles = [p for p in self.speaker_tracker.profiles.keys() if p != spk1]
                        if other_profiles:
                            spk2 = other_profiles[0]
                        else:
                            spk2 = self.speaker_tracker._next_speaker_id()
                            self.speaker_tracker._create_profile(spk2, emb2, dur - split_time, time.monotonic(), pitch=pitch2)
                            self.speaker_tracker.last_active_speaker = spk2

                    words1 = words[: split_idx + 1]
                    words2 = words[split_idx + 1 :]

                    text1 = " ".join(w.get("word", "").strip() for w in words1).strip()
                    text2 = " ".join(w.get("word", "").strip() for w in words2).strip()

                    return [
                        AttributedSubsegment(
                            text=text1,
                            speaker_id=spk1,
                            start_time=base_start_time,
                            end_time=base_start_time + split_time,
                            audio=part1_audio,
                            confidence=0.90,
                            is_intra_split=True,
                            words=words1,
                        ),
                        AttributedSubsegment(
                            text=text2,
                            speaker_id=spk2,
                            start_time=base_start_time + split_time,
                            end_time=base_end_time,
                            audio=part2_audio,
                            confidence=0.90,
                            is_intra_split=True,
                            words=words2,
                        ),
                    ]

        # 3. Si no hubo división válida, asignar como un solo hablante
        spk_id = self.speaker_tracker.register_solo_speech(audio, duration=dur)
        return [
            AttributedSubsegment(
                text=fallback_text or " ".join(w.get("word", "").strip() for w in words).strip(),
                speaker_id=spk_id,
                start_time=base_start_time,
                end_time=base_end_time,
                audio=audio,
                confidence=1.0,
                is_intra_split=False,
                words=words,
            )
        ]
