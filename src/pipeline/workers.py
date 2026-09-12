"""
workers.py
Trabajadores (workers) concurrentes para el pipeline asíncrono desacoplado.
Cada etapa corre en su propio hilo y se comunica mediante colas acotadas (bounded queues),
garantizando orden estricto mediante sequence_id y prevención de saturación de memoria.
"""

import threading
import queue
import time
import logging
from typing import Optional, Callable, Any, TYPE_CHECKING, List, Tuple
import numpy as np

from src.pipeline.models import (
    AudioChunk,
    SpeechSegment,
    TranscriptResult,
    TranslationResult,
    SubtitleItem,
)
from src.asr.base import ASREngine

if TYPE_CHECKING:
    from src.audio.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class VADWorker(threading.Thread):
    """
    Consume bloques de audio crudo (AudioChunk) y emite fragmentos de voz completos (SpeechSegment)
    utilizando VoiceActivityDetector con perfiles de latencia y pre-roll buffer.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        speech_queue: queue.Queue,
        vad: Any,
        stop_event: threading.Event,
        session_id: Optional[str] = None,
        overlap_detector: Optional[Any] = None,
        separation_queue: Optional[queue.Queue] = None,
    ):
        super().__init__(daemon=True, name="VADWorkerThread")
        self.audio_queue = audio_queue
        self.speech_queue = speech_queue
        self.vad = vad
        self.stop_event = stop_event
        self.session_id = session_id
        self.overlap_detector = overlap_detector
        self.separation_queue = separation_queue
        self._seq_counter = 0

    def run(self) -> None:
        logger.info("VADWorker iniciado.")
        while not self.stop_event.is_set():
            try:
                chunk: AudioChunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                segment_audio = self.vad.process_chunk(chunk.data)
                vad_duration = time.monotonic() - t0

                if segment_audio is not None and len(segment_audio) > 0:
                    self._seq_counter += 1
                    seg_duration = len(segment_audio) / 16000.0
                    now = time.monotonic()

                    speech_segment = SpeechSegment(
                        sequence_id=self._seq_counter,
                        session_id=self.session_id,
                        audio=segment_audio,
                        start_time=0.0,
                        end_time=round(seg_duration, 2),
                        captured_at=now - seg_duration,
                        sample_rate=16000,
                        vad_latency=vad_duration,
                        external_vad_processed=True,  # Para suprimir VAD redundante en ASR
                    )

                    # Detección inteligente de habla simultánea (VOZ + VOZ)
                    if self.overlap_detector and self.separation_queue is not None:
                        try:
                            overlap_res = self.overlap_detector.detect(segment_audio, base_start_time=0.0)
                            if overlap_res.has_overlap:
                                speech_segment.is_overlap = True
                                try:
                                    self.separation_queue.put(speech_segment, timeout=1.0)
                                    continue
                                except queue.Full:
                                    logger.warning("Cola de separación llena. Enrutando a ASR normal.")
                        except Exception as oe:
                            logger.warning("Error en detector de overlap: %s", oe)

                    try:
                        self.speech_queue.put(speech_segment, timeout=1.0)
                    except queue.Full:
                        logger.warning("Cola de voz llena. Descartando segmento #%d", self._seq_counter)

            except Exception as e:
                logger.error("Error en VADWorker procesando chunk: %s", e, exc_info=True)
            finally:
                self.audio_queue.task_done()

        logger.info("VADWorker finalizado.")


class SeparationWorker(threading.Thread):
    """
    Consume SpeechSegments detectados con habla solapada (is_overlap=True),
    extrae el contexto con pre-roll/post-roll y ejecuta la separación física en 2 fuentes de audio.
    Despacha cada fuente separada a speech_queue como un SpeechSegment independiente.
    """

    def __init__(
        self,
        separation_queue: queue.Queue,
        speech_queue: queue.Queue,
        separator_manager: Any,
        stop_event: threading.Event,
        speaker_tracker: Optional[Any] = None,
    ):
        super().__init__(daemon=True, name="SeparationWorkerThread")
        self.separation_queue = separation_queue
        self.speech_queue = speech_queue
        self.separator_manager = separator_manager
        self.stop_event = stop_event
        self.speaker_tracker = speaker_tracker
        from src.separation.validator import SeparationValidator
        self.validator = SeparationValidator(sample_rate=16000)

    def run(self) -> None:
        logger.info("SeparationWorker iniciado.")
        while not self.stop_event.is_set():
            try:
                segment: SpeechSegment = self.separation_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Separar físicamente la mezcla en dos señales de audio
                sources, failed = self.separator_manager.separate(
                    segment.audio,
                    sample_rate=segment.sample_rate,
                    start_time=segment.start_time,
                )

                if failed or len(sources) <= 1:
                    # Fallback seguro: enviar segmento original a ASR sin separar
                    segment.is_overlap = False
                    self.speech_queue.put(segment, timeout=1.0)
                    continue

                # Validación acústica: comprobar si las salidas no son la misma voz dominante
                val_res = self.validator.validate(sources[0].audio, sources[1].audio, original_audio=segment.audio)
                if not val_res.is_valid_two_speakers:
                    logger.info("SeparationWorker: Separación rechazada (%s). Fallback a voz única.", val_res.reason)
                    segment.is_overlap = False
                    self.speech_queue.put(segment, timeout=1.0)
                    continue

                # Pre-asignación con SpeakerTracker para resolver permutación
                speaker_map = {}
                if self.speaker_tracker:
                    try:
                        assignments = self.speaker_tracker.assign_sources(sources)
                        for a in assignments:
                            speaker_map[a.source_id] = a.speaker_id
                    except Exception as te:
                        logger.warning("Error en SpeakerTracker durante asignación: %s", te)

                # Despachar cada fuente separada de forma independiente a la cola de ASR
                for idx, src in enumerate(sources):
                    spk_id = speaker_map.get(src.source_id, f"SPEAKER_{idx + 1:02d}")
                    sub_seg = SpeechSegment(
                        sequence_id=segment.sequence_id,
                        session_id=segment.session_id,
                        audio=src.audio,
                        start_time=src.start_time,
                        end_time=src.end_time,
                        captured_at=segment.captured_at,
                        sample_rate=src.sample_rate,
                        vad_latency=segment.vad_latency,
                        external_vad_processed=True,
                        is_overlap=True,
                        source_id=src.source_id,
                        speaker_id=spk_id,
                    )
                    try:
                        self.speech_queue.put(sub_seg, timeout=1.0)
                    except queue.Full:
                        logger.warning("Cola de ASR llena durante separación. Descartando fuente %s", src.source_id)

            except Exception as e:
                logger.error("Error en SeparationWorker para segmento #%d: %s", segment.sequence_id, e, exc_info=True)
                try:
                    segment.is_overlap = False
                    self.speech_queue.put(segment, timeout=1.0)
                except Exception:
                    pass
            finally:
                self.separation_queue.task_done()

        logger.info("SeparationWorker finalizado.")


class ASRWorker(threading.Thread):
    """
    Consume SpeechSegments de la cola y ejecuta la transcripción acústica (ASREngine).
    Aplica protección contra Doble VAD para no recortar palabras ni añadir latencia de re-filtrado.
    """

    def __init__(
        self,
        speech_queue: queue.Queue,
        transcript_queue: queue.Queue,
        engine: ASREngine,
        stop_event: threading.Event,
        source_language: Optional[str] = None,
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
        speaker_tracker: Optional[Any] = None,
    ):
        super().__init__(daemon=True, name="ASRWorkerThread")
        self.speech_queue = speech_queue
        self.transcript_queue = transcript_queue
        self.engine = engine
        self.stop_event = stop_event
        self.source_language = source_language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.speaker_tracker = speaker_tracker

    def run(self) -> None:
        logger.info("ASRWorker iniciado.")
        while not self.stop_event.is_set():
            try:
                segment: SpeechSegment = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                lang = None if self.source_language in (None, "", "auto", "automático") else self.source_language
                transcript: TranscriptResult = self.engine.transcribe_segment(
                    segment,
                    language=lang,
                    beam_size=self.beam_size,
                    initial_prompt=self.initial_prompt,
                )

                if transcript.text:
                    # Determinar o respetar identidad de hablante
                    speaker_id = segment.speaker_id
                    if not speaker_id:
                        if self.speaker_tracker:
                            try:
                                dur = segment.end_time - segment.start_time
                                speaker_id = self.speaker_tracker.register_solo_speech(segment.audio, duration=dur)
                            except Exception:
                                speaker_id = "SPEAKER_01"
                        else:
                            speaker_id = "SPEAKER_01"

                    transcript.speaker_id = speaker_id
                    transcript.source_id = segment.source_id
                    transcript.is_overlap = segment.is_overlap

                    try:
                        self.transcript_queue.put(transcript, timeout=1.0)
                    except queue.Full:
                        logger.warning("Cola de transcripción llena. Descartando transcripción #%d", segment.sequence_id)

            except Exception as e:
                logger.error("Error en ASRWorker transcribiendo segmento #%d: %s", segment.sequence_id, e, exc_info=True)
            finally:
                self.speech_queue.task_done()

        logger.info("ASRWorker finalizado.")


class TranslationWorker(threading.Thread):
    """
    Consume TranscriptResults y ejecuta la traducción neural offline al idioma de destino.
    """

    def __init__(
        self,
        transcript_queue: queue.Queue,
        translation_queue: queue.Queue,
        translator: Optional[Any],
        stop_event: threading.Event,
        target_language: str = "es",
    ):
        super().__init__(daemon=True, name="TranslationWorkerThread")
        self.transcript_queue = transcript_queue
        self.translation_queue = translation_queue
        self.translator = translator
        self.stop_event = stop_event
        self.target_language = target_language
        from src.speakers.duplicate_resolver import DuplicateTranscriptResolver
        self.duplicate_resolver = DuplicateTranscriptResolver()
        self._recent_transcripts: List[Tuple[float, str, str]] = []

    def run(self) -> None:
        logger.info("TranslationWorker iniciado.")
        while not self.stop_event.is_set():
            try:
                transcript: TranscriptResult = self.transcript_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                src_lang = transcript.language or "en"
                tgt_lang = self.target_language or "es"
                orig_text = transcript.text

                # Detección y supresión de transcripciones duplicadas entre canales
                now_t = time.monotonic()
                self._recent_transcripts = [(t, txt, s) for t, txt, s in self._recent_transcripts if now_t - t < 2.5]
                is_dup = False
                for prev_t, prev_txt, prev_spk in self._recent_transcripts:
                    sim = self.duplicate_resolver.calculate_similarity(orig_text, prev_txt)
                    if sim >= 0.75:
                        logger.info("TranslationWorker: Duplicado suprimido ('%s' vs '%s', sim=%.2f)", orig_text[:35], prev_txt[:35], sim)
                        is_dup = True
                        break

                if is_dup:
                    continue

                self._recent_transcripts.append((now_t, orig_text, transcript.speaker_id))

                # Si el idioma origen coincide con el destino, o no hay traductor, conservar original
                if src_lang == tgt_lang or self.translator is None:
                    translated = orig_text
                else:
                    # Limpieza contextual previa si el traductor la ofrece
                    cleaned = orig_text
                    if hasattr(self.translator, "clean_source"):
                        cleaned = self.translator.clean_source(orig_text, src_lang)

                    try:
                        translated = self.translator.translate(cleaned, src_lang, tgt_lang)
                    except Exception as te:
                        logger.warning("Fallo en traducción (%s -> %s): %s. Mostrando original.", src_lang, tgt_lang, te)
                        translated = orig_text

                trans_duration = time.monotonic() - t0

                res = TranslationResult(
                    sequence_id=transcript.sequence_id,
                    session_id=transcript.session_id,
                    source_text=orig_text,
                    translated_text=translated.strip(),
                    source_language=src_lang,
                    target_language=tgt_lang,
                    captured_at=transcript.captured_at,
                    asr_duration=transcript.asr_duration,
                    translation_start_time=t0,
                    translation_duration=trans_duration,
                    start_time=transcript.start_time,
                    end_time=transcript.end_time,
                    speaker_id=transcript.speaker_id,
                    is_overlap=transcript.is_overlap,
                )

                try:
                    self.translation_queue.put(res, timeout=1.0)
                except queue.Full:
                    logger.warning("Cola de traducción llena. Descartando traducción #%d", transcript.sequence_id)

            except Exception as e:
                logger.error("Error en TranslationWorker para segmento #%d: %s", transcript.sequence_id, e, exc_info=True)
            finally:
                self.transcript_queue.task_done()

        logger.info("TranslationWorker finalizado.")


class SubtitleDispatcherWorker(threading.Thread):
    """
    Consume TranslationResults, construye SubtitleItems con métricas finales de latencia
    y despacha a la cola de subtítulos (para UI / Overlay) y a callbacks (ej. almacenamiento).
    """

    def __init__(
        self,
        translation_queue: queue.Queue,
        subtitle_queue: queue.Queue,
        stop_event: threading.Event,
        on_subtitle_callback: Optional[Callable[[SubtitleItem], None]] = None,
    ):
        super().__init__(daemon=True, name="SubtitleDispatcherThread")
        self.translation_queue = translation_queue
        self.subtitle_queue = subtitle_queue
        self.stop_event = stop_event
        self.on_subtitle_callback = on_subtitle_callback

    def run(self) -> None:
        logger.info("SubtitleDispatcherWorker iniciado.")
        while not self.stop_event.is_set():
            try:
                trans: TranslationResult = self.translation_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                now = time.monotonic()
                total_latency = max(0.0, now - trans.captured_at)

                sub = SubtitleItem(
                    sequence_id=trans.sequence_id,
                    original=trans.source_text,
                    translated=trans.translated_text,
                    source_language=trans.source_language,
                    target_language=trans.target_language,
                    start_time=trans.start_time,
                    end_time=trans.end_time,
                    total_latency=total_latency,
                    asr_latency=trans.asr_duration,
                    trans_latency=trans.translation_duration,
                    created_at=now,
                    speaker_id=trans.speaker_id,
                    is_overlap=trans.is_overlap,
                )

                try:
                    self.subtitle_queue.put(sub, timeout=1.0)
                except queue.Full:
                    logger.warning("Cola de subtítulos llena. Descartando subtítulo #%d", trans.sequence_id)

                if self.on_subtitle_callback:
                    try:
                        self.on_subtitle_callback(sub)
                    except Exception as cb_err:
                        logger.warning("Error en callback de subtítulo: %s", cb_err)

            except Exception as e:
                logger.error("Error en SubtitleDispatcherWorker para #%d: %s", trans.sequence_id, e, exc_info=True)
            finally:
                self.translation_queue.task_done()

        logger.info("SubtitleDispatcherWorker finalizado.")
