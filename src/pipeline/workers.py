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
    ConfirmedUtterance,
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
        self._stream_samples = 0

    def run(self) -> None:
        logger.info("VADWorker iniciado.")
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                chunk: AudioChunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                segment_audio = self.vad.process_chunk(chunk.data)
                self._stream_samples += len(chunk.data)
                vad_duration = time.monotonic() - t0

                if segment_audio is not None and len(segment_audio) > 0:
                    self._seq_counter += 1
                    seg_duration = len(segment_audio) / 16000.0
                    now = time.monotonic()
                    retained_samples = getattr(self.vad, "speech_samples_count", 0)
                    end_time = (self._stream_samples - retained_samples) / 16000.0
                    start_time = max(0.0, end_time - seg_duration)

                    speech_segment = SpeechSegment(
                        sequence_id=self._seq_counter,
                        session_id=self.session_id,
                        audio=segment_audio,
                        start_time=round(start_time, 3),
                        end_time=round(end_time, 3),
                        # captured_at denotes when the end of this audio reached capture;
                        # end-to-subtitle latency must not include spoken duration.
                        captured_at=now,
                        sample_rate=16000,
                        vad_latency=vad_duration,
                        external_vad_processed=True,  # Para suprimir VAD redundante en ASR
                        end_reason=getattr(self.vad, "last_finalize_reason", None) or "silence",
                        trailing_silence_ms=getattr(self.vad, "last_trailing_silence_ms", 0),
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

    def flush_pending(self) -> None:
        """Emit speech retained inside VAD when capture stops."""
        segment_audio = self.vad.flush() if hasattr(self.vad, "flush") else None
        if segment_audio is None or len(segment_audio) == 0:
            return
        self._seq_counter += 1
        duration = len(segment_audio) / 16000.0
        end_time = self._stream_samples / 16000.0
        segment = SpeechSegment(
            sequence_id=self._seq_counter,
            session_id=self.session_id,
            audio=segment_audio,
            start_time=max(0.0, end_time - duration),
            end_time=end_time,
            captured_at=time.monotonic(),
            sample_rate=16000,
            external_vad_processed=True,
            end_reason="shutdown",
            trailing_silence_ms=0,
        )
        self.speech_queue.put(segment)


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
        while not self.stop_event.is_set() or not self.separation_queue.empty():
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
                        end_reason=segment.end_reason,
                        trailing_silence_ms=segment.trailing_silence_ms,
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


class PresegmentedSourceWorker(threading.Thread):
    """Feeds legacy WindowsCapture VAD segments into the typed async pipeline."""

    def __init__(self, source_queue, speech_queue, separation_queue, stop_event,
                 session_id=None, overlap_detector=None):
        super().__init__(daemon=True, name="PresegmentedSourceThread")
        self.source_queue = source_queue
        self.speech_queue = speech_queue
        self.separation_queue = separation_queue
        self.stop_event = stop_event
        self.session_id = session_id
        self.overlap_detector = overlap_detector
        self.sequence = 0

    def run(self):
        logger.info("PresegmentedSourceWorker iniciado.")
        while not self.stop_event.is_set() or not self.source_queue.empty():
            try:
                source = self.source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.sequence += 1
                segment = SpeechSegment(
                    sequence_id=self.sequence,
                    session_id=self.session_id,
                    audio=source.audio,
                    start_time=source.start_time,
                    end_time=source.end_time,
                    captured_at=source.captured_at,
                    sample_rate=16000,
                    external_vad_processed=True,
                    end_reason=getattr(source, "end_reason", "silence"),
                    trailing_silence_ms=getattr(source, "trailing_silence_ms", 0),
                )
                destination = self.speech_queue
                if self.overlap_detector is not None and self.separation_queue is not None:
                    try:
                        detected = self.overlap_detector.detect(segment.audio, segment.start_time)
                        if detected.has_overlap:
                            segment.is_overlap = True
                            destination = self.separation_queue
                    except Exception as exc:
                        logger.warning("Overlap detection failed; using single-speaker path: %s", exc)
                destination.put(segment)
            finally:
                self.source_queue.task_done()
        logger.info("PresegmentedSourceWorker finalizado.")


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
        while not self.stop_event.is_set() or not self.speech_queue.empty():
            try:
                segment: SpeechSegment = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                lang = None if self.source_language in (None, "", "auto", "automático") else self.source_language
                if hasattr(self.engine, "transcribe_segment"):
                    transcript: TranscriptResult = self.engine.transcribe_segment(
                        segment,
                        language=lang,
                        beam_size=self.beam_size,
                        initial_prompt=self.initial_prompt,
                    )
                else:
                    asr_started = time.monotonic()
                    result = self.engine.transcribe(
                        segment.audio,
                        language=lang,
                        beam_size=self.beam_size,
                        initial_prompt=self.initial_prompt,
                        vad_filter=False,
                    )
                    transcript = TranscriptResult(
                        sequence_id=segment.sequence_id,
                        session_id=segment.session_id,
                        text=str(result.get("text", "")).strip(),
                        language=result.get("language") or lang or "en",
                        confidence=float(result.get("confidence", result.get("language_probability", 0.0))),
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        captured_at=segment.captured_at,
                        asr_start_time=asr_started,
                        asr_duration=float(result.get("elapsed_time", 0.0)),
                        audio=segment.audio,
                        word_timestamps=result.get("words", []),
                        avg_logprob=float(result.get("avg_logprob", 0.0)),
                        no_speech_probability=float(result.get("no_speech_probability", 0.0)),
                        is_partial=segment.end_reason not in ("silence", "shutdown"),
                        end_reason=segment.end_reason,
                        trailing_silence_ms=segment.trailing_silence_ms,
                    )

                if transcript.text:
                    # Determinar o respetar identidad de hablante
                    # Normal speech is attributed continuously by IntraSegmentDiarizer
                    # in the assembler stage. Pre-separated overlap keeps its assignment.
                    speaker_id = segment.speaker_id or ("" if self.speaker_tracker else "SPEAKER_01")

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


class UtteranceAssemblerWorker(threading.Thread):
    """Diarizes TranscriptResults, assembles sentences, and emits confirmations."""

    def __init__(
        self,
        transcript_queue: queue.Queue,
        confirmed_queue: queue.Queue,
        engine: ASREngine,
        stop_event: threading.Event,
        source_language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        speaker_tracker: Optional[Any] = None,
        retry_engine: Optional[Any] = None,
        on_debug: Optional[Callable[[str, str, dict], None]] = None,
        acoustic_progress: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
    ):
        super().__init__(daemon=True, name="UtteranceAssemblerThread")
        from src.pipeline.sentence_turn_assembler import SentenceTurnAssembler
        self.transcript_queue = transcript_queue
        self.confirmed_queue = confirmed_queue
        self.stop_event = stop_event
        self.speaker_tracker = speaker_tracker
        self.diarizer = None
        if speaker_tracker is not None:
            from src.speakers.intra_segment_diarizer import IntraSegmentDiarizer
            self.diarizer = IntraSegmentDiarizer(speaker_tracker=speaker_tracker)
        self.assembler = SentenceTurnAssembler(
            asr_engine=engine,
            retry_engine=retry_engine,
            initial_prompt=initial_prompt,
            on_debug=on_debug,
        )
        self.source_language = source_language
        self.acoustic_progress = acoustic_progress
        self._assembler_lock = threading.Lock()

    def _check_timeouts(self) -> None:
        progress = self.acoustic_progress() if self.acoustic_progress else None
        self._emit(self.assembler.check_timeouts(acoustic_progress=progress))

    def _emit(self, utterances: List[ConfirmedUtterance]) -> None:
        for utterance in utterances:
            try:
                self.confirmed_queue.put(utterance, timeout=1.0)
            except queue.Full:
                logger.error("Confirmed sentence queue full; preserving order by waiting for capacity.")
                self.confirmed_queue.put(utterance)

    def _process(self, transcript: TranscriptResult) -> None:
        events = []
        if self.diarizer is not None and not transcript.is_overlap and transcript.audio is not None:
            events = self.diarizer.diarize_segment(
                audio=transcript.audio,
                base_start_time=transcript.start_time,
                base_end_time=transcript.end_time,
                words=transcript.word_timestamps,
                fallback_text=transcript.text,
            )
        if not events:
            from types import SimpleNamespace
            events = [SimpleNamespace(
                text=transcript.text,
                speaker_id=transcript.speaker_id or "SPEAKER_01",
                start_time=transcript.start_time,
                end_time=transcript.end_time,
                audio=transcript.audio if transcript.audio is not None else np.zeros(0, dtype=np.float32),
                confidence=transcript.confidence,
                words=transcript.word_timestamps,
            )]
        for index, event in enumerate(events):
            # Convert the capture timestamp of the VAD block into the timestamp
            # of this attributed subsegment's real end. This includes VAD wait,
            # ASR, assembly and translation in end-to-output latency.
            event_captured_at = transcript.captured_at - max(
                0.0, transcript.end_time - event.end_time
            )
            self._emit(self.assembler.add_event(
                speaker_id=event.speaker_id,
                text=event.text,
                word_timestamps=event.words,
                audio=event.audio,
                audio_start=event.start_time,
                audio_end=event.end_time,
                is_partial=transcript.is_partial and index == len(events) - 1,
                confidence=min(transcript.confidence, getattr(event, "confidence", transcript.confidence)),
                source_language=transcript.language or self.source_language or "en",
                sequence_id=transcript.sequence_id,
                session_id=transcript.session_id,
                captured_at=event_captured_at,
                asr_duration=transcript.asr_duration,
                trailing_silence_ms=transcript.trailing_silence_ms if index == len(events) - 1 else 0,
                is_overlap=transcript.is_overlap,
            ))

    def flush_pending(self) -> None:
        with self._assembler_lock:
            self._emit(self.assembler.flush_all())

    def run(self) -> None:
        logger.info("UtteranceAssemblerWorker iniciado.")
        while not self.stop_event.is_set() or not self.transcript_queue.empty():
            try:
                transcript = self.transcript_queue.get(timeout=0.1)
            except queue.Empty:
                with self._assembler_lock:
                    self._check_timeouts()
                continue
            try:
                with self._assembler_lock:
                    self._process(transcript)
            except Exception as exc:
                logger.error("Error assembling transcript #%s: %s", getattr(transcript, "sequence_id", "?"), exc, exc_info=True)
            finally:
                self.transcript_queue.task_done()
            with self._assembler_lock:
                self._check_timeouts()
        self.flush_pending()
        logger.info("UtteranceAssemblerWorker finalizado.")


class TranslationWorker(threading.Thread):
    """
    Consume únicamente ConfirmedUtterance y ejecuta la traducción neural offline.
    """

    def __init__(
        self,
        transcript_queue: queue.Queue,
        translation_queue: queue.Queue,
        translator: Optional[Any],
        stop_event: threading.Event,
        target_language: str = "es",
        on_debug: Optional[Callable[[str, str, dict], None]] = None,
    ):
        super().__init__(daemon=True, name="TranslationWorkerThread")
        self.confirmed_queue = transcript_queue
        self.translation_queue = translation_queue
        self.translator = translator
        self.stop_event = stop_event
        self.target_language = target_language
        self.on_debug = on_debug
        from src.speakers.duplicate_resolver import DuplicateTranscriptResolver
        self.duplicate_resolver = DuplicateTranscriptResolver()
        self._recent_transcripts: List[Tuple[float, float, str, str]] = []

    def run(self) -> None:
        logger.info("TranslationWorker iniciado.")
        while not self.stop_event.is_set() or not self.confirmed_queue.empty():
            try:
                utterance = self.confirmed_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                if not isinstance(utterance, ConfirmedUtterance):
                    raise TypeError("TranslationWorker only accepts ConfirmedUtterance")
                src_lang = utterance.source_language or "en"
                tgt_lang = self.target_language or "es"
                orig_text = utterance.source_text
                if self.on_debug:
                    self.on_debug("TRANSLATING", utterance.speaker_id, {
                        "text": orig_text,
                        "utterance_id": utterance.utterance_id,
                    })

                # Detección y supresión de transcripciones duplicadas entre canales
                self._recent_transcripts = [
                    (start, end, txt, speaker)
                    for start, end, txt, speaker in self._recent_transcripts
                    if utterance.start_time - end < 2.5
                ]
                is_dup = False
                for prev_start, prev_end, prev_txt, prev_spk in self._recent_transcripts:
                    if prev_spk != utterance.speaker_id:
                        continue
                    temporal_overlap = min(prev_end, utterance.end_time) - max(prev_start, utterance.start_time)
                    same_boundary = abs(prev_start - utterance.start_time) <= 0.15
                    if temporal_overlap <= 0.0 and not same_boundary:
                        continue
                    sim = self.duplicate_resolver.calculate_similarity(orig_text, prev_txt)
                    if sim >= 0.75:
                        logger.info("TranslationWorker: Duplicado suprimido ('%s' vs '%s', sim=%.2f)", orig_text[:35], prev_txt[:35], sim)
                        is_dup = True
                        break

                if is_dup:
                    continue

                self._recent_transcripts.append((
                    utterance.start_time,
                    utterance.end_time,
                    orig_text,
                    utterance.speaker_id,
                ))

                # Preserve interrupted/incomplete content in subtitles, but do not
                # translate it or contaminate the translator's sentence context.
                if not utterance.is_complete or utterance.is_interrupted or src_lang == tgt_lang or self.translator is None:
                    translated = orig_text
                else:
                    # Limpieza contextual previa si el traductor la ofrece
                    cleaned = orig_text
                    if hasattr(self.translator, "clean_source"):
                        cleaned = self.translator.clean_source(orig_text, src_lang)

                    try:
                        import inspect
                        sig = inspect.signature(self.translator.translate)
                        if "speaker_id" in sig.parameters:
                            translated = self.translator.translate(cleaned, src_lang, tgt_lang, speaker_id=utterance.speaker_id)
                        else:
                            translated = self.translator.translate(cleaned, src_lang, tgt_lang)
                    except Exception as te:
                        logger.warning("Fallo en traducción (%s -> %s): %s. Mostrando original.", src_lang, tgt_lang, te)
                        translated = orig_text

                trans_duration = time.monotonic() - t0

                res = TranslationResult(
                    sequence_id=utterance.sequence_id,
                    session_id=utterance.session_id,
                    source_text=orig_text,
                    translated_text=translated.strip(),
                    source_language=src_lang,
                    target_language=tgt_lang,
                    captured_at=utterance.captured_at,
                    asr_duration=utterance.asr_duration,
                    translation_start_time=t0,
                    translation_duration=trans_duration,
                    start_time=utterance.start_time,
                    end_time=utterance.end_time,
                    speaker_id=utterance.speaker_id,
                    is_overlap=utterance.is_overlap,
                    utterance_id=utterance.utterance_id,
                    is_complete=utterance.is_complete,
                    is_interrupted=utterance.is_interrupted,
                    final_asr_pass_used=utterance.final_asr_pass_used,
                    completion_score=utterance.completion_score,
                    completion_reason=utterance.completion_reason,
                    partial_fragments=utterance.partial_fragments,
                    word_timestamps=utterance.word_timestamps,
                )
                if self.on_debug:
                    self.on_debug("TRANSLATED", utterance.speaker_id, {
                        "text": translated.strip(),
                        "source_text": orig_text,
                        "utterance_id": utterance.utterance_id,
                    })

                try:
                    self.translation_queue.put(res, timeout=1.0)
                except queue.Full:
                    logger.warning("Cola de traducción llena. Descartando traducción #%d", utterance.sequence_id)

            except Exception as e:
                logger.error("Error en TranslationWorker para utterance #%s: %s", getattr(utterance, "sequence_id", "?"), e, exc_info=True)
            finally:
                self.confirmed_queue.task_done()

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
        while not self.stop_event.is_set() or not self.translation_queue.empty():
            try:
                trans: TranslationResult = self.translation_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                now = time.monotonic()
                total_latency = max(0.0, now - trans.captured_at)

                from src.ui.subtitle_formatter import SubtitleFormatter
                formatter = SubtitleFormatter(max_chars_per_line=42, max_lines=2)
                formatted_trans = formatter.format(trans.translated_text)

                sub = SubtitleItem(
                    sequence_id=trans.sequence_id,
                    original=trans.source_text,
                    translated=formatted_trans,
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
                    utterance_id=trans.utterance_id,
                    is_complete=trans.is_complete,
                    is_interrupted=trans.is_interrupted,
                    final_asr_pass_used=trans.final_asr_pass_used,
                    completion_score=trans.completion_score,
                    completion_reason=trans.completion_reason,
                    partial_fragments=trans.partial_fragments,
                    word_timestamps=trans.word_timestamps,
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
