"""
controller.py
Coordinador del pipeline asíncrono (PipelineController).
Orquesta la captura WASAPI, VAD, ASR, Traducción y Subtítulos en tiempo real,
garantizando desacoplamiento completo y baja latencia constante.
"""

import threading
import queue
import time
import logging
from typing import Optional, Callable, Any, List

from src.pipeline.models import (
    AudioSourceMode,
    VADProfile,
    AudioChunk,
    SpeechSegment,
    TranscriptResult,
    TranslationResult,
    SubtitleItem,
    PipelineMetrics,
)
from src.asr.base import ASREngine
from src.pipeline.workers import (
    VADWorker,
    ASRWorker,
    TranslationWorker,
    SubtitleDispatcherWorker,
    SeparationWorker,
)

logger = logging.getLogger(__name__)


class PipelineController:
    """Controlador central del flujo asíncrono de audio a subtítulo traducido con soporte multihablante."""

    def __init__(
        self,
        audio_capture: Any,
        vad: Any,
        asr_engine: ASREngine,
        translator: Optional[Any] = None,
        source_language: Optional[str] = None,
        target_language: str = "es",
        on_subtitle: Optional[Callable[[SubtitleItem], None]] = None,
        session_id: Optional[str] = None,
        max_queue_size: int = 50,
        overlap_detector: Optional[Any] = None,
        separator_manager: Optional[Any] = None,
        speaker_tracker: Optional[Any] = None,
    ):
        self.audio_capture = audio_capture
        self.vad = vad
        self.asr_engine = asr_engine
        self.translator = translator
        self.source_language = source_language
        self.target_language = target_language
        self.on_subtitle = on_subtitle
        self.session_id = session_id
        self.overlap_detector = overlap_detector
        self.separator_manager = separator_manager
        self.speaker_tracker = speaker_tracker

        # Colas acotadas entre fases
        self.audio_queue: queue.Queue = queue.Queue(maxsize=max_queue_size * 2)
        self.separation_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.speech_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.transcript_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.translation_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.subtitle_queue: queue.Queue = queue.Queue(maxsize=max_queue_size * 2)

        # Estado de ejecución
        self.stop_event = threading.Event()
        self._is_running = False
        self.workers: List[threading.Thread] = []

        # Métricas de latencia
        self._processed_count = 0
        self._last_asr_ms = 0.0
        self._last_trans_ms = 0.0
        self._last_total_ms = 0.0
        self._sum_asr_ms = 0.0
        self._sum_trans_ms = 0.0
        self._sum_total_ms = 0.0

    def _on_sub_internal(self, item: SubtitleItem) -> None:
        """Registra métricas y delega al callback de usuario."""
        self._processed_count += 1
        self._last_asr_ms = item.asr_latency * 1000.0
        self._last_trans_ms = item.trans_latency * 1000.0
        self._last_total_ms = item.total_latency * 1000.0

        self._sum_asr_ms += self._last_asr_ms
        self._sum_trans_ms += self._last_trans_ms
        self._sum_total_ms += self._last_total_ms

        if self.on_subtitle:
            self.on_subtitle(item)

    def start(self) -> None:
        """Inicia el pipeline completo: captura y trabajadores en hilos independientes."""
        if self._is_running:
            logger.warning("PipelineController ya está corriendo.")
            return

        self.stop_event.clear()
        self._is_running = True

        # Iniciar captura de audio transmitiendo a self.audio_queue
        self.audio_capture.stream_to_queue(self.audio_queue, self.stop_event)

        # Configurar prompt inicial si el traductor lo ofrece
        initial_prompt = None
        if self.translator and hasattr(self.translator, "get_whisper_prompt"):
            initial_prompt = self.translator.get_whisper_prompt()

        # Crear e iniciar trabajadores
        vad_worker = VADWorker(
            audio_queue=self.audio_queue,
            speech_queue=self.speech_queue,
            vad=self.vad,
            stop_event=self.stop_event,
            session_id=self.session_id,
            overlap_detector=self.overlap_detector,
            separation_queue=self.separation_queue,
        )

        self.workers = [vad_worker]

        if self.separator_manager is not None:
            sep_worker = SeparationWorker(
                separation_queue=self.separation_queue,
                speech_queue=self.speech_queue,
                separator_manager=self.separator_manager,
                stop_event=self.stop_event,
                overlap_detector=self.overlap_detector,
                speaker_tracker=self.speaker_tracker,
            )
            self.workers.append(sep_worker)

        asr_worker = ASRWorker(
            speech_queue=self.speech_queue,
            transcript_queue=self.transcript_queue,
            engine=self.asr_engine,
            stop_event=self.stop_event,
            source_language=self.source_language,
            initial_prompt=initial_prompt,
            speaker_tracker=self.speaker_tracker,
        )
        self.workers.append(asr_worker)

        trans_worker = TranslationWorker(
            transcript_queue=self.transcript_queue,
            translation_queue=self.translation_queue,
            translator=self.translator,
            stop_event=self.stop_event,
            target_language=self.target_language,
        )
        self.workers.append(trans_worker)

        sub_worker = SubtitleDispatcherWorker(
            translation_queue=self.translation_queue,
            subtitle_queue=self.subtitle_queue,
            stop_event=self.stop_event,
            on_subtitle_callback=self._on_sub_internal,
        )
        self.workers.append(sub_worker)

        for w in self.workers:
            w.start()

        logger.info("PipelineController iniciado exitosamente con %d etapas asíncronas.", len(self.workers))

    def stop(self) -> None:
        """Detiene de forma limpia el pipeline y todos los trabajadores."""
        if not self._is_running:
            return

        logger.info("Deteniendo PipelineController...")
        self.stop_event.set()

        # Detener captura
        try:
            self.audio_capture.stop()
        except Exception as e:
            logger.warning("Error deteniendo audio_capture: %s", e)

        # Esperar a que los hilos finalicen
        for w in self.workers:
            w.join(timeout=1.5)

        self.workers.clear()
        self._is_running = False
        logger.info("PipelineController detenido limpiamente.")

    def get_subtitle(self, block: bool = False, timeout: float = 0.1) -> Optional[SubtitleItem]:
        """Obtiene el siguiente subtítulo disponible en la cola (para actualización de UI)."""
        try:
            return self.subtitle_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_metrics(self) -> PipelineMetrics:
        """Devuelve el estado de salud y métricas de latencia actuales del pipeline."""
        count = max(1, self._processed_count)
        avg_asr = self._sum_asr_ms / count if self._processed_count > 0 else 0.0
        avg_trans = self._sum_trans_ms / count if self._processed_count > 0 else 0.0
        avg_total = self._sum_total_ms / count if self._processed_count > 0 else 0.0

        current_mode = "system"
        if hasattr(self.audio_capture, "source_mode"):
            current_mode = self.audio_capture.source_mode.value

        current_vad = "natural"
        if hasattr(self.vad, "profile"):
            current_vad = self.vad.profile.value

        model_name = getattr(self.asr_engine, "model_size", "unknown")
        device_name = getattr(self.asr_engine, "device", "cpu")

        swaps = self.speaker_tracker.speaker_swaps_count if self.speaker_tracker else 0
        active_spks = len(self.speaker_tracker.profiles) if self.speaker_tracker and self.speaker_tracker.profiles else 1

        return PipelineMetrics(
            active=self._is_running,
            audio_queue_size=self.audio_queue.qsize(),
            speech_queue_size=self.speech_queue.qsize(),
            transcript_queue_size=self.transcript_queue.qsize(),
            subtitle_queue_size=self.subtitle_queue.qsize(),
            separation_queue_size=self.separation_queue.qsize(),
            total_segments_processed=self._processed_count,
            last_asr_latency_ms=round(self._last_asr_ms, 1),
            last_trans_latency_ms=round(self._last_trans_ms, 1),
            last_total_latency_ms=round(self._last_total_ms, 1),
            avg_asr_latency_ms=round(avg_asr, 1),
            avg_trans_latency_ms=round(avg_trans, 1),
            avg_total_latency_ms=round(avg_total, 1),
            current_source_mode=current_mode,
            current_vad_profile=current_vad,
            current_asr_model=model_name,
            current_device=device_name,
            speaker_swaps_corrected=swaps,
            active_speakers_count=active_spks,
        )

    def set_vad_profile(self, profile: VADProfile) -> None:
        """Actualiza dinámicamente el perfil VAD."""
        if hasattr(self.vad, "set_profile"):
            self.vad.set_profile(profile)

    def set_source_language(self, language: Optional[str]) -> None:
        """Actualiza el idioma de origen para el worker ASR."""
        self.source_language = language
        for w in self.workers:
            if isinstance(w, ASRWorker):
                w.source_language = language

    def set_target_language(self, language: str) -> None:
        """Actualiza el idioma de destino para el worker de traducción."""
        self.target_language = language
        for w in self.workers:
            if isinstance(w, TranslationWorker):
                w.target_language = language

    @property
    def is_running(self) -> bool:
        return self._is_running
