"""
test_pipeline_async.py
Pruebas para el PipelineController y el procesamiento desacoplado en hilos.
Verifica orden estricto de secuencias, métricas de latencia y parada limpia.
"""

import unittest
import queue
import time
import threading
from unittest.mock import MagicMock
import numpy as np

from src.pipeline.models import (
    AudioChunk,
    SpeechSegment,
    TranscriptResult,
    TranslationResult,
    SubtitleItem,
    AudioSourceMode,
    VADProfile,
)
from src.pipeline.workers import (
    VADWorker,
    ASRWorker,
    TranslationWorker,
    SubtitleDispatcherWorker,
)
from src.pipeline.controller import PipelineController
from src.asr.base import ASREngine


class MockASR(ASREngine):
    def transcribe(self, audio, language=None, beam_size=1, initial_prompt=None, vad_filter=None, **kwargs):
        return {
            "text": "hello world",
            "language": "en",
            "probability": 0.98,
            "elapsed_time": 0.02,
            "segments": [],
        }

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def compute_type(self) -> str:
        return "int8"


class MockTranslator:
    def translate(self, text, source, target):
        if text == "hello world":
            return "hola mundo"
        return text


class MockCapture:
    def __init__(self):
        self.source_mode = AudioSourceMode.SYSTEM
        self.target_queue = None
        self.stopped = threading.Event()

    def stream_to_queue(self, q, stop_ev):
        self.target_queue = q
        self.stopped = stop_ev

    def stop(self):
        self.stopped.set()


class AsyncPipelineTests(unittest.TestCase):

    def test_workers_end_to_end_flow(self):
        """Verifica el flujo desacoplado completo desde VADWorker hasta SubtitleDispatcher."""
        audio_q = queue.Queue()
        speech_q = queue.Queue()
        transcript_q = queue.Queue()
        trans_q = queue.Queue()
        sub_q = queue.Queue()

        stop_ev = threading.Event()
        vad_mock = MagicMock()
        # VAD emite un bloque de voz de 16000 muestras
        vad_mock.process_chunk.return_value = np.zeros(16000, dtype=np.float32)

        engine = MockASR()
        translator = MockTranslator()

        received_subs = []
        def on_sub(item: SubtitleItem):
            received_subs.append(item)

        vad_w = VADWorker(audio_q, speech_q, vad_mock, stop_ev)
        asr_w = ASRWorker(speech_q, transcript_q, engine, stop_ev)
        trans_w = TranslationWorker(transcript_q, trans_q, translator, stop_ev, target_language="es")
        disp_w = SubtitleDispatcherWorker(trans_q, sub_q, stop_ev, on_subtitle_callback=on_sub)

        workers = [vad_w, asr_w, trans_w, disp_w]
        for w in workers:
            w.start()

        try:
            # Enviamos un chunk de audio
            chunk = AudioChunk(data=np.zeros(1600, dtype=np.float32))
            audio_q.put(chunk)

            # Esperamos a que el subtítulo llegue a la cola final
            sub = sub_q.get(timeout=3.0)
            self.assertEqual(sub.original, "hello world")
            self.assertEqual(sub.translated, "hola mundo")
            self.assertEqual(sub.sequence_id, 1)
            self.assertEqual(len(received_subs), 1)
            self.assertGreater(sub.total_latency, 0.0)

        finally:
            stop_ev.set()
            for w in workers:
                w.join(timeout=1.0)

    def test_pipeline_controller_lifecycle_and_metrics(self):
        """Verifica inicio, parada y reporte de métricas en PipelineController."""
        capture = MockCapture()
        vad_mock = MagicMock()
        vad_mock.process_chunk.return_value = None
        vad_mock.profile = VADProfile.NATURAL
        engine = MockASR()

        controller = PipelineController(
            audio_capture=capture,
            vad=vad_mock,
            asr_engine=engine,
            translator=MockTranslator(),
            source_language="en",
            target_language="es",
        )

        self.assertFalse(controller.is_running)
        controller.start()
        self.assertTrue(controller.is_running)

        metrics = controller.get_metrics()
        self.assertTrue(metrics.active)
        self.assertEqual(metrics.current_source_mode, "system")
        self.assertEqual(metrics.current_vad_profile, "natural")
        self.assertEqual(metrics.current_device, "cpu")

        controller.stop()
        self.assertFalse(controller.is_running)


if __name__ == "__main__":
    unittest.main()
