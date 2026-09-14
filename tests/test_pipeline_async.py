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
    UtteranceAssemblerWorker,
    SubtitleDispatcherWorker,
)
from src.pipeline.controller import PipelineController
from src.asr.base import ASREngine


class MockASR(ASREngine):
    def transcribe(self, audio, language=None, beam_size=1, initial_prompt=None, vad_filter=None, **kwargs):
        return {
            "text": "hello world.",
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
        if text == "hello world.":
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

    def test_translation_worker_rejects_raw_transcript(self):
        raw_q, trans_q = queue.Queue(), queue.Queue()
        stop_ev = threading.Event()
        worker = TranslationWorker(raw_q, trans_q, MockTranslator(), stop_ev)
        worker.start()
        raw_q.put(TranscriptResult(1, None, "fragment", "en", 0.9, 0, 1, time.monotonic(), 0, 0.01))
        deadline = time.monotonic() + 1.0
        while raw_q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        stop_ev.set(); worker.join(timeout=1.0)
        self.assertTrue(trans_q.empty(), "un fragmento STT crudo jamás debe llegar al traductor")

    def test_workers_end_to_end_flow(self):
        """Verifica el flujo desacoplado completo desde VADWorker hasta SubtitleDispatcher."""
        audio_q = queue.Queue()
        speech_q = queue.Queue()
        transcript_q = queue.Queue()
        confirmed_q = queue.Queue()
        trans_q = queue.Queue()
        sub_q = queue.Queue()

        stop_ev = threading.Event()
        vad_mock = MagicMock()
        # VAD emite un bloque de voz de 16000 muestras
        vad_mock.process_chunk.return_value = np.zeros(16000, dtype=np.float32)
        vad_mock.speech_samples_count = 0
        vad_mock.last_finalize_reason = "silence"
        vad_mock.last_trailing_silence_ms = 700

        engine = MockASR()
        translator = MockTranslator()

        received_subs = []
        def on_sub(item: SubtitleItem):
            received_subs.append(item)

        vad_w = VADWorker(audio_q, speech_q, vad_mock, stop_ev)
        asr_w = ASRWorker(speech_q, transcript_q, engine, stop_ev)
        assembler_w = UtteranceAssemblerWorker(transcript_q, confirmed_q, engine, stop_ev, source_language="en")
        trans_w = TranslationWorker(confirmed_q, trans_q, translator, stop_ev, target_language="es")
        disp_w = SubtitleDispatcherWorker(trans_q, sub_q, stop_ev, on_subtitle_callback=on_sub)

        workers = [vad_w, asr_w, assembler_w, trans_w, disp_w]
        for w in workers:
            w.start()

        try:
            # Enviamos un chunk de audio
            chunk = AudioChunk(data=np.zeros(1600, dtype=np.float32))
            audio_q.put(chunk)

            # Esperamos a que el subtítulo llegue a la cola final
            sub = sub_q.get(timeout=3.0)
            self.assertEqual(sub.original, "hello world.")
            self.assertEqual(sub.translated, "hola mundo")
            self.assertEqual(sub.sequence_id, 1)
            self.assertEqual(len(received_subs), 1)
            self.assertGreaterEqual(sub.total_latency, 0.0)

        finally:
            stop_ev.set()
            for w in workers:
                w.join(timeout=1.0)

    def test_incomplete_utterance_keeps_original_without_translation(self):
        from src.pipeline.sentence_turn_assembler import SentenceTurnAssembler
        assembler = SentenceTurnAssembler()
        assembler.add_event(speaker_id="A", text="I want to", word_timestamps=[],
                            audio=np.zeros(1600, dtype=np.float32), audio_start=0,
                            audio_end=0.1, is_partial=True, confidence=0.9)
        utterance = assembler.flush_all()[0]
        for complete, interrupted in ((False, False), (False, True), (True, True)):
            with self.subTest(complete=complete, interrupted=interrupted):
                utterance.is_complete = complete
                utterance.is_interrupted = interrupted
                incoming, outgoing = queue.Queue(), queue.Queue()
                stop = threading.Event()
                translator = MagicMock()
                worker = TranslationWorker(incoming, outgoing, translator, stop)
                worker.start()
                try:
                    incoming.put(utterance)
                    result = outgoing.get(timeout=2)
                    self.assertEqual(result.translated_text, "I want to")
                    translator.translate.assert_not_called()
                    translator.clean_source.assert_not_called()
                finally:
                    stop.set()
                    worker.join(timeout=2)

    def test_assembler_worker_uses_capture_progress_for_timeout(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        progress = [1.0, 1.0]
        worker = UtteranceAssemblerWorker(
            incoming, outgoing, None, threading.Event(),
            acoustic_progress=lambda: tuple(progress),
        )
        worker.assembler.add_event(
            speaker_id="A", text="This matters", word_timestamps=[],
            audio=np.zeros(16000, dtype=np.float32), audio_start=0,
            audio_end=1, is_partial=True, confidence=0.9,
        )
        worker._check_timeouts()
        self.assertTrue(outgoing.empty())
        progress[:] = [2.0, 1.0]
        worker._check_timeouts()
        self.assertEqual(outgoing.get_nowait().source_text, "This matters")

    def test_vad_valley_split_timestamps_preserve_continuity(self):
        from src.audio.vad import VoiceActivityDetector
        vad = VoiceActivityDetector(max_speech_duration_ms=240, min_speech_duration_ms=30)
        incoming, outgoing = queue.Queue(), queue.Queue()
        stop = threading.Event()
        worker = VADWorker(incoming, outgoing, vad, stop)
        frames = [np.full(480, 0.1 + i * 0.01, dtype=np.float32) for i in range(12)]
        for frame in frames:
            incoming.put(AudioChunk(data=frame))
        stop.set()
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        worker.flush_pending()
        segments = []
        while not outgoing.empty():
            segments.append(outgoing.get_nowait())
        self.assertGreater(len(segments), 1)
        self.assertEqual(segments[0].start_time, 0.0)
        for left, right in zip(segments, segments[1:]):
            self.assertAlmostEqual(left.end_time, right.start_time, places=3)
        np.testing.assert_array_equal(np.concatenate([s.audio for s in segments]), np.concatenate(frames))
        self.assertAlmostEqual(segments[-1].end_time, 0.36)

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
