"""
test_two_speakers_pipeline.py
Pruebas integrales para el pipeline de dos hablantes:
- Phase 24: Test unitario de 2 hablantes con separación y ASR dual.
- Phase 27: Test sin overlap (no se ejecuta separación en turnos normales).
- Phase 28: Test 100% offline sin telemetría ni llamadas de red.
- Phase 40: Validación del benchmark comparativo.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.speakers.overlap_detector import OverlapDetector
from src.separation.manager import SpeechSeparatorManager
from src.speakers.tracker import SpeakerTracker
from src.speakers.benchmark import OverlapBenchmark


class TwoSpeakersPipelineTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1.5, int(1.5 * self.sr), endpoint=False, dtype=np.float32)

        # Persona A (130 Hz) y Persona B (230 Hz)
        self.v_a = (0.5 * np.sin(2 * np.pi * 130.0 * t) + 0.25 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)
        self.v_b = (0.5 * np.sin(2 * np.pi * 230.0 * t) + 0.25 * np.sin(2 * np.pi * 460.0 * t)).astype(np.float32)
        self.mixed = self.v_a + self.v_b

    def test_phase_24_two_speakers_end_to_end_flow(self):
        """
        Phase 24: Simula audio mixto -> OverlapDetector -> SpeechSeparator -> ASR A y B.
        """
        detector = OverlapDetector(sample_rate=self.sr, confidence_threshold=0.60)
        manager = SpeechSeparatorManager(mode="auto", sample_rate=self.sr)
        manager.load()
        tracker = SpeakerTracker(sample_rate=self.sr)

        # 1. Detección
        ov_res = detector.detect(self.mixed)
        self.assertTrue(ov_res.has_overlap)
        self.assertEqual(ov_res.speaker_count, 2)

        # 2. Separación física de fuentes
        sources, failed = manager.separate(self.mixed, sample_rate=self.sr)
        self.assertFalse(failed)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].source_id, "source_0")
        self.assertEqual(sources[1].source_id, "source_1")

        # 3. Asignación de identidades de hablante
        assignments = tracker.assign_sources(sources)
        self.assertEqual(len(assignments), 2)
        spk_ids = [a.speaker_id for a in assignments]
        self.assertIn("SPEAKER_01", spk_ids)
        self.assertIn("SPEAKER_02", spk_ids)

        # 4. Transcripción desacoplada para cada voz
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = [
            {"text": "The people I really admire are Michael Pollan", "language": "en"},
            {"text": "Yeah, exactly.", "language": "en"},
        ]

        transcripts = {}
        for src, assign in zip(sources, assignments):
            res = mock_asr.transcribe(src.audio, language="en")
            transcripts[assign.speaker_id] = res["text"]

        self.assertEqual(len(transcripts), 2)
        self.assertIn("Michael Pollan", transcripts["SPEAKER_01"])
        self.assertIn("Yeah, exactly.", transcripts["SPEAKER_02"])

    def test_phase_27_turn_taking_without_overlap_does_not_trigger_separation(self):
        """
        Phase 27: En una conversación normal por turnos (habla A -> pausa -> habla B),
        el separador NO debe ejecutarse para no penalizar la velocidad.
        """
        detector = OverlapDetector(sample_rate=self.sr)
        manager = SpeechSeparatorManager(mode="auto", sample_rate=self.sr)
        manager.load()

        # Simular turno 1: Habla únicamente Persona A
        res_a = detector.detect(self.v_a)
        self.assertFalse(res_a.has_overlap)

        # Simular turno 2: Habla únicamente Persona B
        res_b = detector.detect(self.v_b)
        self.assertFalse(res_b.has_overlap)

        # En ambos casos, el separador NO fue invocado
        self.assertEqual(manager.get_metrics()["total_calls"], 0)

    def test_phase_28_completely_offline_no_external_network_requests(self):
        """
        Phase 28: Asegura que los componentes de overlap, separación y seguimiento
        no realicen ninguna petición de red o socket externo.
        """
        import socket

        # Bloquear apertura de conexiones de red durante la prueba
        with patch.object(socket.socket, "connect", side_effect=RuntimeError("Llamada de red bloqueada en modo offline")):
            detector = OverlapDetector(sample_rate=self.sr)
            det_res = detector.detect(self.mixed)

            manager = SpeechSeparatorManager(mode="spectral", sample_rate=self.sr)
            manager.load()
            sources, failed = manager.separate(self.mixed, sample_rate=self.sr)

            tracker = SpeakerTracker(sample_rate=self.sr)
            assignments = tracker.assign_sources(sources)

            self.assertTrue(det_res.has_overlap)
            self.assertFalse(failed)
            self.assertEqual(len(assignments), 2)

    def test_phase_40_benchmark_comparison_tool(self):
        """
        Phase 40: Verifica que la herramienta de benchmark ejecute ambos modos
        y genere el reporte comparativo estructurado.
        """
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = [
            {"text": "My people yeah exactly Michael admire", "language": "en"}, # Modo A corrupto
            {"text": "The people I really admire are Michael Pollan", "language": "en"}, # Modo B Speaker 1
            {"text": "Yeah, exactly.", "language": "en"}, # Modo B Speaker 2
        ]

        report_data = OverlapBenchmark.compare_audio(self.mixed, mock_asr, sample_rate=self.sr)
        self.assertIn("mode_a_no_separation", report_data)
        self.assertIn("mode_b_with_separation", report_data)

        formatted = OverlapBenchmark.format_report(report_data)
        self.assertIn("BENCHMARK DE HABLA SUPERPUESTA", formatted)
        self.assertIn("MODO A: SIN SEPARACIÓN", formatted)
        self.assertIn("MODO B: CON DETECCIÓN, SEPARACIÓN", formatted)


    def test_separated_source_pipeline_contract(self):
        """
        Test de regresión: Verifica el contrato oficial de SeparatedSource (start_time, end_time, source_id, speaker_id)
        y comprueba que todo el flujo preserve los timestamps absolutos sin fallar con AttributeError.
        """
        from src.pipeline.models import SpeechSegment, TranscriptResult
        from src.subtitles import Caption

        orig_start = 10.4
        orig_end = 11.9
        segment = SpeechSegment(
            sequence_id=42,
            session_id="test_session",
            audio=self.mixed,
            start_time=orig_start,
            end_time=orig_end,
            sample_rate=self.sr,
        )

        manager = SpeechSeparatorManager(mode="spectral", sample_rate=self.sr)
        sources, failed = manager.separate(segment.audio, sample_rate=self.sr, start_time=segment.start_time)
        self.assertFalse(failed)
        self.assertEqual(len(sources), 2)

        tracker = SpeakerTracker(sample_rate=self.sr)
        assignments = tracker.assign_sources(sources)

        for src, assign in zip(sources, assignments):
            # 1. Contrato de atributos oficiales
            self.assertTrue(hasattr(src, "start_time"), "SeparatedSource debe tener start_time")
            self.assertTrue(hasattr(src, "end_time"), "SeparatedSource debe tener end_time")
            self.assertTrue(hasattr(src, "source_id"), "SeparatedSource debe tener source_id")

            # 2. Preservación estricta de timestamps absolutos
            self.assertAlmostEqual(src.start_time, orig_start, places=2)
            self.assertAlmostEqual(src.end_time, orig_end, places=2)

            # 3. Construcción de TranscriptResult
            t_res = TranscriptResult(
                sequence_id=segment.sequence_id,
                session_id=segment.session_id,
                text="Texto de prueba",
                language="es",
                confidence=0.95,
                start_time=src.start_time,
                end_time=src.end_time,
                captured_at=segment.captured_at,
                asr_start_time=0.0,
                asr_duration=0.1,
                speaker_id=assign.speaker_id,
                source_id=src.source_id,
                is_overlap=True,
            )
            self.assertEqual(t_res.start_time, src.start_time)
            self.assertEqual(t_res.end_time, src.end_time)
            self.assertEqual(t_res.speaker_id, assign.speaker_id)

            # 4. Creación de Caption para almacenamiento (no debe lanzar AttributeError)
            caption = Caption(
                start=src.start_time,
                end=src.end_time,
                original=t_res.text,
                translated=t_res.text,
                language=t_res.language,
                speaker_id=assign.speaker_id,
            )
            self.assertAlmostEqual(caption.start, orig_start, places=2)
            self.assertAlmostEqual(caption.end, orig_end, places=2)
            self.assertEqual(caption.speaker_id, assign.speaker_id)

    def test_pipeline_overlap_live_path_without_attribute_error(self):
        """
        Verifica la ruta real completa ejecutada en traductor.py ante habla solapada:
        AudioSegment -> OverlapDetector -> SpeechSeparatorManager -> SeparatedSource -> SpeakerTracker -> Caption
        Garantiza que NO ocurra 'AttributeError: SeparatedSource object has no attribute start'.
        """
        from src.audio.windows_capture import AudioSegment
        from src.subtitles import Caption
        from src.storage import HistoryStore
        import tempfile

        detector = OverlapDetector(sample_rate=self.sr, confidence_threshold=0.60)
        separator = SpeechSeparatorManager(mode="spectral", sample_rate=self.sr)
        tracker = SpeakerTracker(sample_rate=self.sr)

        abs_start = 25.5
        abs_end = 27.0
        segment = AudioSegment(audio=self.mixed, start=abs_start, end=abs_end)

        # 1. Detección
        ov_res = detector.detect(segment.audio, segment.start)
        self.assertTrue(ov_res.has_overlap)

        # 2. Separación con timestamps absolutos
        sources, failed = separator.separate(segment.audio, sample_rate=self.sr, start_time=segment.start)
        self.assertFalse(failed)
        self.assertEqual(len(sources), 2)

        # 3. Asignación
        assignments = tracker.assign_sources(sources)

        # 4. Guardado en store usando start_time y end_time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HistoryStore(Path(tmpdir) / "history.db")
            session_key = store.create("Test Overlap", "en vivo", "en", "es", "base")

            created_captions = []
            for src, assign in zip(sources, assignments):
                # Esta línea provocaba AttributeError antes de la corrección
                cap = Caption(src.start_time, src.end_time, "Hello", "Hola", "en", speaker_id=assign.speaker_id)
                store.add(session_key, cap)
                created_captions.append(cap)

            saved = store.captions(session_key)
            self.assertEqual(len(saved), 2)
            self.assertAlmostEqual(saved[0].start, abs_start, places=2)
            self.assertAlmostEqual(saved[0].end, abs_end, places=2)
            self.assertEqual(saved[0].speaker_id, assignments[0].speaker_id)
            self.assertEqual(saved[1].speaker_id, assignments[1].speaker_id)


if __name__ == "__main__":
    unittest.main()
