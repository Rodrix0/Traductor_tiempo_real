"""
test_double_vad.py
Pruebas para verificar la protección contra Doble VAD y la sanitización de modelos monolingües.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.pipeline.models import SpeechSegment, TranscriptResult
from src.asr.base import ASREngine
from src.asr.manager import sanitize_model_for_language


class DummyEngine(ASREngine):
    def __init__(self):
        self.last_vad_filter = None
        self.transcribe_called = False

    def transcribe(self, audio, language=None, beam_size=1, initial_prompt=None, vad_filter=None, **kwargs):
        self.transcribe_called = True
        self.last_vad_filter = vad_filter
        return {
            "text": "test speech",
            "language": language or "en",
            "probability": 0.95,
            "elapsed_time": 0.05,
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


class DoubleVADTests(unittest.TestCase):

    def test_external_vad_disables_whisper_vad_filter(self):
        """Si el segmento ya fue recortado por el VAD externo, vad_filter debe ser False."""
        engine = DummyEngine()
        segment = SpeechSegment(
            sequence_id=1,
            session_id="test",
            audio=np.zeros(16000, dtype=np.float32),
            start_time=0.0,
            end_time=1.0,
            external_vad_processed=True,
        )

        res = engine.transcribe_segment(segment)
        self.assertTrue(engine.transcribe_called)
        self.assertFalse(engine.last_vad_filter, "vad_filter en ASR debe ser False cuando external_vad_processed=True")
        self.assertEqual(res.text, "test speech")
        self.assertEqual(res.sequence_id, 1)

    def test_non_external_segment_preserves_default_vad(self):
        """Si el segmento no fue procesado externamente, no se fuerza vad_filter=False."""
        engine = DummyEngine()
        segment = SpeechSegment(
            sequence_id=2,
            session_id="test",
            audio=np.zeros(16000, dtype=np.float32),
            start_time=0.0,
            end_time=1.0,
            external_vad_processed=False,
        )

        engine.transcribe_segment(segment)
        self.assertIsNone(engine.last_vad_filter)

    def test_sanitize_model_for_language(self):
        """small.en solo se debe permitir si el idioma es estrictamente inglés."""
        # En inglés, small.en se preserva
        self.assertEqual(sanitize_model_for_language("small.en", "en"), "small.en")
        
        # En español o automático, se convierte a multilingüe
        self.assertEqual(sanitize_model_for_language("small.en", "es"), "small")
        self.assertEqual(sanitize_model_for_language("small.en", "auto"), "small")
        self.assertEqual(sanitize_model_for_language("small.en", None), "small")

        # Modelos multilingües normales se conservan
        self.assertEqual(sanitize_model_for_language("small", "es"), "small")
        self.assertEqual(sanitize_model_for_language("base", "en"), "base")


if __name__ == "__main__":
    unittest.main()
