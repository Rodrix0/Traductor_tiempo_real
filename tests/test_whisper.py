"""Pruebas para el motor de transcripción local faster-whisper."""

import unittest
import numpy as np

from src.asr.whisper_engine import WhisperEngine
from config.settings import WHISPER_MODEL_SIZE


class TestWhisperEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Inicializar el motor una sola vez para la suite de pruebas."""
        cls.engine = WhisperEngine(model_size=WHISPER_MODEL_SIZE)

    def test_engine_initialization(self):
        """Verifica que el modelo se haya cargado correctamente en CPU o GPU."""
        self.assertIsNotNone(self.engine.model)
        self.assertIn(self.engine.actual_device, ["cpu", "cuda"])
        self.assertIsNotNone(self.engine.actual_compute_type)

    def test_transcription_result_structure(self):
        """Verifica la estructura del resultado devuelto por transcribe()."""
        # Audio sintético de 1 segundo (16.000 muestras)
        dummy_audio = np.zeros(16000, dtype=np.float32)

        result = self.engine.transcribe(dummy_audio, language="es")

        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("language", result)
        self.assertIn("probability", result)
        self.assertIn("elapsed_time", result)
        self.assertIn("segments", result)

        self.assertIsInstance(result["text"], str)
        self.assertIsInstance(result["elapsed_time"], float)
        self.assertGreater(result["elapsed_time"], 0.0)

    def test_transcription_latency(self):
        """Verifica que la latencia de procesamiento sea adecuada para tiempo real."""
        audio_1s = np.zeros(16000, dtype=np.float32)
        result = self.engine.transcribe(audio_1s)
        # En CPU int8 o GPU float16 debería tardar menos de 2.0s para un audio de 1s
        self.assertLess(result["elapsed_time"], 2.0, "La inferencia tomó demasiado tiempo.")


if __name__ == "__main__":
    unittest.main()
