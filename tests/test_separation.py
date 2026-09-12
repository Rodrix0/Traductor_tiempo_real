"""
test_separation.py
Pruebas unitarias para la capa de separación de fuentes vocales.
"""

import unittest
import numpy as np
from src.separation.manager import SpeechSeparatorManager
from src.separation.spectral_separator import SpectralSpeechSeparator
from src.speakers.models import SeparatedSource


class SpeechSeparationTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1.0, self.sr, endpoint=False, dtype=np.float32)
        # Mezcla sintética de 2 voces: 130Hz y 220Hz
        v1 = 0.4 * np.sin(2 * np.pi * 130.0 * t).astype(np.float32)
        v2 = 0.4 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        self.mixed_audio = v1 + v2

    def test_spectral_separator_produces_two_distinct_sources(self):
        separator = SpectralSpeechSeparator(sample_rate=self.sr)
        separator.load()
        self.assertTrue(separator.is_ready())

        sources = separator.separate(self.mixed_audio, sample_rate=self.sr, start_time=10.0)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].source_id, "source_0")
        self.assertEqual(sources[1].source_id, "source_1")
        self.assertEqual(len(sources[0].audio), len(self.mixed_audio))
        self.assertEqual(len(sources[1].audio), len(self.mixed_audio))
        self.assertEqual(sources[0].start_time, 10.0)
        self.assertEqual(sources[0].end_time, 11.0)

        # Verificar que ambas fuentes tienen energía y son diferentes entre sí
        rms0 = np.sqrt(np.mean(sources[0].audio ** 2))
        rms1 = np.sqrt(np.mean(sources[1].audio ** 2))
        self.assertGreater(rms0, 0.05)
        self.assertGreater(rms1, 0.05)
        diff = np.mean(np.abs(sources[0].audio - sources[1].audio))
        self.assertGreater(diff, 0.05)

    def test_manager_auto_fallback_and_metrics(self):
        manager = SpeechSeparatorManager(mode="auto", sample_rate=self.sr)
        manager.load()
        self.assertTrue(manager.is_ready())

        sources, failed = manager.separate(self.mixed_audio, sample_rate=self.sr, start_time=0.0)
        self.assertFalse(failed)
        self.assertEqual(len(sources), 2)

        metrics = manager.get_metrics()
        self.assertIn("backend", metrics)
        self.assertGreater(metrics["total_calls"], 0)
        self.assertEqual(metrics["failures"], 0)

    def test_manager_disabled_mode_returns_original_audio(self):
        manager = SpeechSeparatorManager(mode="disabled", sample_rate=self.sr)
        manager.load()

        sources, failed = manager.separate(self.mixed_audio, sample_rate=self.sr, start_time=2.0)
        self.assertFalse(failed)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "original")
        np.testing.assert_array_equal(sources[0].audio, self.mixed_audio)


if __name__ == "__main__":
    unittest.main()
