"""
test_overlap_detector.py
Pruebas unitarias para el detector de habla solapada (OverlapDetector).
"""

import unittest
import numpy as np
from src.speakers.overlap_detector import OverlapDetector
from src.speakers.models import OverlapResult


class OverlapDetectorTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        self.detector = OverlapDetector(
            sample_rate=self.sr,
            confidence_threshold=0.60,
            min_overlap_duration=0.15,
            pre_roll_ms=400,
            post_roll_ms=400,
        )

    def test_silence_produces_no_overlap(self):
        silence = np.zeros(self.sr * 2, dtype=np.float32)
        res = self.detector.detect(silence)
        self.assertFalse(res.has_overlap)
        self.assertEqual(res.speaker_count, 1)
        self.assertEqual(res.confidence, 0.0)

    def test_single_speaker_harmonic_voice_produces_no_overlap(self):
        # Simular una sola voz con tono fundamental 140Hz y sus armónicos naturales
        t = np.linspace(0, 1.5, int(1.5 * self.sr), endpoint=False, dtype=np.float32)
        f0 = 140.0
        single_voice = (
            0.50 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        ).astype(np.float32)

        res = self.detector.detect(single_voice)
        self.assertFalse(res.has_overlap)
        self.assertEqual(res.speaker_count, 1)

    def test_dual_speakers_simultaneous_voices_triggers_overlap(self):
        # Simular dos hablantes simultáneos: Persona A (125Hz, voz masculina) + Persona B (220Hz, voz femenina)
        t = np.linspace(0, 1.5, int(1.5 * self.sr), endpoint=False, dtype=np.float32)
        voice_a = (0.4 * np.sin(2 * np.pi * 125.0 * t) + 0.2 * np.sin(2 * np.pi * 250.0 * t)).astype(np.float32)
        voice_b = (0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        mixed = (voice_a + voice_b)

        res = self.detector.detect(mixed, base_start_time=5.0)
        self.assertTrue(res.has_overlap)
        self.assertEqual(res.speaker_count, 2)
        self.assertGreaterEqual(res.confidence, 0.60)
        self.assertGreaterEqual(res.start_time, 5.0)

    def test_pre_roll_and_post_roll_margins(self):
        # Audio de 4 segundos
        t = np.linspace(0, 4.0, int(4.0 * self.sr), endpoint=False, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 150.0 * t).astype(np.float32)

        # Crear un OverlapResult simulado entre 1.5s y 2.5s
        res = OverlapResult(
            has_overlap=True,
            speaker_count=2,
            confidence=0.85,
            start_time=1.5,
            end_time=2.5,
        )

        cropped, abs_start, abs_end = self.detector.get_overlap_region_with_margins(
            full_audio=audio,
            result=res,
            audio_start_time=0.0,
        )

        # Con pre-roll de 400ms y post-roll de 400ms:
        # Inicio esperado: max(0, 1.5 - 0.4) = 1.1s
        # Fin esperado: min(4.0, 2.5 + 0.4) = 2.9s
        self.assertAlmostEqual(abs_start, 1.1, places=2)
        self.assertAlmostEqual(abs_end, 2.9, places=2)
        expected_samples = int((2.9 - 1.1) * self.sr)
        self.assertAlmostEqual(len(cropped), expected_samples, delta=10)


if __name__ == "__main__":
    unittest.main()
