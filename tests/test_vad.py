"""Pruebas unitarias para el detector de actividad vocal (VAD)."""

import unittest
import numpy as np

from src.audio.vad import VoiceActivityDetector


class TestVoiceActivityDetector(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            energy_threshold=0.02,
            silence_duration_ms=200,      # 200ms para pruebas rápidas
            min_speech_duration_ms=100,
            max_speech_duration_ms=2000,
            pre_speech_padding_ms=100,
        )

    def test_energy_calculation_zero(self):
        """Un frame de ceros debe tener energía 0.0."""
        zero_frame = np.zeros(480, dtype=np.float32)
        energy = self.vad.calculate_energy(zero_frame)
        self.assertEqual(energy, 0.0)

    def test_energy_calculation_signal(self):
        """Una onda senoidal debe dar un RMS cercano a su amplitud / sqrt(2)."""
        amplitude = 0.5
        t = np.linspace(0, 0.03, 480, endpoint=False)
        sine_frame = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        energy = self.vad.calculate_energy(sine_frame)
        expected_rms = amplitude / np.sqrt(2)
        self.assertAlmostEqual(energy, expected_rms, places=2)

    def test_speech_detection_and_silence_finalization(self):
        """Verifica la transición de silencio -> habla -> corte tras silencio prolongado."""
        frame_size = 480  # 30ms a 16kHz
        loud_frame = np.full(frame_size, 0.1, dtype=np.float32)  # Energía 0.1 > umbral 0.02
        silent_frame = np.zeros(frame_size, dtype=np.float32)

        # 1. Alimentar silencio inicial
        result = self.vad.process_frame(silent_frame)
        self.assertIsNone(result)
        self.assertFalse(self.vad.is_speaking)

        # 2. Alimentar varios frames de voz (6 frames = 180ms)
        for _ in range(6):
            res = self.vad.process_frame(loud_frame)
            self.assertIsNone(res)
            self.assertTrue(self.vad.is_speaking)

        # 3. Alimentar silencio (silence_duration_ms = 200ms => ~7 frames de 30ms)
        final_segment = None
        for _ in range(8):
            res = self.vad.process_frame(silent_frame)
            if res is not None:
                final_segment = res
                break

        self.assertIsNotNone(final_segment, "El VAD debió emitir el segmento completo tras el silencio.")
        self.assertFalse(self.vad.is_speaking)
        self.assertGreater(len(final_segment), 6 * frame_size)

    def test_reset(self):
        """Verifica que reset() limpie el estado interno por completo."""
        loud_frame = np.full(480, 0.1, dtype=np.float32)
        self.vad.process_frame(loud_frame)
        self.assertTrue(self.vad.is_speaking)

        self.vad.reset()
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(self.vad.speech_samples_count, 0)
        self.assertEqual(len(self.vad.current_speech_frames), 0)


if __name__ == "__main__":
    unittest.main()
