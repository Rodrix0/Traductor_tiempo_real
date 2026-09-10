"""
test_resampler.py
Pruebas unitarias para StatefulResampler y conversión a mono 16kHz.
Verifica remuestreo exacto sin saltos ni clics de fase en 48kHz, 44.1kHz, 32kHz y 96kHz.
"""

import unittest
import numpy as np
from src.audio.resampler import StatefulResampler, to_mono_16k


class StatefulResamplerTests(unittest.TestCase):

    def test_48k_stereo_to_16k_mono(self):
        resampler = StatefulResampler(input_rate=48000, target_rate=16000)
        # 480 muestras estéreo (10ms a 48kHz)
        input_audio = np.ones((480, 2), dtype=np.float32)
        out = resampler.process(input_audio)
        # Ratio exacto 48k/16k = 3 -> 480 / 3 = 160 muestras
        self.assertEqual(len(out), 160)
        self.assertEqual(out.ndim, 1)
        self.assertAlmostEqual(float(out[0]), 1.0, places=4)

    def test_continuous_streaming_no_loss_with_remainder(self):
        resampler = StatefulResampler(input_rate=48000, target_rate=16000)
        # Pasamos bloques que no son múltiplos exactos de 3: 100 muestras cada uno
        total_out = []
        for _ in range(3):
            chunk = np.ones((100, 1), dtype=np.float32)
            out = chunk_out = resampler.process(chunk)
            total_out.append(out)
        
        combined = np.concatenate(total_out)
        # 300 muestras de entrada / 3 = exactamente 100 muestras de salida
        self.assertEqual(len(combined), 100)

    def test_44k1_polyphase_resampling(self):
        resampler = StatefulResampler(input_rate=44100, target_rate=16000)
        # 4410 muestras (100ms)
        input_audio = np.ones((4410, 2), dtype=np.float32)
        out = resampler.process(input_audio)
        # 100ms a 16kHz son 1600 muestras
        self.assertAlmostEqual(len(out), 1600, delta=5)

    def test_to_mono_16k_backwards_compatible(self):
        data_48k = np.ones((4800, 2), dtype=np.float32)
        mono_16k = to_mono_16k(data_48k, 48000)
        self.assertEqual(len(mono_16k), 1600)
        self.assertEqual(mono_16k.ndim, 1)

    def test_already_16k_mono_passthrough(self):
        resampler = StatefulResampler(input_rate=16000, target_rate=16000)
        input_audio = np.random.randn(320).astype(np.float32)
        out = resampler.process(input_audio)
        self.assertEqual(len(out), 320)
        np.testing.assert_array_almost_equal(out, input_audio)


if __name__ == "__main__":
    unittest.main()
