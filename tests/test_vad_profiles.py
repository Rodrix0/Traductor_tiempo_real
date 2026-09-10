"""
test_vad_profiles.py
Pruebas para perfiles de latencia VAD (FAST, BALANCED, NATURAL) y división inteligente por valles.
"""

import unittest
import numpy as np
from src.audio.vad import VoiceActivityDetector
from src.pipeline.models import VADProfile


class VADProfileTests(unittest.TestCase):

    def test_profile_silence_durations(self):
        self.assertEqual(VADProfile.FAST.silence_duration_ms, 350)
        self.assertEqual(VADProfile.BALANCED.silence_duration_ms, 550)
        self.assertEqual(VADProfile.NATURAL.silence_duration_ms, 800)

    def test_detector_initialization_with_profile(self):
        vad_fast = VoiceActivityDetector(profile=VADProfile.FAST)
        self.assertEqual(vad_fast.profile, VADProfile.FAST)
        self.assertEqual(vad_fast.silence_duration_ms, 350)

        vad_nat = VoiceActivityDetector(profile=VADProfile.NATURAL)
        self.assertEqual(vad_nat.profile, VADProfile.NATURAL)
        self.assertEqual(vad_nat.silence_duration_ms, 800)

    def test_dynamic_profile_switching(self):
        vad = VoiceActivityDetector(profile=VADProfile.NATURAL)
        self.assertEqual(vad.silence_duration_ms, 800)

        vad.set_profile(VADProfile.FAST)
        self.assertEqual(vad.profile, VADProfile.FAST)
        self.assertEqual(vad.silence_duration_ms, 350)

    def test_fast_profile_cuts_sooner_than_natural(self):
        vad_fast = VoiceActivityDetector(profile=VADProfile.FAST, energy_threshold=0.01)
        vad_nat = VoiceActivityDetector(profile=VADProfile.NATURAL, energy_threshold=0.01)

        # 1600 muestras (100ms) de voz
        speech_frame = np.sin(np.linspace(0, 2 * np.pi * 440, 1600)).astype(np.float32) * 0.1
        # 1600 muestras (100ms) de silencio
        silence_frame = np.zeros(1600, dtype=np.float32)

        # Alimentar 500ms de voz a ambos
        for _ in range(5):
            vad_fast.process_chunk(speech_frame)
            vad_nat.process_chunk(speech_frame)

        # 400ms de silencio (4 frames de 100ms)
        # FAST (350ms) debería cortar, mientras que NATURAL (800ms) aún no
        fast_cut = None
        nat_cut = None
        for _ in range(4):
            f_res = vad_fast.process_chunk(silence_frame)
            n_res = vad_nat.process_chunk(silence_frame)
            if f_res is not None:
                fast_cut = f_res
            if n_res is not None:
                nat_cut = n_res

        self.assertIsNotNone(fast_cut, "El perfil FAST debió haber emitido segmento tras 400ms de silencio")
        self.assertIsNone(nat_cut, "El perfil NATURAL no debió emitir segmento con solo 400ms de silencio")


if __name__ == "__main__":
    unittest.main()
