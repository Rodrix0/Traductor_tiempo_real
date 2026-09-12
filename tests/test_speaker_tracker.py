"""
test_speaker_tracker.py
Pruebas unitarias para SpeakerTracker, resolución de permutación (Phase 26)
y buffer de enunciados desacoplado (Phase 25).
"""

import unittest
import numpy as np
from src.speakers.tracker import SpeakerTracker
from src.speakers.models import SeparatedSource
from src.speakers.utterance_buffer import MultiSpeakerUtteranceManager


class SpeakerTrackerTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        self.tracker = SpeakerTracker(sample_rate=self.sr)
        t = np.linspace(0, 1.0, self.sr, endpoint=False, dtype=np.float32)

        # Voz sintética Persona A (grave: 120Hz + armónicos)
        self.audio_a = (0.5 * np.sin(2 * np.pi * 120.0 * t) + 0.25 * np.sin(2 * np.pi * 240.0 * t)).astype(np.float32)
        # Voz sintética Persona B (aguda: 240Hz + armónicos)
        self.audio_b = (0.5 * np.sin(2 * np.pi * 240.0 * t) + 0.25 * np.sin(2 * np.pi * 480.0 * t)).astype(np.float32)

    def test_solo_speech_registration_and_tracking(self):
        # Registrar Persona A hablando sola
        spk_1 = self.tracker.register_solo_speech(self.audio_a, duration=1.0)
        self.assertEqual(spk_1, "SPEAKER_01")

        # Persona A vuelve a hablar sola
        spk_1_again = self.tracker.register_solo_speech(self.audio_a, duration=1.0)
        self.assertEqual(spk_1_again, "SPEAKER_01")

        # Ahora habla Persona B sola
        spk_2 = self.tracker.register_solo_speech(self.audio_b, duration=1.0)
        self.assertEqual(spk_2, "SPEAKER_02")

    def test_speaker_swap_permutation_resolution(self):
        """
        Phase 26: Simula la inversión de canales del separador entre Chunk 1 y Chunk 2.
        SpeakerTracker debe corregir la permutación automáticamente.
        """
        # Registrar previamente a Persona A y Persona B
        id_a = self.tracker.register_solo_speech(self.audio_a, duration=1.0)
        id_b = self.tracker.register_solo_speech(self.audio_b, duration=1.0)
        self.assertEqual(id_a, "SPEAKER_01")
        self.assertEqual(id_b, "SPEAKER_02")

        # Chunk 1: Separador devuelve: source_0 = A, source_1 = B
        chunk1_sources = [
            SeparatedSource(source_id="source_0", audio=self.audio_a, sample_rate=self.sr),
            SeparatedSource(source_id="source_1", audio=self.audio_b, sample_rate=self.sr),
        ]
        assignments_1 = self.tracker.assign_sources(chunk1_sources)
        map1 = {a.source_id: a.speaker_id for a in assignments_1}
        self.assertEqual(map1["source_0"], "SPEAKER_01")
        self.assertEqual(map1["source_1"], "SPEAKER_02")

        # Chunk 2: El separador INVIERTE los canales: source_0 = B, source_1 = A
        chunk2_sources = [
            SeparatedSource(source_id="source_0", audio=self.audio_b, sample_rate=self.sr),
            SeparatedSource(source_id="source_1", audio=self.audio_a, sample_rate=self.sr),
        ]
        assignments_2 = self.tracker.assign_sources(chunk2_sources)
        map2 = {a.source_id: a.speaker_id for a in assignments_2}

        # ¡SpeakerTracker debe haber corregido la permutación!
        self.assertEqual(map2["source_0"], "SPEAKER_02")
        self.assertEqual(map2["source_1"], "SPEAKER_01")
        self.assertGreaterEqual(self.tracker.speaker_swaps_count, 1)

    def test_interruption_utterance_buffers_remain_isolated(self):
        """
        Phase 25: Simula la interrupción de Persona B sobre Persona A.
        Verifica que los buffers de texto de cada hablante no se contaminen mutuamente.
        """
        manager = MultiSpeakerUtteranceManager()

        # Persona A empieza a hablar (no final)
        res_a1 = manager.add_transcript("SPEAKER_01", "One of the people I really admire are", 10.0, 12.0, is_final=False)
        self.assertIsNone(res_a1)

        # Persona B interrumpe con una exclamación breve
        res_b = manager.add_transcript("SPEAKER_02", "Yeah, absolutely.", 11.5, 12.2, is_final=True)
        self.assertEqual(res_b, "Yeah, absolutely.")

        # Persona A continúa y termina su frase
        res_a2 = manager.add_transcript("SPEAKER_01", "Michael Pollan and Michael Singer.", 12.1, 14.0, is_final=True)
        self.assertEqual(res_a2, "One of the people I really admire are Michael Pollan and Michael Singer.")


if __name__ == "__main__":
    unittest.main()
