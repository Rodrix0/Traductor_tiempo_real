"""
tests/test_speaker_diarization_reassembly.py
Suite integral de verificación y regresión para:
- TEST A: Alternancia consecutiva de hablantes (A -> B -> A) con SpeakerMemory y SpeakerTurnDetector.
- TEST B: Respuestas e interjecciones cortas sin absorción ni colapso de hablante.
- TEST C: Análisis de discontinuidad sintáctica y reensamblado multi-factor (SegmentReassembler).
- TEST D: Aislamiento estricto de hablantes (cero fusiones cruzadas entre personas distintas).
- TEST E: AudioRingBuffer con padding temporal (pre-roll y post-roll).
- TEST F: Cálculo riguroso de métricas (DER, WER, CER y seguimiento de latencia).
"""

import unittest
import numpy as np

from src.speakers.tracker import SpeakerTracker
from src.speakers.memory import SpeakerMemory, estimate_pitch_f0
from src.speakers.turn_detector import SpeakerTurnDetector
from src.audio.ring_buffer import AudioRingBuffer
from src.pipeline.continuity_analyzer import SegmentContinuityAnalyzer
from src.pipeline.segment_reassembler import SegmentReassembler, ReassembledSegment
from src.utils.metrics import compute_der, compute_wer, compute_cer, PipelineLatencyTracker
from src.asr.anomaly_detector import STTAnomalyDetector


class SpeakerDiarizationReassemblyTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        # Generación de señales sintéticas con perfiles tímbricos y de pitch contrastantes
        t_1s = np.linspace(0, 1.0, self.sr, endpoint=False, dtype=np.float32)
        t_05s = np.linspace(0, 0.5, int(0.5 * self.sr), endpoint=False, dtype=np.float32)

        # Voz A: Tono fundamental ~130 Hz (masculino/grave) con armónicos
        self.voice_a_long = (
            0.5 * np.sin(2 * np.pi * 130.0 * t_1s)
            + 0.3 * np.sin(2 * np.pi * 260.0 * t_1s)
            + 0.15 * np.sin(2 * np.pi * 390.0 * t_1s)
        ).astype(np.float32)

        # Voz B: Tono fundamental ~230 Hz (femenino/agudo) con armónicos
        self.voice_b_long = (
            0.5 * np.sin(2 * np.pi * 230.0 * t_1s)
            + 0.3 * np.sin(2 * np.pi * 460.0 * t_1s)
            + 0.15 * np.sin(2 * np.pi * 690.0 * t_1s)
        ).astype(np.float32)

        # Respuesta corta de Voz B (0.5s)
        self.voice_b_short = (
            0.5 * np.sin(2 * np.pi * 230.0 * t_05s)
            + 0.3 * np.sin(2 * np.pi * 460.0 * t_05s)
        ).astype(np.float32)

    def test_a_consecutive_alternation_a_b_a(self):
        """
        TEST A:
        Verifica que en una secuencia A -> B -> A:
        1. El primer turno se asigne a SPEAKER_01.
        2. El segundo turno (Voz B) se reconozca y confirme como SPEAKER_02.
        3. El tercer turno (retorno de Voz A) reasigne correctamente a SPEAKER_01 sin colapsar.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)

        # Turno 1: Habla Persona A
        spk_turn_1 = tracker.register_solo_speech(self.voice_a_long, duration=1.0)
        self.assertEqual(spk_turn_1, "SPEAKER_01")
        self.assertEqual(tracker.last_active_speaker, "SPEAKER_01")

        # Turno 2: Habla Persona B (1.0s de voz diferente)
        spk_turn_2 = tracker.register_solo_speech(self.voice_b_long, duration=1.0)
        self.assertEqual(spk_turn_2, "SPEAKER_02", "Persona B debe asignarse a SPEAKER_02 sin colapsar en SPEAKER_01")
        self.assertEqual(tracker.last_active_speaker, "SPEAKER_02")

        # Turno 3: Vuelve a hablar Persona A
        spk_turn_3 = tracker.register_solo_speech(self.voice_a_long, duration=1.0)
        self.assertEqual(
            spk_turn_3,
            "SPEAKER_01",
            "Persona A al regresar debe recuperar SPEAKER_01 mediante SpeakerMemory (no debe quedarse en SPEAKER_02 ni crear SPEAKER_03)",
        )
        self.assertEqual(tracker.last_active_speaker, "SPEAKER_01")

    def test_b_short_response_attribution(self):
        """
        TEST B:
        Verifica que respuestas cortas (0.5s: "Well, I'm honored.", "Yeah."):
        1. No sean ignoradas ni descartadas.
        2. Se asignen al hablante correcto cuando ya se conocen los perfiles.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)

        # Inicializar ambos perfiles en memoria
        tracker.register_solo_speech(self.voice_a_long, duration=1.0)
        tracker.register_solo_speech(self.voice_b_long, duration=1.0)

        # Ahora Persona A habla
        tracker.register_solo_speech(self.voice_a_long, duration=1.0)
        self.assertEqual(tracker.last_active_speaker, "SPEAKER_01")

        # Persona B interviene con una respuesta corta (0.5s)
        spk_short = tracker.register_solo_speech(self.voice_b_short, duration=0.5)
        self.assertEqual(
            spk_short,
            "SPEAKER_02",
            "La respuesta corta de Persona B debe atribuirse a SPEAKER_02 y no ser absorbida por SPEAKER_01",
        )

    def test_c_syntactic_discontinuity_and_reassembly(self):
        """
        TEST C:
        Verifica que fragmentos sintácticamente incompletos:
        Fragmento 1: 'The people I really admire'
        Fragmento 2: 'are Michael Pollan.'
        sean analizados como continuos por SegmentContinuityAnalyzer y reensamblados
        en una sola oración por SegmentReassembler.
        """
        analyzer = SegmentContinuityAnalyzer()
        reassembler = SegmentReassembler(analyzer=analyzer, debounce_seconds=0.5, max_pause_seconds=1.2)

        frag1 = "The people I really admire"
        frag2 = "are Michael Pollan."

        # Frag 1: No termina en signo terminal y termina con verbo transitivo incompleto
        res1 = reassembler.add_segment(
            text=frag1,
            speaker_id="SPEAKER_01",
            start_time=1.0,
            end_time=2.2,
            confidence=0.92,
        )
        self.assertIsNone(res1, "El fragmento 1 incompleto debe retenerse en el búfer de reensamblado")

        # Frag 2: Completa el predicado y tiene punto final
        res2 = reassembler.add_segment(
            text=frag2,
            speaker_id="SPEAKER_01",
            start_time=2.4,
            end_time=3.3,
            confidence=0.95,
        )
        self.assertIsNotNone(res2, "Al recibir el complemento con punto, debe emitirse la oración reensamblada")
        self.assertEqual(res2.speaker_id, "SPEAKER_01")
        self.assertIn("The people I really admire are Michael Pollan.", res2.text)
        self.assertTrue(res2.is_merged)
        self.assertEqual(res2.start_time, 1.0)
        self.assertEqual(res2.end_time, 3.3)

    def test_d_cross_speaker_non_merging_protection(self):
        """
        TEST D:
        Verifica que el reensamblador NUNCA fusione fragmentos de distintos hablantes,
        incluso si el primero termina con una palabra de enlace o preposición abierta.
        """
        reassembler = SegmentReassembler(debounce_seconds=0.5)

        # Hablante 01 dice algo inconcluso
        frag_spk1 = "and so I was talking to"
        reassembler.add_segment(
            text=frag_spk1,
            speaker_id="SPEAKER_01",
            start_time=0.5,
            end_time=1.5,
            confidence=0.90,
        )

        # Hablante 02 interrumpe inmediatamente después (0.1s después)
        frag_spk2 = "Well, I'm honored."
        emitted_on_spk2 = reassembler.add_segment(
            text=frag_spk2,
            speaker_id="SPEAKER_02",
            start_time=1.6,
            end_time=2.3,
            confidence=0.95,
        )

        # La llegada de SPEAKER_02 debe provocar el flush forzado de lo que tenía SPEAKER_01
        self.assertIsNotNone(emitted_on_spk2)
        self.assertEqual(emitted_on_spk2.speaker_id, "SPEAKER_01")
        self.assertEqual(emitted_on_spk2.text, "and so I was talking to")
        self.assertNotIn("Well", emitted_on_spk2.text, "El fragmento de SPEAKER_02 jamás debe mezclarse con SPEAKER_01")

        # Si vaciamos todos, SPEAKER_02 debe salir de forma totalmente independiente
        remaining = reassembler.flush_all()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].speaker_id, "SPEAKER_02")
        self.assertEqual(remaining[0].text, "Well, I'm honored.")

    def test_e_audio_ring_buffer_pre_post_roll(self):
        """
        TEST E:
        Verifica el comportamiento de AudioRingBuffer para preservación de fonemas:
        - Capacidad y escritura sin errores de desbordamiento.
        - Obtención de audio expandido (get_padded_audio) con pre-roll y post-roll.
        """
        ring = AudioRingBuffer(capacity_seconds=2.0, sample_rate=16000)

        # Escribir 1 segundo de audio ambiente inicial
        ambient = np.zeros(16000, dtype=np.float32)
        ring.write(ambient)

        # Audio de voz
        speech = np.ones(8000, dtype=np.float32) * 0.5  # 0.5s de voz
        ring.write(speech)

        # Padded audio con pre_ms=300 y post_ms=200
        padded = ring.get_padded_audio(current_speech=speech, pre_ms=300, post_ms=200)

        expected_samples = 8000 + int(16000 * 0.3) + int(16000 * 0.2)
        self.assertEqual(len(padded), expected_samples)
        # Verificar que las primeras muestras (pre-roll) provienen del audio previo
        self.assertEqual(padded[0], 0.0)
        # Verificar que el centro contiene las muestras de voz
        self.assertEqual(padded[int(16000 * 0.3) + 100], 0.5)

    def test_f_metrics_der_wer_latency(self):
        """
        TEST F:
        Verifica el cómputo exacto de métricas objetivas:
        1. Diarization Error Rate (DER) detectando confusión de hablantes.
        2. Word Error Rate (WER) y Character Error Rate (CER).
        3. PipelineLatencyTracker registrando tiempos de procesamiento.
        """
        # 1. DER Test
        ref = [(0.0, 2.0, "SPEAKER_01"), (2.5, 4.5, "SPEAKER_02")]
        # Hyp con confusión parcial en el segundo segmento
        hyp = [(0.0, 2.0, "SPEAKER_01"), (2.5, 3.5, "SPEAKER_01"), (3.5, 4.5, "SPEAKER_02")]
        der_res = compute_der(ref, hyp)
        self.assertGreater(der_res.speaker_confusion_time, 0.0)
        self.assertGreater(der_res.der_percentage, 0.0)
        self.assertEqual(der_res.total_speech_time, 4.0)

        # 2. WER & CER Test
        ref_text = "I really admire Michael Pollan"
        hyp_text = "I really admire Michael Pollan"
        self.assertEqual(compute_wer(ref_text, hyp_text), 0.0)
        self.assertEqual(compute_cer(ref_text, hyp_text), 0.0)

        hyp_err = "I really admire Michael"
        self.assertGreater(compute_wer(ref_text, hyp_err), 0.0)

        # 3. Latency Tracker
        tracker = PipelineLatencyTracker()
        tracker.record_stage("stt", 0.120)
        tracker.record_stage("stt", 0.140)
        tracker.record_stage("translation", 0.045)
        summary = tracker.get_summary()
        self.assertIn("stt", summary)
        self.assertIn("translation", summary)
        self.assertAlmostEqual(summary["stt"]["avg_ms"], 130.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
