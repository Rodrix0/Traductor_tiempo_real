"""
test_speaker_separation_stability.py
Suite de pruebas para validar la estabilidad de SpeakerTracker,
el validador acústico SeparationValidator y el deduplicador DuplicateTranscriptResolver.
"""

import unittest
import numpy as np

from src.speakers.tracker import SpeakerTracker
from src.speakers.embeddings import SpeakerEmbeddingExtractor
from src.separation.validator import SeparationValidator
from src.speakers.duplicate_resolver import DuplicateTranscriptResolver
from src.speakers.models import SeparatedSource


class SpeakerSeparationStabilityTests(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1.5, int(1.5 * self.sr), endpoint=False, dtype=np.float32)

        # Voz A: 130 Hz con armónicos
        self.v_a = (0.5 * np.sin(2 * np.pi * 130.0 * t) + 0.25 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)
        # Voz B: 240 Hz con armónicos
        self.v_b = (0.5 * np.sin(2 * np.pi * 240.0 * t) + 0.25 * np.sin(2 * np.pi * 480.0 * t)).astype(np.float32)
        # Mezcla
        self.mixed = self.v_a + self.v_b

    def test_exact_duplicate_sources_suppression(self):
        """
        Test 23: Source 0 y Source 1 contienen la misma voz (fuga del mismo speaker).
        SeparationValidator y DuplicateTranscriptResolver deben evitar emitir dos hablantes.
        """
        validator = SeparationValidator(sample_rate=self.sr)
        resolver = DuplicateTranscriptResolver()

        # Simular dos salidas casi idénticas con leve diferencia de amplitud (fuga espectral)
        s0 = self.v_a
        s1 = 0.85 * self.v_a + 0.05 * np.random.normal(0, 0.01, len(s0)).astype(np.float32)

        val_res = validator.validate(s0, s1)
        self.assertFalse(val_res.is_valid_two_speakers)
        self.assertIn("DUPLICATED_DOMINANT_SPEAKER", val_res.reason)

        # Post-ASR: Si ambas produjeran texto similar
        raw_transcripts = [
            {"text": "I love working on myself", "speaker_id": "SPEAKER_01", "start_time": 10.0, "end_time": 12.0, "confidence": 0.90},
            {"text": "I love working on myself", "speaker_id": "SPEAKER_02", "start_time": 10.0, "end_time": 12.0, "confidence": 0.85},
        ]
        resolved = resolver.resolve(raw_transcripts)
        self.assertEqual(len(resolved), 1, "Debe suprimir la transcripción duplicada y emitir 1 sola")
        self.assertEqual(resolved[0]["speaker_id"], "SPEAKER_01")
        self.assertEqual(resolver.suppressed_duplicates_count, 1)

    def test_partial_duplicate_transcript_resolution(self):
        """
        Test 24: Un canal tiene frase parcial y el otro la frase completa.
        Debe resolver a una sola intervención conservando la más completa.
        """
        resolver = DuplicateTranscriptResolver()

        item_a = {
            "text": "I just find so much comfort and joy",
            "speaker_id": "SPEAKER_01",
            "start_time": 5.0,
            "end_time": 7.5,
            "confidence": 0.85,
        }
        item_b = {
            "text": "I just find so much comfort and joy in actually finding where I could better myself",
            "speaker_id": "SPEAKER_02",
            "start_time": 5.0,
            "end_time": 8.5,
            "confidence": 0.92,
        }

        resolved = resolver.resolve([item_a, item_b])
        self.assertEqual(len(resolved), 1)
        self.assertIn("better myself", resolved[0]["text"])

    def test_real_two_speakers_overlap_preserved(self):
        """
        Test 25: Dos personas reales hablando simultáneamente con textos y voces diferentes.
        Deben conservarse ambas intervenciones independientes.
        """
        validator = SeparationValidator(sample_rate=self.sr)
        resolver = DuplicateTranscriptResolver()

        # Fuentes acústicas claramente diferentes (130 Hz vs 240 Hz)
        val_res = validator.validate(self.v_a, self.v_b)
        self.assertTrue(val_res.is_valid_two_speakers)
        self.assertEqual(val_res.reason, "VALID_TWO_SPEAKERS")

        # Textos completamente diferentes
        transcripts = [
            {"text": "I think we should leave right now.", "speaker_id": "SPEAKER_01", "start_time": 10.0, "end_time": 12.0},
            {"text": "No, wait! Stay here.", "speaker_id": "SPEAKER_02", "start_time": 10.2, "end_time": 11.8},
        ]
        resolved = resolver.resolve(transcripts)
        self.assertEqual(len(resolved), 2, "Deben conservarse los dos hablantes reales")
        self.assertEqual(resolved[0]["speaker_id"], "SPEAKER_01")
        self.assertEqual(resolved[1]["speaker_id"], "SPEAKER_02")

    def test_speaker_continuity_across_consecutive_segments(self):
        """
        Test 26: El mismo hablante a lo largo de 6 segmentos consecutivos
        con pequeñas variaciones de volumen/entonación debe mantenerse establemente en SPEAKER_01.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)

        # Generar 6 variaciones de la voz A
        variations = [
            self.v_a * 1.0,
            self.v_a * 0.85 + 0.05 * np.sin(2 * np.pi * 390.0 * np.linspace(0, 1.5, int(1.5 * self.sr))),
            self.v_a * 1.15,
            self.v_a * 0.90,
            self.v_a * 1.05,
            self.v_a * 0.95,
        ]

        assigned_ids = []
        for v in variations:
            spk = tracker.register_solo_speech(v.astype(np.float32), duration=1.5, is_clean=True)
            assigned_ids.append(spk)

        # Comprobar que en los 6 turnos se mantuvo en SPEAKER_01 sin oscilar
        self.assertEqual(assigned_ids, ["SPEAKER_01"] * 6, f"Se esperaba continuidad total, obtenido: {assigned_ids}")
        self.assertEqual(len(tracker.profiles), 1)

    def test_real_speaker_turn_taking(self):
        """
        Test 27: Diálogo por turnos reales: A, A, A, B, B, B, A.
        Debe asignar SPEAKER_01 y SPEAKER_02 de manera correcta y estable.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)

        sequence = [
            ("A", self.v_a),
            ("A", self.v_a * 0.95),
            ("A", self.v_a * 1.05),
            ("B", self.v_b),
            ("B", self.v_b * 0.92),
            ("B", self.v_b * 1.08),
            ("A", self.v_a),
        ]

        results = []
        for name, audio in sequence:
            spk = tracker.register_solo_speech(audio.astype(np.float32), duration=1.5, is_clean=True)
            results.append(spk)

        expected = [
            "SPEAKER_01",
            "SPEAKER_01",
            "SPEAKER_01",
            "SPEAKER_02",
            "SPEAKER_02",
            "SPEAKER_02",
            "SPEAKER_01",
        ]
        self.assertEqual(results, expected, f"Asignación incorrecta en turnos: {results}")

    def test_no_embedding_contamination_on_dirty_overlap(self):
        """
        Test 28: A habla sola -> se crea perfil A.
        Luego audio mezclado ruidoso/solapado (A+B) no debe contaminar el perfil de SPEAKER_01.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)

        # 1. Registrar voz limpia de A
        spk_a = tracker.register_solo_speech(self.v_a, duration=1.5, is_clean=True)
        self.assertEqual(spk_a, "SPEAKER_01")
        initial_embedding = tracker.profiles["SPEAKER_01"].embedding.copy()

        # 2. Llegada de segmento solapado o sucio (is_clean=False)
        dirty_audio = self.mixed
        tracker.register_solo_speech(dirty_audio, duration=1.5, is_clean=False)

        # 3. Comprobar que el embedding del perfil permanece idéntico (no contaminado)
        current_embedding = tracker.profiles["SPEAKER_01"].embedding
        self.assertTrue(np.allclose(initial_embedding, current_embedding), "El embedding fue contaminado por audio solapado")

    def test_short_interjection_handling(self):
        """
        Verifica que una interjección muy corta (<0.40s) como 'yeah' no cree un nuevo perfil de hablante.
        """
        tracker = SpeakerTracker(sample_rate=self.sr)
        tracker.register_solo_speech(self.v_a, duration=1.5, is_clean=True)
        self.assertEqual(len(tracker.profiles), 1)

        # Interjección de 200 ms
        short_audio = self.v_b[: int(0.20 * self.sr)]
        spk = tracker.register_solo_speech(short_audio, duration=0.20, is_clean=False)

        # No debe haber creado SPEAKER_02 para una interjección tan breve
        self.assertEqual(len(tracker.profiles), 1)
        self.assertEqual(spk, "SPEAKER_01")

    def test_separation_validator_acoustic_metrics(self):
        """
        Prueba los umbrales específicos de SeparationValidator:
        - Ratio de energía bajo -> WEAK_RESIDUAL_LEAKAGE
        - Correlación alta -> DUPLICATED_DOMINANT_SPEAKER
        - Señales distintas -> VALID_TWO_SPEAKERS
        """
        validator = SeparationValidator(sample_rate=self.sr)

        # 1. Desbalance extremo de energía (fuga)
        res_leak = validator.validate(self.v_a, self.v_a * 0.02)
        self.assertFalse(res_leak.is_valid_two_speakers)
        self.assertEqual(res_leak.reason, "WEAK_RESIDUAL_LEAKAGE")

        # 2. Correlación idéntica
        res_corr = validator.validate(self.v_a, self.v_a * 0.9)
        self.assertFalse(res_corr.is_valid_two_speakers)
        self.assertIn("DUPLICATED", res_corr.reason)

        # 3. Señales independientes
        res_valid = validator.validate(self.v_a, self.v_b)
        self.assertTrue(res_valid.is_valid_two_speakers)
        self.assertEqual(res_valid.reason, "VALID_TWO_SPEAKERS")


    def test_real_world_user_reported_duplicate_cases(self):
        """
        Prueba los casos reales exactos reportados por el usuario:
        Caso 1: 'I love working on myself. Like, it is actually like a passion of mine...'
        Caso 2: 'Even if it\'s physical, like the physicalism m' vs '...physically as in my physical health...'
        """
        resolver = DuplicateTranscriptResolver()

        # Caso 1: Frase idéntica duplicada en ambos canales
        raw_1 = [
            {
                "text": "I love working on myself. Like, it is actually like a passion of mine...",
                "speaker_id": "SPEAKER_01",
                "start_time": 12.0,
                "end_time": 15.0,
                "confidence": 0.88,
            },
            {
                "text": "I love working on myself. Like, it is actually like a passion of mine...",
                "speaker_id": "SPEAKER_02",
                "start_time": 12.05,
                "end_time": 14.95,
                "confidence": 0.86,
            },
        ]
        res_1 = resolver.resolve(raw_1)
        self.assertEqual(len(res_1), 1, "Caso 1 debe producir exactamente 1 intervención")
        self.assertIn("passion of mine", res_1[0]["text"])

        # Caso 2: Canal B tiene versión truncada/garble 'm', Canal A tiene versión completa
        raw_2 = [
            {
                "text": "Even if it's physical, like the physicalism m",
                "speaker_id": "SPEAKER_01",
                "start_time": 20.0,
                "end_time": 22.5,
                "confidence": 0.80,
            },
            {
                "text": "Even if it's physical, like physically as in my physical health...",
                "speaker_id": "SPEAKER_02",
                "start_time": 20.0,
                "end_time": 23.0,
                "confidence": 0.85,
            },
        ]
        res_2 = resolver.resolve(raw_2)
        self.assertEqual(len(res_2), 1, "Caso 2 debe producir exactamente 1 intervención")
        self.assertEqual(res_2[0]["text"], "Even if it's physical, like physically as in my physical health...")
        self.assertEqual(res_2[0]["speaker_id"], "SPEAKER_02")

    def test_all_recent_modules_have_valid_logger_and_execute_logging(self):
        """
        Test de regresión para auditoría de logging:
        Verifica que todos los módulos nuevos o modificados tengan su logger oficial
        configurado sin variables globales indefinidas y que las rutas que emiten
        logs no lancen NameError.
        """
        import logging
        import traductor
        from src.separation.validator import logger as val_logger
        from src.speakers.duplicate_resolver import logger as dup_logger
        from src.speakers.tracker import logger as trk_logger
        from src.speakers.embeddings import logger as emb_logger
        from src.pipeline.workers import logger as wrk_logger

        # 1. Comprobar que cada módulo tiene un logger oficial tipo logging.Logger
        self.assertIsInstance(traductor.logger, logging.Logger)
        self.assertIsInstance(val_logger, logging.Logger)
        self.assertIsInstance(dup_logger, logging.Logger)
        self.assertIsInstance(trk_logger, logging.Logger)
        self.assertIsInstance(emb_logger, logging.Logger)
        self.assertIsInstance(wrk_logger, logging.Logger)

        # 2. Ejecutar operaciones que disparan logs en cada uno de estos módulos
        # a) SeparationValidator emite log en rechazo por fuga
        validator = SeparationValidator(sample_rate=self.sr)
        val_res = validator.validate(self.v_a, self.v_a * 0.95)
        self.assertFalse(val_res.is_valid_two_speakers)

        # b) DuplicateTranscriptResolver emite log al suprimir duplicado
        resolver = DuplicateTranscriptResolver()
        resolved = resolver.resolve([
            {"text": "Sample text", "speaker_id": "SPEAKER_01", "start_time": 1.0, "end_time": 2.0},
            {"text": "Sample text", "speaker_id": "SPEAKER_02", "start_time": 1.0, "end_time": 2.0},
        ])
        self.assertEqual(len(resolved), 1)

        # c) SpeakerTracker emite log al registrar primer hablante
        tracker = SpeakerTracker(sample_rate=self.sr)
        spk = tracker.register_solo_speech(self.v_a, duration=1.5, is_clean=True)
        self.assertEqual(spk, "SPEAKER_01")


if __name__ == "__main__":
    unittest.main()
