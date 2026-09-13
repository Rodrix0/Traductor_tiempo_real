"""
test_conversational_translation.py
Suite de pruebas y benchmark para traducción conversacional humana,
ensamblado semántico de segmentos cortados, gestión de contexto y subtitulado profesional offline.
"""

import unittest
import numpy as np

from src.pipeline.segment_assembler import (
    SemanticSegmentAssembler,
    is_incomplete_sentence,
    starts_with_continuation,
)
from src.translation.stt_postprocessor import STTPostProcessor
from src.translation.context_manager import ConversationContextManager
from src.translation.context_aware_translator import (
    ContextAwareTranslator,
    SpokenLanguageNormalizer,
)
from src.ui.subtitle_formatter import SubtitleFormatter
from src.speakers.tracker import SpeakerTracker


class ConversationalTranslationBenchmarkTests(unittest.TestCase):
    """
    Evaluación de los 5 casos de prueba de referencia y comportamiento general.
    """

    @classmethod
    def setUpClass(cls):
        cls.context_mgr = ConversationContextManager()
        cls.translator = ContextAwareTranslator(context_manager=cls.context_mgr)
        cls.assembler = SemanticSegmentAssembler()
        cls.formatter = SubtitleFormatter(max_chars_per_line=42, max_lines=2)

    def setUp(self):
        self.context_mgr.clear()

    def test_benchmark_case_1_idiom_pity_party(self):
        """
        TEST 1:
        Input: 'Not as a pity party at all.'
        Esperado: 'No para dar lástima.' o equivalente natural.
        Rechaza expresamente: 'fiesta de lástima'.
        """
        source = "Not as a pity party at all."
        translation = self.translator.translate(source, "en", "es", speaker_id="SPEAKER_01")

        self.assertNotIn(
            "fiesta de lástima",
            translation.lower(),
            f"La traducción no debe ser literal 'fiesta de lástima'. Obtenido: '{translation}'"
        )
        # Debe contener la noción de dar lástima o hacerse la víctima
        has_natural_meaning = (
            "lástima" in translation.lower() or
            "víctima" in translation.lower() or
            "dar pena" in translation.lower()
        )
        self.assertTrue(
            has_natural_meaning,
            f"Se esperaba una traducción con 'dar lástima' o 'víctima'. Obtenido: '{translation}'"
        )

    def test_benchmark_case_2_actions_and_journaling(self):
        """
        TEST 2:
        Input: 'walking outside, taking a few deep breaths and journaling'
        Esperado: 'salir a caminar, respirar profundamente y escribir en un diario'
        Rechaza expresamente: 'haciendo un diario' o 'mirando afuera'.
        """
        source = "walking outside, taking a few deep breaths and journaling"
        translation = self.translator.translate(source, "en", "es", speaker_id="SPEAKER_01")

        self.assertNotIn("haciendo un diario", translation.lower())
        self.assertNotIn("mirando afuera", translation.lower())

        # Debe contener la traducción idiomática de 'journaling'
        has_journal = "diario" in translation.lower()
        has_walking = "caminar" in translation.lower()
        self.assertTrue(has_journal, f"Debe traducir journaling adecuadamente. Obtenido: '{translation}'")
        self.assertTrue(has_walking, f"Debe traducir walking como caminar. Obtenido: '{translation}'")

    def test_benchmark_case_3_metaphor_shines_through(self):
        """
        TEST 3:
        Input: 'That positivity shines through too.'
        Esperado: 'Esa positividad también se nota.' o equivalente natural.
        Rechaza: 'brilla a través'.
        """
        source = "That positivity shines through too."
        translation = self.translator.translate(source, "en", "es", speaker_id="SPEAKER_01")

        self.assertNotIn("brilla a través", translation.lower())
        has_note_or_shine = (
            "nota" in translation.lower() or
            "destaca" in translation.lower() or
            "percibe" in translation.lower()
        )
        self.assertTrue(
            has_note_or_shine,
            f"Se esperaba 'se nota' o 'destaca'. Obtenido: '{translation}'"
        )

    def test_benchmark_case_4_cross_speaker_dialogue_absolutely(self):
        """
        TEST 4:
        Contexto:
          A: 'I want to be helpful more than anything.'
          B: 'Absolutely.'
        Esperado:
          B: 'Totalmente.' / 'Exacto.' / 'Por supuesto.'
        """
        self.translator.translate(
            "I want to be helpful more than anything.",
            "en",
            "es",
            speaker_id="SPEAKER_01",
        )
        translation_b = self.translator.translate(
            "Absolutely.",
            "en",
            "es",
            speaker_id="SPEAKER_02",
        )

        valid_natural_responses = {"totalmente.", "exacto.", "por supuesto.", "sin duda.", "totalmente", "exacto"}
        self.assertIn(
            translation_b.strip().lower(),
            valid_natural_responses,
            f"Respuesta contextual esperada ('Totalmente.' o 'Exacto.'). Obtenido: '{translation_b}'"
        )

    def test_benchmark_case_5_semantic_segment_assembler(self):
        """
        TEST 5:
        Segment 1: 'I want this to be a learning experience'
        Segment 2: 'for everyone'
        Esperado MERGED:
        'I want this to be a learning experience for everyone.'
        """
        assembler = SemanticSegmentAssembler(debounce_seconds=0.5, max_pause_seconds=1.2)

        # Agregar segmento 1 (incompleto, sin puntuación)
        res1 = assembler.add_segment(
            text="I want this to be a learning experience",
            speaker_id="SPEAKER_01",
            start_time=0.0,
            end_time=2.1,
        )
        self.assertIsNone(res1, "El segmento 1 incompleto debe permanecer en el búfer.")

        # Agregar segmento 2 (continuación 'for everyone.')
        res2 = assembler.add_segment(
            text="for everyone.",
            speaker_id="SPEAKER_01",
            start_time=2.3,
            end_time=3.2,
        )
        self.assertIsNotNone(res2, "Debe emitirse la oración ensamblada completa.")
        self.assertEqual(
            res2.text,
            "I want this to be a learning experience for everyone.",
            f"Texto ensamblado inesperado: '{res2.text}'"
        )
        self.assertTrue(res2.is_merged)


class IncompleteSentenceHeuristicTests(unittest.TestCase):
    """Pruebas unitarias de detección sintáctica de oraciones incompletas."""

    def test_incomplete_patterns(self):
        self.assertTrue(is_incomplete_sentence("That's at least how"))
        self.assertTrue(is_incomplete_sentence("I just wanted to"))
        self.assertTrue(is_incomplete_sentence("because I feel like"))
        self.assertTrue(is_incomplete_sentence("I want to"))
        self.assertTrue(is_incomplete_sentence("something like..."))
        self.assertTrue(is_incomplete_sentence("and"))
        self.assertTrue(is_incomplete_sentence("looking for"))
        self.assertTrue(is_incomplete_sentence("she said that"))

    def test_complete_sentences(self):
        self.assertFalse(is_incomplete_sentence("I want to be helpful more than anything."))
        self.assertFalse(is_incomplete_sentence("This is a complete statement with sufficient words and punctuation."))
        self.assertFalse(is_incomplete_sentence("Everything worked out perfectly in the end!"))

    def test_starts_with_continuation(self):
        self.assertTrue(starts_with_continuation("for everyone"))
        self.assertTrue(starts_with_continuation("and that is how it works"))
        self.assertTrue(starts_with_continuation("because they wanted to"))
        self.assertTrue(starts_with_continuation("running down the road"))
        self.assertFalse(starts_with_continuation("The weather is pleasant."))


class STTPostProcessorTests(unittest.TestCase):
    """Pruebas del post-procesador acústico ligero y de repeticiones."""

    def setUp(self):
        self.processor = STTPostProcessor(min_confidence=0.85)

    def test_deduplicate_stutter(self):
        result = self.processor.process("I think the the the person was right.")
        self.assertEqual(result.corrected_text, "I think the person was right.")
        self.assertTrue(result.was_modified)

    def test_contraction_expansion(self):
        result = self.processor.process("I wanna see what is gonna happen.")
        self.assertEqual(result.corrected_text, "I want to see what is going to happen.")
        self.assertTrue(result.was_modified)

    def test_punctuation_cleanup(self):
        result = self.processor.process(", hello world -")
        self.assertEqual(result.corrected_text, "hello world")


class SpokenLanguageNormalizerTests(unittest.TestCase):
    """Pruebas de normalización de muletillas y lenguaje oral."""

    def setUp(self):
        self.normalizer = SpokenLanguageNormalizer()

    def test_hedging_normalization(self):
        text, modified = self.normalizer.normalize("I am kind of just trying to be helpful")
        self.assertEqual(text, "I am just trying to be helpful")
        self.assertTrue(modified)

    def test_leading_fillers(self):
        text, modified = self.normalizer.normalize("Like, you know, this was an amazing event")
        self.assertNotIn("Like,", text)
        self.assertIn("this was an amazing event", text)


class SubtitleFormatterTests(unittest.TestCase):
    """Pruebas de formateo profesional de subtítulos."""

    def setUp(self):
        self.formatter = SubtitleFormatter(max_chars_per_line=42, max_lines=2)

    def test_short_subtitle_single_line(self):
        text = "Totalmente de acuerdo."
        formatted = self.formatter.format(text)
        self.assertEqual(formatted, text)
        self.assertNotIn("\n", formatted)

    def test_long_subtitle_balanced_split(self):
        text = "Normalmente comparto algún consejo o una técnica sencilla para resolver problemas."
        formatted = self.formatter.format(text)
        lines = formatted.split("\n")
        self.assertLessEqual(len(lines), 2)
        for line in lines:
            self.assertLessEqual(len(line), 45)

    def test_clean_redundancy(self):
        text = "técnica técnica sencilla"
        formatted = self.formatter.format(text)
        self.assertNotIn("técnica técnica", formatted)


class SpeakerTrackerContinuityDecayTests(unittest.TestCase):
    """Pruebas del decaimiento del bono de continuidad y suavizado de estabilidad."""

    def setUp(self):
        self.tracker = SpeakerTracker(sample_rate=16000)

    def test_stability_and_continuity(self):
        # Audio sintético para simular voz A (frecuencia 220Hz)
        t = np.linspace(0, 1.0, 16000)
        audio_a = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        # Primer registro
        spk_1 = self.tracker.register_solo_speech(audio_a, duration=1.0)
        self.assertEqual(spk_1, "SPEAKER_01")
        self.assertEqual(self.tracker.last_active_speaker, "SPEAKER_01")

        # Segundo registro con la misma voz: debe mantener identidad y acumular estabilidad
        spk_1_again = self.tracker.register_solo_speech(audio_a, duration=1.0)
        self.assertEqual(spk_1_again, "SPEAKER_01")
        self.assertGreaterEqual(self.tracker.speaker_stability_frames, 1)


if __name__ == "__main__":
    unittest.main()
