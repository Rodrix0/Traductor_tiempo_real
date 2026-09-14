import time
import unittest

import numpy as np

from src.pipeline.models import ConfirmedUtterance
from src.pipeline.sentence_turn_assembler import SentenceTurnAssembler, is_likely_incomplete


class FakeFinalASR:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        words = []
        cursor = 0.28
        for word in self.text.split():
            words.append({"word": word, "start": cursor, "end": cursor + 0.1, "probability": 0.95})
            cursor += 0.11
        return {"text": self.text, "language": "en", "confidence": 0.95, "words": words}


def audio(seconds=0.8, value=0.1):
    return np.full(int(16000 * seconds), value, dtype=np.float32)


def add(assembler, text, speaker="SPEAKER_01", start=0.0, end=0.8,
        pause=0, partial=False, words=None):
    return assembler.add_event(
        speaker_id=speaker,
        text=text,
        word_timestamps=words or [],
        audio=audio(max(0.1, end - start)),
        audio_start=start,
        audio_end=end,
        is_partial=partial,
        confidence=0.9,
        source_language="en",
        sequence_id=int(start * 10) + 1,
        trailing_silence_ms=pause,
    )


class SentenceTurnAssemblerTests(unittest.TestCase):
    def test_1_three_whisper_chunks_become_one_confirmed_utterance(self):
        final = FakeFinalASR("I was really nervous coming in here because this means a lot to me.")
        asm = SentenceTurnAssembler(asr_engine=final)
        self.assertEqual(add(asm, "I was really nervous", end=1.0, pause=80, partial=True), [])
        self.assertEqual(add(asm, "coming in here", start=1.08, end=2.0, pause=100, partial=True), [])
        result = add(asm, "because this means a lot to me.", start=2.1, end=3.4, pause=700)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ConfirmedUtterance)
        self.assertEqual(result[0].source_text, final.text)
        self.assertEqual(result[0].partial_fragments, ["I was really nervous", "coming in here", "because this means a lot to me."])
        self.assertTrue(result[0].final_asr_pass_used)
        self.assertFalse(final.calls[0][1]["vad_filter"])

    def test_2_speaker_change_produces_two_isolated_utterances(self):
        asm = SentenceTurnAssembler()
        out = add(asm, "I've done a lot of interviews in my life.", pause=700)
        out += add(asm, "Absolutely.", "SPEAKER_02", 1.0, 1.5, pause=700)
        self.assertEqual([item.speaker_id for item in out], ["SPEAKER_01", "SPEAKER_02"])
        self.assertEqual([item.source_text for item in out], ["I've done a lot of interviews in my life.", "Absolutely."])

    def test_3_micro_pause_does_not_cut_sentence(self):
        asm = SentenceTurnAssembler()
        self.assertEqual(add(asm, "I think...", end=0.8, pause=200), [])
        out = add(asm, "this is really important.", start=1.0, end=2.0, pause=700)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source_text, "I think... this is really important.")

    def test_4_open_clause_waits_for_complement(self):
        asm = SentenceTurnAssembler()
        self.assertEqual(add(asm, "I was thinking about", end=1.0, pause=600), [])
        out = add(asm, "what you said yesterday.", start=1.6, end=2.7, pause=700)
        self.assertEqual(len(out), 1)
        self.assertIn("thinking about what", out[0].source_text)

    def test_5_interruption_never_merges_speakers(self):
        asm = SentenceTurnAssembler()
        self.assertEqual(add(asm, "I think that's...", pause=100), [])
        out = add(asm, "Exactly.", "SPEAKER_02", 0.9, 1.4, pause=700)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].is_interrupted)
        self.assertEqual(out[0].speaker_id, "SPEAKER_01")
        self.assertEqual(out[1].speaker_id, "SPEAKER_02")
        self.assertNotIn("Exactly", out[0].source_text)

    def test_6_long_silence_splits_same_speaker_sentences(self):
        asm = SentenceTurnAssembler()
        first = add(asm, "Hello.", pause=800)
        second = add(asm, "How are you?", start=1.6, end=2.4, pause=800)
        self.assertEqual([item.source_text for item in first + second], ["Hello.", "How are you?"])

    def test_7_long_turn_splits_internal_sentence_boundaries(self):
        asm = SentenceTurnAssembler()
        words = [
            {"word": "I", "start": 0.0, "end": 0.2}, {"word": "went", "start": 0.2, "end": 0.5}, {"word": "there", "start": 0.5, "end": 0.8}, {"word": "yesterday.", "start": 0.8, "end": 1.1},
            {"word": "It", "start": 1.1, "end": 1.3}, {"word": "was", "start": 1.3, "end": 1.5}, {"word": "interesting.", "start": 1.5, "end": 2.0},
            {"word": "Then", "start": 2.0, "end": 2.2}, {"word": "we", "start": 2.2, "end": 2.4}, {"word": "went", "start": 2.4, "end": 2.6}, {"word": "home.", "start": 2.6, "end": 3.0},
        ]
        out = add(asm, "I went there yesterday. It was interesting. Then we went home.", end=3.2, pause=700, words=words)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(item.speaker_id == "SPEAKER_01" for item in out))

    def test_8_bidirectional_interruptions_have_zero_textual_merge(self):
        asm = SentenceTurnAssembler()
        out = []
        out += add(asm, "A begins because", "SPEAKER_01", 0.0, 0.7, pause=50)
        out += add(asm, "B interrupts.", "SPEAKER_02", 0.75, 1.3, pause=50)
        out += add(asm, "A returns.", "SPEAKER_01", 1.35, 2.0, pause=700)
        out += asm.flush_all()
        self.assertGreaterEqual(len(out), 3)
        for item in out:
            if item.speaker_id == "SPEAKER_01":
                self.assertNotIn("B interrupts", item.source_text)
            else:
                self.assertNotIn("A begins", item.source_text)

    def test_timeout_requires_acoustic_not_wall_clock_silence(self):
        asm = SentenceTurnAssembler()
        add(asm, "This clause is complete", pause=300)
        state = asm.current_turn["SPEAKER_01"]
        self.assertEqual(asm.check_timeouts(state.fragments[-1].received_at + 30), [])
        out = asm.check_timeouts(acoustic_progress=(1.5, 0.5))
        self.assertEqual(len(out), 1)

    def test_partial_survives_slow_asr_and_retains_continuation(self):
        asm = SentenceTurnAssembler()
        add(asm, "I was thinking", partial=True)
        state = asm.current_turn["SPEAKER_01"]
        self.assertEqual(asm.check_timeouts(state.fragments[-1].received_at + 30), [])
        self.assertEqual(asm.check_timeouts(acoustic_progress=(4.0, 4.0)), [])
        out = add(asm, "about what you said.", start=0.8, end=2.0, pause=700)
        self.assertEqual([item.source_text for item in out], ["I was thinking about what you said."])

    def test_resumed_voice_blocks_timeout_while_asr_is_pending(self):
        asm = SentenceTurnAssembler()
        add(asm, "This clause is complete", pause=300)
        self.assertEqual(asm.check_timeouts(acoustic_progress=(5.0, 2.0)), [])

    def test_incomplete_clause_waits_even_after_real_silence(self):
        asm = SentenceTurnAssembler()
        add(asm, "I was thinking about", pause=600)
        self.assertEqual(asm.check_timeouts(acoustic_progress=(4.0, 0.2)), [])
        out = add(asm, "what you said.", start=4.0, end=5.0, pause=700)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].is_complete)

    def test_backlogged_sentences_use_audio_gap(self):
        asm = SentenceTurnAssembler()
        add(asm, "This clause is complete", pause=300)
        out = add(asm, "This is another sentence.", start=2.0, end=3.0, pause=700)
        self.assertEqual([item.source_text for item in out], ["This clause is complete", "This is another sentence."])

    def test_ellipsis_in_word_timestamps_is_not_a_sentence_boundary(self):
        asm = SentenceTurnAssembler()
        words = [{"word": "I", "start": 0.0, "end": 0.1},
                 {"word": "think...", "start": 0.1, "end": 0.4},
                 {"word": "this", "start": 0.5, "end": 0.6},
                 {"word": "matters.", "start": 0.6, "end": 0.9}]
        out = add(asm, "I think... this matters.", end=1.0, pause=700, words=words)
        self.assertEqual([item.source_text for item in out], ["I think... this matters."])
        self.assertEqual(asm.completion_score("I think...", 700, False, 1).punctuation_score, 0.0)

    def test_internal_split_preserves_every_audio_sample_and_word_time(self):
        asm = SentenceTurnAssembler()
        samples = np.arange(32000, dtype=np.float32) / 32000
        words = [{"word": "Hello.", "start": 0.2, "end": 0.6},
                 {"word": "Goodbye.", "start": 1.0, "end": 1.5}]
        out = asm.add_event(speaker_id="A", text="Hello. Goodbye.", word_timestamps=words,
                            audio=samples, audio_start=5.0, audio_end=7.0,
                            is_partial=False, confidence=0.9, trailing_silence_ms=700)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(np.concatenate([item.audio for item in out]), samples)
        self.assertEqual(out[0].start_time, 5.0)
        self.assertEqual(out[-1].end_time, 7.0)
        self.assertEqual(out[0].end_time, out[1].start_time)
        self.assertAlmostEqual(out[1].word_timestamps[0]["start"], 6.0)

    def test_speaker_change_after_internal_split_closes_open_tail(self):
        asm = SentenceTurnAssembler()
        words = [{"word": "Hello.", "start": 0.0, "end": 0.3},
                 {"word": "I", "start": 0.3, "end": 0.4},
                 {"word": "want", "start": 0.4, "end": 0.5},
                 {"word": "to", "start": 0.5, "end": 0.6}]
        out = add(asm, "Hello. I want to", partial=True, words=words)
        out += add(asm, "Absolutely.", speaker="SPEAKER_02", start=0.8, end=1.4, pause=700)
        self.assertEqual([item.source_text for item in out], ["Hello.", "I want to", "Absolutely."])
        self.assertTrue(out[1].is_interrupted)
        self.assertFalse(asm.current_turn)

    def test_incomplete_heuristic_uses_structure(self):
        self.assertTrue(is_likely_incomplete("I want to"))
        self.assertTrue(is_likely_incomplete("The way"))
        self.assertTrue(is_likely_incomplete("I was thinking about"))
        self.assertFalse(is_likely_incomplete("This really matters."))
        self.assertFalse(is_likely_incomplete("This really matters"))


if __name__ == "__main__":
    unittest.main()
