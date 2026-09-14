import unittest

from src.diagnostics.ground_truth_evaluator import GroundTruthEvaluator


class GroundTruthCompletionMetricTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = GroundTruthEvaluator()
        self.reference = [
            {"speaker_id": "A", "start": 0.0, "end": 2.0, "text": "I was really nervous coming in here."},
            {"speaker_id": "B", "start": 2.2, "end": 2.8, "text": "Absolutely."},
        ]

    def test_complete_isolated_turns_score_full_coverage_and_zero_merge(self):
        outputs = [
            {"speaker_id": "A", "start": 0.0, "end": 2.0, "text": "I was really nervous coming in here."},
            {"speaker_id": "B", "start": 2.2, "end": 2.8, "text": "Absolutely."},
        ]
        result = self.evaluator.evaluate(self.reference, outputs)
        self.assertEqual(result.source_dialogue_coverage, 100.0)
        self.assertEqual(result.sentence_completeness_rate, 100.0)
        self.assertEqual(result.cross_speaker_merge_rate, 0.0)

    def test_truncated_start_and_end_reduce_completeness_and_coverage(self):
        outputs = [{"speaker_id": "A", "start": 0.4, "end": 1.7, "text": "really nervous coming in"}]
        result = self.evaluator.evaluate(self.reference, outputs)
        self.assertLess(result.source_dialogue_coverage, 100.0)
        self.assertEqual(result.sentence_completeness_rate, 0.0)

    def test_words_from_overlapping_speakers_are_counted_as_merge(self):
        outputs = [{
            "speaker_id": "A", "start": 0.0, "end": 2.8,
            "text": "I was really nervous coming in here. Absolutely.",
        }]
        result = self.evaluator.evaluate(self.reference, outputs)
        self.assertEqual(result.cross_speaker_merge_rate, 100.0)


if __name__ == "__main__":
    unittest.main()
