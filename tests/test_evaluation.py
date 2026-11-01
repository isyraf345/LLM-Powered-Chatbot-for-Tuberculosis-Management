import unittest
from app.evaluate_dpr import score_retrieval, validate_labels


class RetrievalMetricsTests(unittest.TestCase):
    def test_misses_are_not_top_five_hits(self):
        self.assertEqual(score_retrieval([{1}, {2}, {3}, {4}, {5}], [99]),
                         {'hit_at_1': 0., 'hit_at_3': 0., 'hit_at_5': 0., 'mrr_at_5': 0.})

    def test_original_rank(self):
        result = score_retrieval([{10}, {11}, {12}], [12])
        self.assertEqual(result['hit_at_1'], 0)
        self.assertEqual(result['hit_at_3'], 1)
        self.assertAlmostEqual(result['mrr_at_5'], 1 / 3)

    def test_multiple_relevant_results(self):
        self.assertEqual(score_retrieval([{1}, {2, 20}, {3}], [20, 3])['mrr_at_5'], .5)

    def test_empty_results(self):
        self.assertEqual(score_retrieval([], [1])['hit_at_5'], 0)

    def test_unreviewed_and_unknown_labels(self):
        for row in ({'id': 'a', 'reviewed': False, 'relevant_passage_ids': [1]},
                    {'id': 'a', 'reviewed': True, 'relevant_passage_ids': [99]}):
            with self.assertRaises(ValueError):
                validate_labels([row], {1})


if __name__ == '__main__':
    unittest.main()
