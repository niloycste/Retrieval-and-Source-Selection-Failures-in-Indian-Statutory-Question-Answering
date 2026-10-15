"""Small deterministic checks independent of the full experiment run."""
import itertools
import unittest

from retrieval_diagnostic_controls import BM25, parse_section, rank, retrieval_metrics, tokenize
from section_filter_act_extension import bootstrap


class SectionControlTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_section('Under Section 30a, see Section 19'), '30A')
        self.assertIsNone(parse_section('Who may apply?'))

    def test_tie_break_and_restricted_order(self):
        scores = BM25([tokenize(t) for t in ['patent right','trade mark','patent right']]).scores(['patent'])
        self.assertEqual(rank(scores), [0,2,1])
        self.assertEqual([i for i in rank(scores) if i in {1,2}], [2,1])

    def test_expected_uniform_rank(self):
        for n in range(1,7):
            rankings = list(itertools.permutations(range(n)))
            observed = sum(0 in r[:5] for r in rankings)/len(rankings)
            self.assertAlmostEqual(observed, min(1,5/n))

    def test_size_only_random_pool(self):
        # Every size-3 ordered sample of five corpus items, top two.
        rankings = list(itertools.permutations(range(5),3))
        self.assertAlmostEqual(sum(0 in r[:2] for r in rankings)/len(rankings), 2/5)

    def test_empty_rankings(self):
        metrics = retrieval_metrics([[],[0,1]],[0,1])
        self.assertEqual(metrics['recall@1'],0)
        self.assertEqual(metrics['recall@5'],50)
        self.assertEqual(metrics['mrr@10'],25)

    def test_bootstrap_no_difference(self):
        rows = [{'group_id':str(i//2),'r5':{'a':i%2,'b':i%2}} for i in range(6)]
        report = bootstrap(rows,'a','b')
        self.assertEqual(report['difference_points'],0)
        self.assertEqual(report['source_cluster_95_interval'],[0,0])


if __name__ == '__main__':
    unittest.main()
