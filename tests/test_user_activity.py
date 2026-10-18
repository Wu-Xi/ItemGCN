import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.user_activity import partition_users, metrics_from_hits, paired_gains, evaluate_users, aggregate
from utils.checkpoint import read_metadata, sha256


class ActivityTest(unittest.TestCase):
    def test_partition_uses_unique_training_degree_and_id_ties(self):
        ids = np.array([8, 3, 1, 7, 2, 9, 4, 6, 5])
        train = {1: [0, 0], 2: [0], 3: [0], 4: [0, 1], 5: [0, 1],
                 6: [0, 1, 2], 7: [0, 1, 2], 8: [0, 1, 2], 9: [0, 1, 2, 3]}
        groups, degrees, stats = partition_users(ids, train)
        by_id = dict(zip(ids, groups))
        self.assertEqual(by_id, {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4})
        self.assertEqual([s['users'] for s in stats], [3, 2, 2, 2])
        self.assertEqual(degrees[list(ids).index(1)], 1)
        reverse, _, _ = partition_users(ids[::-1], train)
        np.testing.assert_array_equal(reverse[::-1], groups)

    def test_hand_calculated_ranking_metrics(self):
        metrics = metrics_from_hits([[1, 0, 1], [0, 1, 0]], [2, 1], [1, 3])
        np.testing.assert_allclose(metrics['recall'], [[.5, 1], [0, 1]])
        np.testing.assert_allclose(metrics['hit_ratio'], [[1, 1], [0, 1]])
        np.testing.assert_allclose(metrics['ndcg'], [[1, 1.5 / (1 + 1 / np.log2(3))],
                                                   [0, 1 / np.log2(3)]])
        with self.assertRaises(ValueError):
            metrics_from_hits([[False]], [0], [1])

    def test_equal_width_inclusive_integer_ranges_and_ties(self):
        ids = np.arange(9)
        train = {u: list(range(d)) for u, d in enumerate([2, 3, 4, 4, 5, 6, 7, 8, 11])}
        groups, _, stats = partition_users(ids, train, method='equal_width')
        # Inclusive span 2..11: ranges 2..4, 5..6, 7..9, 10..11.
        np.testing.assert_array_equal(groups, [1, 1, 1, 1, 2, 2, 3, 3, 4])
        self.assertEqual([s['degree_range'] for s in stats], ['2–4', '5–6', '7–9', '10–11'])
        reverse, _, _ = partition_users(ids[::-1], train, method='equal_width')
        np.testing.assert_array_equal(reverse[::-1], groups)

    def test_equal_width_degenerate_and_short_ranges(self):
        for degrees, expected in [([7, 7, 7, 7], [1, 1, 1, 1]),
                                  ([0, 0, 1, 1], [1, 1, 3, 3]),
                                  ([0, 1, 2, 2], [1, 2, 3, 3])]:
            with self.subTest(degrees=degrees):
                groups, _, stats = partition_users(np.arange(4),
                    {u: list(range(d)) for u, d in enumerate(degrees)}, method='equal_width')
                np.testing.assert_array_equal(groups, expected)
                for s in stats:
                    if s['users'] == 0:
                        self.assertIsNone(s['degree_min'])
                        self.assertIsNone(s['degree_max'])
                        self.assertIsNone(s['degree_mean'])

    def test_fixed_boundaries_include_cutoffs_in_lower_group(self):
        degrees = [0, 5, 6, 20, 21, 50, 51, 100]
        groups, _, stats = partition_users(np.arange(8),
            {u: list(range(d)) for u, d in enumerate(degrees)}, method='fixed', thresholds=[5, 20, 50])
        np.testing.assert_array_equal(groups, [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([s['degree_range'] for s in stats], ['≤5', '6–20', '21–50', '≥51'])

    def test_reject_invalid_grouping_inputs(self):
        for thresholds in ([5, 5, 20], [-1, 5, 20], [5, 20], [5.5, 20, 50], [True, 20, 50]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                partition_users(np.arange(4), {}, method='fixed', thresholds=thresholds)
        with self.assertRaises(ValueError):
            partition_users(np.arange(4), {}, method='unknown')

    def test_empty_group_metrics_and_total_contribution(self):
        groups = np.array([2, 2, 3, 3])
        original = {m: np.full((4, 2), .4) for m in ('recall', 'ndcg', 'hit_ratio')}
        changed = {m: values + .1 for m, values in original.items()}
        aggregated = aggregate(original, groups)
        self.assertEqual(aggregated['recall']['1'], [None, None])
        self.assertEqual(aggregated['recall']['4'], [None, None])
        rows = paired_gains(original, changed, groups, [10, 20])
        json.dumps(dict(metrics=aggregated, rows=rows), allow_nan=False)
        for metric in original:
            for k in [10, 20]:
                selected = [r for r in rows if r['metric'] == metric and r['k'] == k]
                self.assertAlmostEqual(sum(r['overall_gain_contribution'] for r in selected), .1)
                self.assertIsNone(selected[0]['absolute_gain'])
                self.assertEqual(selected[0]['overall_gain_contribution'], 0)

    def test_weighted_gains_and_zero_denominator(self):
        groups = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4])
        original = {m: np.array([0, 0, 0, .2, .2, .4, .4, .5, .5])[:, None]
                    for m in ('recall', 'ndcg', 'hit_ratio')}
        changed = {m: a + np.array([.1, .1, .1, 0, 0, -.1, -.1, .2, .2])[:, None]
                   for m, a in original.items()}
        rows = paired_gains(original, changed, groups, [20])
        recall = [r for r in rows if r['metric'] == 'recall']
        self.assertIsNone(recall[0]['relative_gain_pct'])
        self.assertAlmostEqual(recall[3]['relative_gain_pct'], 40.)
        self.assertAlmostEqual(sum(r['overall_gain_contribution'] for r in recall),
                               (changed['recall'] - original['recall']).mean())
        self.assertLess(recall[2]['absolute_gain'], 0)

    def test_per_user_evaluator_matches_existing_overall_evaluator(self):
        import torch
        import scipy.sparse as sp
        from utils.evaluate import test as legacy_test

        class FixedScores(torch.nn.Module):
            def generate(self):
                return torch.eye(4), torch.eye(7)

            def rating(self, users, items):
                scores = torch.tensor([[7., 1., 6., 2., 5., 3., 4.],
                                       [1., 7., 2., 6., 3., 5., 4.],
                                       [6., 5., 7., 4., 3., 2., 1.],
                                       [1., 2., 3., 7., 4., 5., 6.]])
                return users @ scores

        train_users = {u: [u] for u in range(4)}
        positives = {0: [2, 6], 1: [3], 2: [0, 1], 3: [4, 5]}
        swaps, reverse = np.empty(4, dtype=object), np.empty(4, dtype=object)
        for u, items in positives.items():
            swaps[u] = torch.FloatTensor(sorted(set(range(len(items))) ^ set(items)))
            reverse[u] = swaps[u].flip(0)
        pre = swaps, reverse, [len(v) for v in positives.values()]
        users = dict(train_user_set=train_users, test_user_set=positives)
        matrices = dict(train_sp_mat=sp.csr_matrix((np.ones(4), (range(4), range(4))), shape=(4, 7)))
        sizes = dict(n_users=4, n_items=7)
        data = None, users, matrices, sizes, None, None, None, pre, None
        model = FixedScores()
        ids, actual = evaluate_users(model, data, [1, 3], 2)
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        expected = legacy_test(model, users, matrices, sizes, None, pre,
                               evaluation_args=SimpleNamespace(cuda=False, test_batch_size=2, Ks='[1,3]'))
        for name, values in actual.items():
            np.testing.assert_allclose(values.mean(axis=0), expected[name], rtol=0, atol=1e-15)
        _, different_batch = evaluate_users(model, data, [1, 3], 3)
        for name in actual:
            np.testing.assert_array_equal(actual[name], different_batch[name])

    def test_fp16_manifest_requires_explicit_opt_in_and_matching_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'model_.ckpt'
            checkpoint.write_bytes(b'converted test placeholder')
            metadata = dict(format_version=1, checkpoint_sha256='old_fp32_hash')
            (root / 'checkpoint.json').write_text(json.dumps(metadata), encoding='utf-8')
            manifest = root / 'FP16_CONVERSION_MANIFEST.json'
            content = dict(files=[dict(path='model_.ckpt', fp16_bytes=checkpoint.stat().st_size)])
            manifest.write_text(json.dumps(content), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                read_metadata(checkpoint)
            _, result = read_metadata(checkpoint, fp16_manifest=manifest)
            self.assertEqual(result['evaluation_provenance']['actual_checkpoint_sha256'], sha256(checkpoint))
            self.assertEqual(result['checkpoint_sha256'], 'old_fp32_hash')
            self.assertNotIn('evaluation_provenance', json.loads((root / 'checkpoint.json').read_text()))
            content['files'][0]['fp16_bytes'] += 1
            manifest.write_text(json.dumps(content), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'path/size mismatch'):
                read_metadata(checkpoint, fp16_manifest=manifest)

    def test_summary_pairs_identical_users_and_labels_all_negative_results(self):
        from evaluate_user_activity import summarize
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
            for phase in ('original', 'item_only'):
                directory = root / 'tiny' / 'rns' / phase
                directory.mkdir(parents=True)
                values = np.full((8, 1), .5 if phase == 'original' else .4)
                np.savez_compressed(directory / 'users.npz', user_ids=np.arange(8), groups=groups,
                                    degrees=np.arange(8), recall=values, ndcg=values, hit_ratio=values)
                result = dict(users_npz_sha256=sha256(directory / 'users.npz'), review_required=False,
                              data_files={}, ks=[20], group_method='equal_freq', degree_thresholds=None,
                              groups=[dict(degree_min=g, degree_max=g + 1, degree_range=f'{g}–{g+1}') for g in range(4)])
                (directory / 'evaluation.json').write_text(json.dumps(result), encoding='utf-8')
            opts = SimpleNamespace(output_root=root, ks=[20], plot=False,
                                   group_method='equal_freq', degree_thresholds=[5, 20, 50])
            summarize(opts, [dict(dataset='tiny', model='rns')])
            summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(summary['rows']), 12)
            self.assertEqual(summary['comparisons'][0]['largest_absolute_gain_groups'], [1, 2, 3, 4])
            self.assertFalse(summary['comparisons'][0]['any_positive_gain'])
            self.assertIn('no positive gain', (root / 'REPORT.md').read_text(encoding='utf-8'))
            self.assertTrue((root / 'group_gains.csv').exists())
            (root / 'tiny/rns/item_only/users.npz').write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                summarize(opts, [dict(dataset='tiny', model='rns')])

    def test_summary_all_methods_and_empty_endpoints(self):
        from evaluate_user_activity import summarize
        for method in ('equal_freq', 'equal_width', 'fixed'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                groups, degrees, stats = partition_users(np.arange(8),
                    {u: list(range(10 if u < 4 else 30)) for u in range(8)}, method=method)
                for phase in ('original', 'item_only'):
                    directory = root / 'tiny/rns' / phase
                    directory.mkdir(parents=True)
                    values = np.full((8, 1), .5 if phase == 'original' else .4)
                    np.savez_compressed(directory / 'users.npz', user_ids=np.arange(8), groups=groups,
                        degrees=degrees, recall=values, ndcg=values, hit_ratio=values)
                    result = dict(users_npz_sha256=sha256(directory / 'users.npz'), review_required=False,
                        data_files={}, groups=stats, ks=[20], group_method=method,
                        degree_thresholds=[5, 20, 50] if method == 'fixed' else None)
                    (directory / 'evaluation.json').write_text(json.dumps(result), encoding='utf-8')
                opts = SimpleNamespace(output_root=root, ks=[20], plot=False,
                    group_method=method, degree_thresholds=[5, 20, 50])
                summarize(opts, [dict(dataset='tiny', model='rns')])
                summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['group_method'], method)
                self.assertAlmostEqual(summary['comparisons'][0]['overall_gain'], -.1)
                self.assertEqual(len(summary['rows']), 12)
                if method == 'fixed':
                    self.assertIsNone(summary['comparisons'][0]['g1_minus_g4_absolute_gain'])
                    self.assertIn('N/A (empty group)', (root / 'REPORT.md').read_text(encoding='utf-8'))
                    self.assertEqual(summary['comparisons'][0]['largest_absolute_gain_groups'], [2, 3])
                opts.group_method = 'fixed' if method != 'fixed' else 'equal_freq'
                with self.assertRaisesRegex(ValueError, 'grouping/cutoffs differ'):
                    summarize(opts, [dict(dataset='tiny', model='rns')])


if __name__ == '__main__':
    unittest.main()
