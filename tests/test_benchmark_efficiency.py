import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_efficiency as bench


def make_fixture(root, models=None):
    folder = root / 'yelp2018'
    folder.mkdir()
    (folder / 'train.txt').write_text('0 0 1\n1 1 2\n2 3\n', encoding='utf-8')
    (folder / 'test.txt').write_text('0 2\n1 3\n2 0\n', encoding='utf-8')
    frozen = bench.runner.read_json(ROOT / 'best54_configs.json')
    jobs = [copy.deepcopy(j) for j in frozen['jobs']
            if j['dataset'] == 'yelp2018' and (models is None or j['model'] in models)]
    for job in jobs:
        job['params'].update(dim=8, batch_size=2, n_negs=4, window_length=2, Ks='[1,2]')
        # No historical checkpoint or logs are provided to the benchmark.
        job['source'] = 'not-present/result.json'
    config = dict(jobs=jobs, data_files={'yelp2018': bench.data_fingerprints(
        SimpleNamespace(data_path=str(root), dataset='yelp2018'))})
    path = root / 'selected.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return path, jobs


class BenchmarkTest(unittest.TestCase):
    def invoke(self, config=None, data=None, output=None, extra=(), expected=0, cpu=True):
        command = [sys.executable, str(ROOT / 'benchmark_efficiency.py')]
        for option, path in (('--config', config), ('--data-path', data), ('--output-root', output)):
            if path is not None:
                command.extend([option, str(path)])
        if cpu:
            command.append('--cpu')
        command.extend(extra)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
        logs = ''
        if output is not None and process.returncode != expected:
            logs = '\n'.join(p.read_text(encoding='utf-8') for p in output.glob('*/*/*/attempt_*/train.log'))
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr + logs)
        return process

    def test_real54_plan_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'benchmark'
            result = self.invoke(output=output, extra=['--dry-run'])
            self.assertIn('54 jobs', result.stdout)
            self.assertIn('1 warm-up + 3 measured', result.stdout)
            self.assertLess(result.stdout.index('ali/rns/original'), result.stdout.index('ali/rns/item_only'))
            self.assertFalse(output.exists())
            config, _ = make_fixture(root)
            self.invoke(config, root, output, ['--warmup-epochs', '0'], expected=2)
            self.invoke(config, root, output, ['--measure-epochs', '0'], expected=2)
            (root / 'yelp2018/test.txt').write_text('0 3\n1 0\n2 1\n', encoding='utf-8')
            result = self.invoke(config, root, output, ['--dry-run'], expected=2)
            self.assertIn('Data files differ', result.stderr)
            self.assertFalse(output.exists())

    def test_all18_from_scratch_resume_and_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, jobs = make_fixture(root)
            output = root / 'benchmark'
            result = self.invoke(config, root, output, ['--measure-epochs', '2'])
            self.assertEqual(result.stdout.count('SUCCESS'), 18)
            counts = {}
            for job in jobs:
                folder = output / 'yelp2018' / job['model'] / job['phase']
                saved = bench.runner.read_json(folder / 'attempt_001/result.json')
                counts[(job['model'], job['phase'])] = saved['trainable_params']
                self.assertIsNone(saved['peak_allocated_gib'])
                self.assertIsNone(saved['peak_reserved_gib'])
                self.assertEqual(saved['initialization'], 'random')
                self.assertFalse(saved['config']['save'])
                self.assertEqual([e['measured'] for e in saved['epoch_records']], [False, True, True])
                self.assertTrue(all(e['batches'] == 3 and e['interactions'] == 5 for e in saved['epoch_records']))
                measured = [e['seconds'] for e in saved['epoch_records'] if e['measured']]
                self.assertAlmostEqual(saved['time_epoch_mean_s'], sum(measured) / 2)
                self.assertGreater(saved['time_epoch_mean_s'], 0)
                self.assertGreaterEqual(saved['time_epoch_std_s'], 0)
            for model in bench.MODELS:
                self.assertEqual(counts[(model, 'original')] - counts[(model, 'item_only')], 3 * 8)
            self.assertEqual(counts[('rns', 'original')], (3 + 4) * 8)
            self.assertEqual(counts[('rns', 'item_only')], 4 * 8)
            self.assertGreater(counts[('recdcl', 'original')], counts[('rns', 'original')])
            self.assertFalse(list(output.rglob('*.ckpt')))
            self.assertFalse(list(output.rglob('*.pt')))
            with (output / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(r['status'] == 'success' and r['peak_allocated_gib'] == '' for r in rows))
            self.assertEqual(self.invoke(config, root, output, ['--measure-epochs', '2']).stdout.count('SKIP'), 18)
            changed = self.invoke(config, root, output, ['--measure-epochs', '3'], expected=2)
            self.assertIn('protocol changed', changed.stderr)
            self.assertFalse((output / '.benchmark.lock').exists())
            broken = output / 'yelp2018/rns/item_only'
            (broken / 'attempt_001/result.json').write_text('{}', encoding='utf-8')
            retry = self.invoke(config, root, output, ['--model', 'rns', '--measure-epochs', '2'])
            self.assertEqual(retry.stdout.count('START'), 1)
            self.assertEqual(retry.stdout.count('SKIP'), 1)
            self.assertTrue((broken / 'attempt_002/result.json').exists())
            # RNS epoch losses agree with the actual training entrypoint, despite
            # the benchmark omitting evaluation and checkpoint IO.
            saved = bench.runner.read_json(output / 'yelp2018/rns/original/attempt_001/result.json')
            params = dict(saved['config'], run_dir=str(root / 'reference'), save=False)
            command = [sys.executable, str(ROOT / 'main.py'), *bench.runner.cli(params)]
            reference = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            self.assertEqual(reference.returncode, 0, reference.stdout + reference.stderr)
            with (root / 'reference/metrics.csv').open() as stream:
                reference_rows = list(csv.DictReader(stream))
            self.assertEqual(len(reference_rows), 3)
            for measured, original in zip(saved['epoch_records'], reference_rows):
                self.assertAlmostEqual(measured['loss_sum'], float(original['loss']), places=6)

    def test_gpu_metrics_when_cuda_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA GPU required for real peak-memory and synchronization validation')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = make_fixture(root, models={'rns', 'sgl', 'recdcl'})
            output = root / 'benchmark'
            self.invoke(config, root, output, ['--measure-epochs', '1'], cpu=False)
            for path in output.glob('*/*/*/attempt_001/result.json'):
                result = bench.runner.read_json(path)
                self.assertGreater(result['peak_allocated_bytes'], 0)
                self.assertGreaterEqual(result['peak_reserved_bytes'], result['peak_allocated_bytes'])
                self.assertEqual(result['peak_allocated_gib'], result['peak_allocated_bytes'] / 1024**3)
                self.assertEqual(result['environment']['device'], 'cuda')


if __name__ == '__main__':
    unittest.main()
