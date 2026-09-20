import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiments as runner


class ExperimentRunnerTest(unittest.TestCase):
    def test_independent_dataset_item_only_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            data = temp / 'yelp2018'
            data.mkdir()
            (data / 'train.txt').write_text('0 0 1\n1 1 2\n2 3\n')
            (data / 'test.txt').write_text('0 2\n1 3\n2 0\n')
            config = dict(datasets=['amazon', 'yelp2018'],
                common=dict(data_path=str(temp)+'/', epoch=1, dim=8,
                            batch_size=2, Ks='[1,2]', save=False),
                models={'rns': {'configs': [{'id': 'base', 'params': {}}]}})
            path, output = temp / 'config.json', temp / 'output'
            path.write_text(json.dumps(config))
            result = self.invoke(path, output, 'item_only', '--dataset', 'yelp2018')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = runner.read_json(output / 'manifest.json')
            self.assertEqual(len(manifest['jobs']), 1)
            self.assertFalse(list(output.rglob('*.ckpt')))
            self.assertTrue(runner.successful(output / 'item_only/rns/yelp2018/base'))

    def invoke(self, config, output, phase, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'run_experiments.py'),
            '--config', str(config), '--output', str(output), '--phase', phase,
            '--cpu', *extra], capture_output=True, text=True, cwd=ROOT)

    def test_manifest_default_count(self):
        manifest = runner.build_manifest(runner.read_json(ROOT / 'experiments.json'), 37)
        self.assertEqual(len(manifest['jobs']), 462)
        self.assertTrue(all(not job['params']['save'] for job in manifest['jobs']))
        for job in manifest['jobs']:
            for key, value in dict(encoder='lightgcn', context_hops=3, batch_size=2048,
                                   lr=.001, dim=64, embedding0=True).items():
                self.assertEqual(job['params'][key], value, (job['id'], key))
            optimizer_reg = job['model'] in ('directau', 'graphau', 'recdcl')
            self.assertEqual(job['params']['l2'], 0. if optimizer_reg else .001)
            self.assertEqual(job['params']['weight_decay'], .001 if optimizer_reg else 0.)
            if job['model'] == 'stabcf':
                self.assertIn(job['params']['alpha'], [20., 21.])
            if job['model'] == 'graphau':
                self.assertEqual(job['params']['graphau_layers'], 4)
        self.assertTrue(all(job['params']['seed'] == 37 for job in manifest['jobs']))

    def test_phases_resume_failure_and_frozen_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            data = temp / 'yelp2018'
            data.mkdir()
            (data / 'train.txt').write_text('0 0 1\n1 1 2\n2 3\n')
            (data / 'test.txt').write_text('0 2\n1 3\n2 0\n')
            config = {
                'datasets': ['yelp2018'],
                'common': dict(data_path=str(temp) + '/', epoch=2, dim=8,
                               batch_size=2, Ks='[1,2]', save=True),
                'models': {
                    'graphau': {'configs': [
                        {'id': 'missing_data', 'params': {'data_path': str(temp / 'late') + '/'}},
                        {'id': 'ready', 'params': {}}]},
                    'simgcl': {'configs': [{'id': 'ready', 'params': {}}]},
                    'mixgcf': {'configs': [{'id': 'ready', 'params': {'n_negs': 4}}]},
                    'stabcf': {'configs': [{'id': 'ready', 'params': {'n_negs': 4}}]}}}
            path, output = temp / 'config.json', temp / 'results'
            path.write_text(json.dumps(config))
            dry = self.invoke(path, output, 'original', '--dry-run')
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse(output.exists())
            first = self.invoke(path, output, 'original', '--seed', '37')
            self.assertEqual(first.returncode, 1, first.stdout + first.stderr)
            missing = output / 'original/graphau/yelp2018/missing_data'
            ready = output / 'original/graphau/yelp2018/ready'
            self.assertEqual(runner.read_json(missing / 'status.json')['status'], 'failed')
            self.assertTrue(runner.successful(ready))
            self.assertTrue(runner.successful(output / 'original/simgcl/yelp2018/ready'))
            blocked = self.invoke(path, output, 'item_only')
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('baseline jobs are incomplete', blocked.stderr)
            late = temp / 'late/yelp2018'
            late.mkdir(parents=True)
            for filename in ('train.txt', 'test.txt'):
                (late / filename).write_text((data / filename).read_text())
            resumed = self.invoke(path, output, 'original')
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertTrue((missing / 'attempt_001/train.log').exists())
            self.assertEqual(runner.read_json(missing / 'attempt_001/status.json')['status'], 'failed')
            self.assertTrue((missing / 'attempt_002/result.json').exists())
            self.assertFalse((ready / 'attempt_002').exists())
            # Second phase must ignore subsequent source-config edits.
            changed = copy.deepcopy(config)
            changed['common']['dim'] = 999
            path.write_text(json.dumps(changed))
            second = self.invoke(path, output, 'item_only')
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            for job in runner.read_json(output / 'manifest.json')['jobs']:
                left = output / 'original' / job['id']
                right = output / 'item_only' / job['id']
                a = runner.read_json(left / runner.read_json(left / 'status.json')['attempt'] / 'config.json')
                b = runner.read_json(right / 'attempt_001/config.json')
                self.assertEqual(b['gnn'], 'x' + a['gnn'])
                for ignored in ('gnn', 'run_dir', 'out_dir'):
                    a.pop(ignored)
                    b.pop(ignored)
                self.assertEqual(a, b)
            again = self.invoke(path, output, 'item_only')
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(again.stdout.count('SKIP'), 5)
            wrong_seed = self.invoke(path, output, 'item_only', '--seed', '38')
            self.assertNotEqual(wrong_seed.returncode, 0)
            self.assertTrue((output / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
