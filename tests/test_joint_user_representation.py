import contextlib
import copy
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_joint_user_representation as joint
import run_experiments as runner


class JointSweepTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('BASH_TEST_EXE'), 'Set BASH_TEST_EXE for launcher integration test')
    def test_two_nodes_six_gpus_partition_the_complete_sweep(self):
        bash = os.environ['BASH_TEST_EXE']
        env = dict(os.environ, PYTHON=Path(sys.executable).as_posix())
        base = runner.read_json(ROOT/'experiments.json')
        expected = set()
        for dataset in base['datasets']:
            config, _ = joint.build_joint_config(base, dataset, model='recdcl')
            expected.update('recdcl/'+dataset+'/'+t['id'] for t in config['models']['recdcl']['configs'])
        seen = set()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'unused'
            for rank in (0, 1):
                command = [bash, 'start_joint_recdcl_multigpu.sh', '0','1','2','3','4','5',
                           '--num-nodes','2','--node-rank',str(rank),
                           '--output-root',output.as_posix(),'--dry-run','--list-jobs']
                run = subprocess.run(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,universal_newlines=True,timeout=180)
                self.assertEqual(run.returncode,0,run.stderr)
                assigned = re.findall(r'Validate (\w+) shard (\d+)/(\d+) -> GPU (\d+)',run.stdout)
                self.assertEqual(assigned,[(d,str(rank*2+i), '4',str(di*2+i))
                                          for di,d in enumerate(['ali','amazon','yelp2018']) for i in (0,1)])
                jobs = [line.split(' -> ')[0] for line in run.stdout.splitlines() if line.endswith(' -> xrecdcl')]
                self.assertEqual(len(jobs),3402)
                self.assertEqual(len(set(jobs)),3402)
                self.assertFalse(seen & set(jobs))
                seen.update(jobs)
                self.assertEqual(run.stdout.count('DRY RUN: validated 567 jobs'),6)
                self.assertFalse(output.exists())
            self.assertEqual(seen,expected)
            for invalid in (['--num-nodes','0'], ['--num-nodes','2','--node-rank','2']):
                run = subprocess.run([bash,'start_joint_recdcl_multigpu.sh','0','1','2',*invalid,'--dry-run'],
                                     cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                     universal_newlines=True,timeout=20)
                self.assertEqual(run.returncode,2)

    def test_shards_cover_all_configs_once_and_preserve_parameters(self):
        base = runner.read_json(ROOT / 'experiments.json')
        for dataset in base['datasets']:
            for include_sym in (False, True):
                full, _ = joint.build_joint_config(base, dataset, model='recdcl', include_sym=include_sym)
                original = copy.deepcopy(full)
                expected = {t['id']: t for t in full['models']['recdcl']['configs']}
                for n in (4, 5):
                    combined = {}
                    sizes = []
                    for i in range(n):
                        part = joint.shard_config(full, n, i)
                        selected = {t['id']: t for t in part['models']['recdcl']['configs']}
                        self.assertFalse(set(combined) & set(selected))
                        combined.update(selected)
                        sizes.append(len(selected))
                    self.assertEqual(combined, expected)
                    self.assertLessEqual(max(sizes)-min(sizes), 1)
                    if n == 4 and not include_sym:
                        self.assertEqual(sizes, [567]*4)
                self.assertEqual(full, original)
                self.assertEqual(joint.shard_config(full), original)
        for count, index in ((0, 0), (4, -1), (4, 4), (3000, 0)):
            with self.assertRaises(ValueError):
                joint.shard_config(full, count, index)
        # Shard boundaries can cross model boundaries without losing or duplicating jobs.
        full, _ = joint.build_joint_config(base, 'ali', models=['rns', 'mixgcf'])
        combined = set()
        for i in range(5):
            part = joint.shard_config(full, 5, i)
            jobs = {(m,t['id']) for m,s in part['models'].items() for t in s['configs']}
            self.assertFalse(combined & jobs)
            combined.update(jobs)
        self.assertEqual(len(combined), 112)

    def test_sharded_execution_gpu_routing_resume_and_frozen_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'shard_2_of_4'
            args = ['--dataset', 'ali', '--model', 'rns', '--gpu_id', '7',
                    '--num-shards', '4', '--shard-index', '2', '--output', str(output)]
            launches = []
            def fake_training(command, **kwargs):
                params = vars(runner.parse_args(command[3:]))
                launches.append(params)
                self.assertEqual(params['gpu_id'], 7)
                self.assertEqual(params['gnn'], 'xlightgcn')
                metrics = dict(recall=[.1,.2,.3],ndcg=[.1,.2,.3],hit_ratio=[.1,.2,.3])
                runner.write_json(Path(params['run_dir'])/'result.json', dict(
                    status='success',ks=[10,20,50],elapsed_seconds=.1,
                    best=dict(epoch=1,selection_split='validation',validation=metrics,test=metrics)))
                return SimpleNamespace(wait=lambda: 0)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(runner.subprocess,'Popen',side_effect=fake_training):
                self.assertEqual(joint.main(args + ['--dry-run']), 0)
                self.assertFalse(output.exists())
                self.assertEqual(len(launches), 0)
                self.assertEqual(joint.main(args), 0)
                self.assertEqual(len(launches), 7)
                self.assertEqual(joint.main(args), 0)
                self.assertEqual(len(launches), 7)
                manifest = runner.read_json(output/'manifest.json')
                expected = {t['id'] for t in joint.representation_trials()[2::4]}
                self.assertEqual({j['id'].split('__B_')[1] for j in manifest['jobs']}, expected)
                changed = list(args)
                changed[changed.index('--shard-index')+1] = '1'
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    joint.main(changed)
                self.assertEqual(len(launches), 7)

    def test_four_groups_partition_all_jobs_without_overlap(self):
        base = runner.read_json(ROOT / 'experiments.json')
        groups = [(['recdcl'], 2268), (['sgl'], 672),
                  (['stabcf', 'ahns'], 672), (['rns', 'mixgcf', 'directau', 'graphau', 'simgcl'], 700)]
        for dataset in base['datasets']:
            all_config, _ = joint.build_joint_config(base, dataset)
            expected = {(m, t['id']) for m, s in all_config['models'].items() for t in s['configs']}
            combined = set()
            for models, count in groups:
                config, counts = joint.build_joint_config(base, dataset, models=models)
                jobs = {(m, t['id']) for m, s in config['models'].items() for t in s['configs']}
                self.assertEqual(len(jobs), count)
                self.assertEqual(sum(n for _, n in counts.values()), count)
                self.assertFalse(combined & jobs)
                combined.update(jobs)
                reverse_config, _ = joint.build_joint_config(base, dataset, models=list(reversed(models)))
                self.assertEqual(config, reverse_config)
            self.assertEqual(combined, expected)
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            joint.build_joint_config(base, 'ali', models=['rns', 'rns'])
        with self.assertRaisesRegex(ValueError, 'absent'):
            joint.build_joint_config(base, 'ali', models=['missing'])

    def test_multi_model_cli_dry_run_excludes_other_models(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(joint.main(['--dataset', 'amazon', '--gpu_id', '7',
                '--models', 'stabcf', 'ahns', '--output', str(Path(temp) / 'unused'), '--dry-run']), 0)
            self.assertIn('validated 672 jobs', out.getvalue())
            self.assertIn('dataset=amazon, gpu=7', out.getvalue())
            self.assertNotIn('recdcl:', out.getvalue())

    def test_default_full_cartesian_space_and_order(self):
        base = runner.read_json(ROOT / 'experiments.json')
        original = copy.deepcopy(base)
        expected = dict(rns=28, mixgcf=84, stabcf=336, ahns=336, directau=168,
                        graphau=252, simgcl=168, sgl=672, recdcl=2268)
        b_trials = joint.representation_trials()
        self.assertEqual(len(b_trials), 28)
        self.assertNotIn('sym', {t['params']['b_mode'] for t in b_trials})
        degree = [t for t in b_trials if t['params']['b_mode'] != 'weighted_mean']
        self.assertEqual({(t['params']['b_a'], t['params']['b_b']) for t in degree},
                         set(itertools.product([0, .25, .5, .75, 1], repeat=2)) - {(.5, .5)})
        self.assertEqual([t['params']['b_b'] for t in b_trials if t['params']['b_mode'] == 'weighted_mean'],
                         [.25, .5, .75, 1.])
        for dataset in base['datasets']:
            config, counts = joint.build_joint_config(base, dataset)
            self.assertEqual(config['datasets'], [dataset])
            self.assertEqual({m: count[1] for m, count in counts.items()}, expected)
            self.assertEqual(sum(n for _, n in counts.values()), 4312)
            self.assertEqual(config['common'], base['common'])
            for model, spec in base['models'].items():
                expanded = config['models'][model]['configs']
                self.assertEqual(config['models'][model]['shared'], spec.get('shared', {}))
                for i, recipe in enumerate(runner.expand_trials(spec)):
                    block = expanded[28*i:28*(i+1)]
                    self.assertEqual(len(block), 28)
                    for b, trial in zip(b_trials, block):
                        self.assertEqual(trial['id'], recipe['id'] + '__B_' + b['id'])
                        for key, value in recipe['params'].items():
                            self.assertEqual(trial['params'][key], value)
                        for key, value in b['params'].items():
                            self.assertEqual(trial['params'][key], value)
                        self.assertFalse(trial['params']['save'])
        self.assertEqual(base, original)

    def test_include_sym_restores_one_sym_per_base_recipe(self):
        base = runner.read_json(ROOT / 'experiments.json')
        config, counts = joint.build_joint_config(base, 'ali', include_sym=True)
        self.assertEqual(sum(n for _, n in counts.values()), 4466)
        for model, spec in config['models'].items():
            sym = [t for t in spec['configs'] if t['params']['b_mode'] == 'sym']
            self.assertEqual(len(sym), counts[model][0])
            self.assertTrue(all(t['params']['b_a'] == t['params']['b_b'] == .5 for t in sym))
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(joint.main(['--dataset', 'ali', '--model', 'rns', '--include-sym',
                                         '--output', str(Path(temp) / 'unused'), '--dry-run']), 0)
            self.assertIn('validated 29 jobs', out.getvalue())

    def test_dry_run_validates_without_creating_output_or_launching_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'not_created'
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(runner.subprocess, 'Popen') as popen:
                result = joint.main(['--dataset', 'ali', '--model', 'mixgcf', '--gpu_id', '3',
                                     '--output', str(output), '--dry-run', '--list-jobs'])
            self.assertEqual(result, 0)
            popen.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn('validated 84 jobs', out.getvalue())
            self.assertNotIn('__B_sym', out.getvalue())
            self.assertIn('n_negs-16__B_weighted_mean_b1 -> xlightgcn', out.getvalue())

    def test_execution_routes_gpu_and_resumes_only_failed_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            base = dict(datasets=['ali', 'amazon'], common=dict(dim=64, save=True),
                        models={'rns': dict(shared=dict(ns='rns', n_negs=1),
                                           configs=[dict(id='baseline', params={})])})
            source = temp / 'base.json'
            runner.write_json(source, base)
            output = temp / 'ali'
            args = ['--dataset', 'ali', '--gpu_id', '2', '--seed', '37',
                    '--base-config', str(source), '--output', str(output)]
            launches = []
            fail_first = [True]

            def fake_training(command, **kwargs):
                params = vars(runner.parse_args(command[3:]))
                launches.append(params)
                self.assertEqual(params['gnn'], 'xlightgcn')
                self.assertEqual(params['seed'], 37)
                self.assertEqual(params['dataset'], 'ali')
                self.assertTrue(params['cuda'])
                self.assertFalse(params['save'])
                self.assertNotEqual(params['b_mode'], 'sym')
                self.assertEqual(kwargs['cwd'], runner.ROOT)
                directory = Path(params['run_dir'])
                runner.write_json(directory / 'config.json', params)
                if fail_first[0]:
                    fail_first[0] = False
                    return SimpleNamespace(wait=lambda: 1)
                metrics = dict(recall=[.1, .2, .3], ndcg=[.1, .2, .3], hit_ratio=[.1, .2, .3])
                runner.write_json(directory / 'result.json', dict(status='success', ks=[10, 20, 50],
                    best=dict(epoch=1, selection_split='validation', validation=metrics, test=metrics),
                    elapsed_seconds=.1))
                return SimpleNamespace(wait=lambda: 0)

            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(runner.subprocess, 'Popen', side_effect=fake_training):
                self.assertEqual(joint.main(args), 1)
                self.assertEqual(len(launches), 28)
                self.assertTrue(all(p['gpu_id'] == 2 for p in launches))
                # Resume on another GPU: only the failed first configuration is retried.
                resume = list(args)
                resume[resume.index('--gpu_id') + 1] = '4'
                self.assertEqual(joint.main(resume), 0)
                self.assertEqual(len(launches), 29)
                self.assertEqual(launches[-1]['gpu_id'], 4)
                self.assertEqual(joint.main(resume), 0)
                self.assertEqual(len(launches), 29)
            directory = output / 'item_only/rns/ali/baseline__B_sum'
            self.assertEqual(runner.read_json(directory / 'attempt_001/status.json')['status'], 'failed')
            self.assertTrue((directory / 'attempt_002/result.json').exists())
            self.assertFalse((output / '.running.lock').exists())
            self.assertFalse((output / '.joint-config.lock').exists())
            self.assertTrue((output / 'summary.csv').exists())
            self.assertFalse((output / 'original').exists())

            # Changed experiment parameters or seed must not silently reuse old results.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                wrong_seed = list(args)
                wrong_seed[wrong_seed.index('--seed') + 1] = '38'
                with self.assertRaises(SystemExit):
                    joint.main(wrong_seed)
                frozen = (output / 'joint_user_representation_config.json').read_bytes()
                base['common']['dim'] = 128
                runner.write_json(source, base)
                with self.assertRaises(SystemExit):
                    joint.main(args)
                self.assertEqual((output / 'joint_user_representation_config.json').read_bytes(), frozen)

    def test_reject_unknown_dataset_and_base_grid_that_already_sweeps_B(self):
        base = runner.read_json(ROOT / 'experiments.json')
        with self.assertRaisesRegex(ValueError, 'Dataset'):
            joint.build_joint_config(base, 'missing')
        base['models']['mixgcf']['grid']['b_a'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'not B'):
            joint.build_joint_config(base, 'ali')


if __name__ == '__main__':
    unittest.main()
