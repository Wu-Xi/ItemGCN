import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_best_checkpoints as runner
from utils.checkpoint import data_fingerprints


def tiny_data(root):
    data = root / 'yelp2018'
    data.mkdir()
    (data / 'train.txt').write_text('0 0 1\n1 1 2\n2 3\n', encoding='utf-8')
    (data / 'test.txt').write_text('0 2\n1 3\n2 0\n', encoding='utf-8')


class SelectedPlanTest(unittest.TestCase):
    def test_exact_54_and_six_worker_partition(self):
        config = runner.runner.read_json(ROOT / 'best54_configs.json')
        jobs = runner.select_jobs(config)
        self.assertEqual(len(jobs), 54)
        seen = []
        for d in ('ali', 'amazon', 'yelp2018'):
            for phase in ('original', 'item_only'):
                group = runner.select_jobs(config, d, phase)
                self.assertEqual(len(group), 9)
                seen.extend((j['dataset'], j['model'], j['phase']) for j in group)
        self.assertEqual(len(set(seen)), 54)
        sym = [j for j in jobs if j['B'] == 'sym']
        self.assertEqual([(j['dataset'], j['model']) for j in sym], [('amazon', 'mixgcf')])
        shards = [runner.select_jobs(config, num_shards=12, shard_index=i) for i in range(12)]
        self.assertEqual(sum(map(len, shards)), 54)
        self.assertEqual(len({(j['dataset'], j['model'], j['phase']) for s in shards for j in s}), 54)
        self.assertTrue(all(j['params']['save'] and j['params']['seed'] == 2025 for j in jobs))

    def test_reject_duplicates_and_wrong_architecture(self):
        config = runner.runner.read_json(ROOT / 'best54_configs.json')
        config['jobs'].append(copy.deepcopy(config['jobs'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.select_jobs(config)
        config['jobs'].pop()
        config['jobs'][0]['params']['gnn'] = 'xlightgcn'
        with self.assertRaisesRegex(ValueError, 'architecture'):
            runner.select_jobs(config)


class CheckpointRoundTripTest(unittest.TestCase):
    def test_all_18_architectures_and_data_integrity(self):
        import torch
        from utils.parser import parse_args
        from utils.data_loader import load_data
        from utils.model_factory import build_model
        from utils.checkpoint import BestCheckpoint, load_for_evaluation
        from utils.evaluate import test as evaluate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            tiny_data(root)
            for name in ('rns','mixgcf','stabcf','ahns','directau','graphau','simgcl','sgl','recdcl'):
                for prefix in ('', 'x'):
                    with self.subTest(gnn=prefix+name):
                        out = root / (prefix+name)
                        args = parse_args(['--gnn',prefix+name,'--dataset','yelp2018','--data_path',str(root)+'/',
                                           '--cuda','false','--dim','8','--n_negs','2','--window_length','2',
                                           '--batch_size','2','--Ks','[1,2]','--out_dir',str(out),
                                           '--b_mode','weighted_mean','--b_b','0.25'])
                        data = load_data(args)
                        _, users, matrices, sizes, adj, b, vp, tp, _ = data
                        self.assertEqual(users['train_user_set'][0], [0,1])
                        model = build_model(sizes,args,adj,b,matrices['train_sp_mat'])
                        model.train()
                        if hasattr(model, 'on_train_epoch_start'):
                            model.on_train_epoch_start()
                        batch = dict(users=torch.tensor([0,1]),pos_items=torch.tensor([0,1]),
                                     neg_items=torch.tensor([[2,3],[0,3]]),observed_pos_items=torch.tensor([[0,1],[1,2]]))
                        if name == 'rns':
                            batch['neg_items'] = batch['neg_items'][:, :1]
                        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                        loss=model(batch,0)[0]
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward(); optimizer.step(); model.eval()
                        expected=model.generate(False).detach().clone()
                        metrics=evaluate(model,users,matrices,sizes,vp,tp,evaluation_args=args)
                        BestCheckpoint(args,sizes).save(model,3,metrics,metrics,'test')
                        with torch.no_grad():
                            next(model.parameters()).add_(10)
                        restored, saved_args, restored_data, metadata=load_for_evaluation(out,data_path=root)
                        torch.testing.assert_close(restored.generate(False),expected)
                        _, ru, rm, rn, _, _, rvp, rtp, _ = restored_data
                        self.assertEqual(evaluate(restored,ru,rm,rn,rvp,rtp,evaluation_args=saved_args),metrics)
                        self.assertEqual(metadata['best']['epoch'],3)
            (root/'yelp2018/test.txt').write_text('0 3\n1 0\n2 1\n')
            with self.assertRaisesRegex(ValueError, 'Dataset files differ'):
                load_for_evaluation(out,data_path=root)

    def test_real_scheduler_save_reload_and_repair_missing_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); tiny_data(root)
            source=runner.runner.read_json(ROOT/'best54_configs.json')
            jobs=[copy.deepcopy(j) for j in source['jobs'] if j['dataset']=='yelp2018' and j['model']=='rns']
            for job in jobs:
                job['params'].update(epoch=2,dim=8,batch_size=2,Ks='[1,2]')
            config=dict(jobs=jobs,data_files={'yelp2018':data_fingerprints(SimpleNamespace(data_path=str(root),dataset='yelp2018'))})
            cfg=root/'selected.json'; cfg.write_text(json.dumps(config))
            output=root/'models'
            command=[sys.executable,str(ROOT/'run_best_checkpoints.py'),'--config',str(cfg),
                     '--data-path',str(root),'--output-root',str(output),'--cpu']
            def invoke(extra=()):
                p=subprocess.run(command+list(extra),cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                return p
            invoke(['--dry-run']); self.assertFalse(output.exists())
            invoke()
            for job in jobs:
                folder=output/'yelp2018/rns'/job['phase']
                self.assertTrue(runner.complete(folder))
                ckpt=folder/'attempt_001/model_.ckpt'
                evaluated=root/(job['phase']+'.json')
                p=subprocess.run([sys.executable,str(ROOT/'evaluate_checkpoint.py'),'--checkpoint',str(ckpt),
                    '--data-path',str(root),'--cpu','--output',str(evaluated)],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                evaluation=json.loads(evaluated.read_text())
                self.assertEqual(evaluation['test'],evaluation['recorded_best']['test'])
            self.assertEqual(invoke().stdout.count('SKIP'),2)
            broken=output/'yelp2018/rns/item_only'
            (broken/'attempt_001/model_.ckpt').unlink()
            self.assertFalse(runner.complete(broken))
            self.assertEqual(invoke().stdout.count('START'),1)
            self.assertTrue((broken/'attempt_002/model_.ckpt').exists())
            self.assertTrue(runner.complete(broken))


if __name__ == '__main__':
    unittest.main()
