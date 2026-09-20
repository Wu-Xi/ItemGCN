import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.LightGCN import SimGCL, XSimGCL, SGL, XSGL, RecDCL, XRecDCL, XGraphAU
from utils.parser import parse_args


class MigratedModelsTest(unittest.TestCase):
    def test_strict_early_stopping(self):
        from utils.helper import early_stopping
        self.assertEqual(early_stopping(.1, .1, 9, flag_step=10), (.1, 10, True))
        self.assertEqual(early_stopping(.100001, .1, 9, flag_step=10), (.100001, 0, False))
        self.assertEqual(early_stopping(0., -float('inf'), 0), (0., 0, False))

    def test_item_only_gradient_and_checkpoint(self):
        r = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1.]]))
        adj = sp.bmat([[None, r], [r.T, None]], format='csr')
        d = sp.diags(np.asarray(adj.sum(1)).ravel() ** -.5)
        adj = d @ adj @ d
        b = sp.diags([.5, .5, 1.]) @ r
        batch = dict(users=torch.tensor([0, 1, 2]), pos_items=torch.tensor([0, 1, 3]),
                     neg_items=torch.tensor([[2], [3], [0]]))
        for cls, name in [(XSimGCL, 'xsimgcl'), (XSGL, 'xsgl'), (XRecDCL, 'xrecdcl'), (XGraphAU, 'xgraphau')]:
            with self.subTest(model=name):
                args = parse_args(['--gnn', name, '--dim', '8', '--cuda', 'false'])
                extra = [r] if name == 'xsgl' else []
                model = cls(dict(n_users=3, n_items=4), args, adj, b, *extra)
                self.assertFalse(any('user_embed' in n for n, _ in model.named_parameters()))
                q = model.item_embedding.weight if name == 'xrecdcl' else model.item_embed
                u = model.build_user_embed()
                torch.testing.assert_close(u, torch.tensor(b.toarray(), dtype=q.dtype) @ q)
                grad = torch.autograd.grad(u.sum(), q)[0]
                torch.testing.assert_close(grad, torch.tensor(b.toarray(), dtype=q.dtype).T @ torch.ones_like(u))
                opt = torch.optim.Adam(model.parameters(), lr=.001)
                for size in (3, 1):
                    opt.zero_grad()
                    loss = model({k: v[:size] for k, v in batch.items()}, 0)[0]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(q.grad).all())
                    opt.step()
                model.eval()
                clone = cls(dict(n_users=3, n_items=4), args, adj, b, *extra)
                clone.load_state_dict(model.state_dict())
                clone.eval()
                torch.testing.assert_close(model.generate(False), clone.generate(False))

    def test_all_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'yelp2018'
            data.mkdir()
            (data / 'train.txt').write_text('0 0 1\n1 1 2\n2 3\n')
            (data / 'test.txt').write_text('0 2\n1 3\n2 0\n')
            for name in ('simgcl', 'xsimgcl', 'sgl', 'xsgl', 'recdcl', 'xrecdcl',
                         'lightgcn', 'xlightgcn', 'igcn', 'ahns', 'xahns',
                         'directau', 'xdirectau', 'graphau', 'xgraphau',
                         'rns', 'xrns', 'mixgcf', 'xmixgcf', 'stabcf', 'xstabcf'):
                for aug in ((0, 1, 2) if name in ('sgl', 'xsgl') else (1,)):
                    with self.subTest(model=name, aug=aug):
                        out = Path(tmp) / (name + str(aug))
                        run = subprocess.run([sys.executable, 'main.py', '--gnn', name,
                            '--data_path', str(Path(tmp)) + '/', '--cuda', 'false',
                            '--epoch', '2', '--dim', '8', '--batch_size', '2', '--Ks', '[1,2]',
                            '--aug_type', str(aug), '--save', 'true', '--out_dir', str(out)],
                            cwd=ROOT, capture_output=True, text=True)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        self.assertTrue((out / 'model_.ckpt').exists())


if __name__ == '__main__':
    unittest.main()
