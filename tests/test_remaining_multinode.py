"""Opt-in Bash dry-run integration: no training, no output directories."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_joint_user_representation as joint
import run_experiments as runner


@unittest.skipUnless(os.environ.get('BASH_TEST_EXE'), 'Set BASH_TEST_EXE to run launcher checks')
class RemainingLauncherTest(unittest.TestCase):
    def test_both_nodes_cover_exactly_2100_jobs(self):
        bash = os.environ['BASH_TEST_EXE']
        env = dict(os.environ, PYTHON=Path(sys.executable).as_posix())
        base = runner.read_json(ROOT/'experiments.json')
        expected = set()
        models = ['rns','mixgcf','directau','graphau','simgcl']
        for dataset in ['ali','amazon','yelp2018']:
            config, _ = joint.build_joint_config(base,dataset,models=models)
            expected.update(m+'/'+dataset+'/'+t['id'] for m,s in config['models'].items() for t in s['configs'])
        self.assertEqual(len(expected),2100)
        seen = set()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'unused'
            for rank in (0,1):
                command = [bash,'start_joint_remaining_multinode.sh',str(rank),
                           '0','1','2','3','4','5','--dry-run','--list-jobs','--output-root',output.as_posix()]
                run = subprocess.run(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
                self.assertEqual(run.returncode,0,run.stderr)
                assigned = re.findall(r'Validate (\w+) shard (\d+)/4 -> GPU (\d+)',run.stdout)
                self.assertEqual(assigned,[(d,str(rank*2+i),str(di*2+i))
                                          for di,d in enumerate(['ali','amazon','yelp2018']) for i in (0,1)])
                jobs = [line.split(' -> ')[0] for line in run.stdout.splitlines() if ' -> x' in line]
                self.assertEqual(len(jobs),1050)
                self.assertEqual(len(set(jobs)),1050)
                self.assertFalse(seen & set(jobs))
                seen.update(jobs)
                self.assertEqual(run.stdout.count('DRY RUN: validated 175 jobs'),6)
                self.assertFalse(output.exists())
            self.assertEqual(seen,expected)
            for args in (['2'], ['0','0','1','2'], ['0','0','1','2','3','4','4']):
                run = subprocess.run([bash,'start_joint_remaining_multinode.sh',*args,'--dry-run'],
                                     cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                     universal_newlines=True,timeout=20)
                self.assertEqual(run.returncode,2)


if __name__ == '__main__':
    unittest.main()
