"""Retrain selected Original/Item-only recipes once and retain best checkpoints."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import run_experiments as runner
from utils.checkpoint import data_fingerprints, read_metadata

ROOT = Path(__file__).resolve().parent


def select_jobs(config, dataset=None, phase='all', num_shards=1, shard_index=0):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Require num_shards >= 1 and 0 <= shard_index < num_shards')
    jobs = config['jobs']
    ids = []
    for job in jobs:
        for key in ('dataset', 'model', 'phase'):
            runner.identifier(job[key])
        if job['phase'] not in ('original', 'item_only'):
            raise ValueError('Unsupported phase: ' + job['phase'])
        if job['params']['dataset'] != job['dataset']:
            raise ValueError('Dataset mismatch in selected config')
        if job['params']['gnn'].startswith('x') != (job['phase'] == 'item_only'):
            raise ValueError('Wrong Original/Item-only architecture')
        ids.append((job['dataset'], job['model'], job['phase']))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate selected jobs')
    eligible = [j for j in jobs if (dataset is None or j['dataset'] == dataset)
                and (phase == 'all' or j['phase'] == phase)]
    if not eligible or num_shards > len(eligible):
        raise ValueError('No selected jobs, or more shards than jobs')
    return eligible[shard_index::num_shards]


def complete(directory):
    """An old success log without its matching weights is not a completed save job."""
    try:
        state = runner.read_json(directory / 'status.json')
        if state['status'] != 'success':
            return False
        attempt_name = runner.identifier(state['attempt'])
        attempt = directory / attempt_name
        result = runner.read_json(attempt / 'result.json')
        _, metadata = read_metadata(attempt / 'model_.ckpt')
        return result['status'] == 'success' and result['best'] == metadata['best']
    except (OSError, ValueError, KeyError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'best54_configs.json')
    parser.add_argument('--dataset', choices=['ali', 'amazon', 'yelp2018'])
    parser.add_argument('--phase', choices=['all', 'original', 'item_only'], default='all')
    parser.add_argument('--gpu-id', '--gpu_id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'checkpoints/best54_20261010')
    parser.add_argument('--data-path', type=Path, default=ROOT / 'data')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    opts = parser.parse_args(argv)
    if opts.gpu_id < 0:
        parser.error('gpu-id must be nonnegative')
    config = runner.read_json(opts.config)
    try:
        jobs = select_jobs(config, opts.dataset, opts.phase, opts.num_shards, opts.shard_index)
    except ValueError as exc:
        parser.error(str(exc))
    output = opts.output_root.resolve()
    digest = runner.code_digest()
    data_path = str(opts.data_path.resolve()) + '/'
    for dataset in {j['dataset'] for j in jobs}:
        observed = data_fingerprints(SimpleNamespace(data_path=data_path, dataset=dataset))
        if observed != config['data_files'][dataset]:
            parser.error('Data files differ from frozen best54 configuration: ' + dataset)
    print(f'{len(jobs)} selected jobs; save=True; output={output}', flush=True)
    plans = []
    for job in jobs:
        params = dict(job['params'], save=True, data_path=data_path)
        # Validate the actual CLI, including all frozen defaults.
        runner.parse_args(runner.cli(params))
        directory = output / job['dataset'] / job['model'] / job['phase']
        frozen = dict(version=1, job=job, params=params, code_digest=digest,
                      data_files=config['data_files'][job['dataset']])
        manifest = directory / 'manifest.json'
        if manifest.exists() and runner.read_json(manifest) != frozen:
            parser.error('Configuration/code changed; use a new output root: ' + str(directory))
        plans.append((job, params, directory, frozen))
        print(f"  {job['dataset']}/{job['model']}/{job['phase']} B={job.get('B') or '-'}", flush=True)
    if opts.dry_run:
        return 0
    failures = 0
    index = []
    for job, params, directory, frozen in plans:
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / '.running.lock'
        try:
            lock.touch(exist_ok=False)
        except FileExistsError:
            parser.error('Job is locked; inspect existing process before removing lock: ' + str(lock))
        try:
            manifest = directory / 'manifest.json'
            if manifest.exists() and runner.read_json(manifest) != frozen:
                raise ValueError('Frozen job changed: ' + str(directory))
            if not manifest.exists():
                runner.write_json(manifest, frozen)
            if complete(directory):
                print('SKIP ' + str(directory), flush=True)
            else:
                number = 1
                while (directory / f'attempt_{number:03d}').exists():
                    number += 1
                attempt = directory / f'attempt_{number:03d}'
                attempt.mkdir()
                runtime = dict(params, cuda=not opts.cpu, gpu_id=opts.gpu_id,
                               run_dir=str(attempt), out_dir=str(attempt))
                command = [sys.executable, '-u', str(ROOT / 'main.py'), *runner.cli(runtime)]
                runner.write_json(attempt / 'command.json', command)
                state = dict(status='running', attempt=attempt.name, started=time.time())
                def save_state():
                    runner.write_json(directory / 'status.json', state)
                    runner.write_json(attempt / 'status.json', state)
                save_state()
                print('START ' + str(attempt), flush=True)
                process = None
                try:
                    with (attempt / 'train.log').open('w', encoding='utf-8') as log:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        code = process.wait()
                    state.update(status='success' if code == 0 else 'failed', returncode=code, ended=time.time())
                    save_state()
                    if code == 0 and not complete(directory):
                        state.update(status='failed', error='Missing, corrupt, or mismatched best checkpoint')
                except KeyboardInterrupt:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.wait()
                    state.update(status='interrupted', ended=time.time())
                    save_state()
                    raise
                except OSError as exc:
                    state.update(status='failed', error=str(exc), ended=time.time())
                save_state()
                print(state['status'].upper() + ' ' + str(attempt), flush=True)
            ok = complete(directory)
            failures += not ok
            state = runner.read_json(directory / 'status.json')
            record = dict(dataset=job['dataset'], model=job['model'], phase=job['phase'], status=state['status'],
                          historical_test_recall20=job['reference_best']['test']['recall'][1])
            if ok:
                ckpt = directory / state['attempt'] / 'model_.ckpt'
                _, metadata = read_metadata(ckpt)
                record.update(checkpoint=str(ckpt.relative_to(output)), best_epoch=metadata['best']['epoch'],
                              test_recall20=metadata['best']['test']['recall'][1])
            index.append(record)
            runner.write_json(output / f"index_{opts.dataset or 'all'}_{opts.phase}_{opts.shard_index}_of_{opts.num_shards}.json", index)
        finally:
            lock.unlink()
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
