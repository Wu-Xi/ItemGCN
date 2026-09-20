"""Sequential, resumable paired experiments. No seed sweep or implicit second phase."""
import argparse
import copy
import csv
import hashlib
import json
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

from utils.parser import parse_args

ROOT = Path(__file__).resolve().parent
MODELS = ('rns', 'mixgcf', 'stabcf', 'lightgcn', 'directau', 'graphau', 'simgcl', 'sgl', 'recdcl', 'ahns')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def cli(params):
    output = []
    for key, value in params.items():
        if value is not None:
            output.extend(['--' + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return output


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', value):
        raise ValueError('Invalid dataset/config identifier: ' + repr(value))
    return value


def code_digest():
    paths = [ROOT / 'main.py', ROOT / 'run_experiments.py']
    paths += sorted((ROOT / 'modules').glob('*.py')) + sorted((ROOT / 'utils').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(config, seed):
    jobs = []
    reserved = {'gnn', 'dataset', 'seed', 'run_dir', 'out_dir', 'gpu_id', 'cuda'}
    if not config['datasets'] or not config['models']:
        raise ValueError('At least one dataset and model is required')
    for model, specification in config['models'].items():
        if model not in MODELS:
            raise ValueError('Unsupported paired model: ' + model)
        for dataset in config['datasets']:
            identifier(dataset)
            for trial in expand_trials(specification):
                label = identifier(trial['id'])
                params = dict(config.get('common', {}))
                params.update(specification.get('shared', {}))
                params.update(trial['params'])
                if reserved & params.keys():
                    raise ValueError('Reserved parameters in config: ' + str(reserved & params.keys()))
                params.update(gnn=model, dataset=dataset, seed=seed)
                # Freeze all resolved defaults, not only explicitly supplied parameters.
                resolved = vars(parse_args(cli(params)))
                for key in ('run_dir', 'out_dir', 'gpu_id', 'cuda'):
                    resolved.pop(key)
                jobs.append(dict(id=f'{model}/{dataset}/{label}', model=model, params=resolved,
                                 source=trial.get('source', specification.get('source', ''))))
    if not jobs or len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Experiment list is empty or contains duplicate IDs')
    return dict(version=1, seed=seed, code_digest=code_digest(), config=config, jobs=jobs)


def expand_trials(specification):
    """Support explicit recipes or a full Cartesian grid, never both."""
    if 'grid' not in specification:
        return specification['configs']
    if 'configs' in specification:
        raise ValueError('Use either grid or configs for a model')
    grid = specification['grid']
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError('Each grid parameter needs a nonempty list')
    trials = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        label = '_'.join(key + '-' + str(value).replace('.', 'p') for key, value in params.items())
        trials.append(dict(id=label, params=params))
    return trials


def successful(directory):
    state_path = directory / 'status.json'
    if not state_path.exists():
        return False
    state = read_json(state_path)
    result = directory / state.get('attempt', '') / 'result.json'
    return state.get('status') == 'success' and result.exists() and read_json(result).get('status') == 'success'


def summarize(root, manifest):
    rows = []
    for phase in ('original', 'item_only'):
        for job in manifest['jobs']:
            directory = root / phase / job['id']
            row = dict(phase=phase, job=job['id'], status='pending', model=job['model'])
            row.update({'param_' + key: value for key, value in job['params'].items()})
            row['param_gnn'] = job['params']['gnn'] if phase == 'original' else 'x' + job['params']['gnn']
            path = directory / 'status.json'
            if path.exists():
                state = read_json(path)
                row.update(status=state['status'], attempt=state.get('attempt'), returncode=state.get('returncode'))
                row['train_log'] = str((directory / state.get('attempt', '') / 'train.log').relative_to(root))
                result_path = directory / state.get('attempt', '') / 'result.json'
                if successful(directory):
                    result = read_json(result_path)
                    row.update(best_epoch=result['best']['epoch'], elapsed_seconds=result['elapsed_seconds'],
                               selection_split=result['best']['selection_split'])
                    for split in ('validation', 'test'):
                        for metric, values in result['best'][split].items():
                            for k, value in zip(result['ks'], values):
                                row[f'{split}_{metric}@{k}'] = value
            rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = root / 'summary.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(root / 'summary.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['original', 'item_only'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments.json')
    parser.add_argument('--output', type=Path, required=True, help='Same directory for both phases')
    parser.add_argument('--seed', type=int, default=None, help='One seed; omitted uses 2025 on first launch')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', help='Run one dataset independently; use a separate output per dataset/phase')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    opts = parser.parse_args(argv)
    root = opts.output.resolve()
    manifest_path = root / 'manifest.json'
    def selected_config():
        config = copy.deepcopy(read_json(opts.config))
        if opts.dataset:
            if opts.dataset not in config['datasets']:
                parser.error('Dataset is not present in the configuration')
            config['datasets'] = [opts.dataset]
        return config
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if opts.dataset and manifest['config']['datasets'] != [opts.dataset]:
            parser.error('Dataset differs from frozen manifest; use a new output directory')
        if opts.seed is not None and opts.seed != manifest['seed']:
            parser.error('Seed differs from frozen manifest; use a new output directory')
        if manifest['code_digest'] != code_digest():
            parser.error('Training code changed; use a new output directory for a consistent comparison')
        if (opts.phase == 'original' or opts.dataset) and selected_config() != manifest['config']:
            parser.error('Configuration changed; restore it or use a new output directory')
    else:
        if opts.phase == 'item_only' and not opts.dataset:
            parser.error('Run original first to create a frozen manifest')
        manifest = build_manifest(selected_config(), 2025 if opts.seed is None else opts.seed)
    if opts.phase == 'item_only' and not opts.dry_run and not opts.dataset:
        missing = [j['id'] for j in manifest['jobs'] if not successful(root / 'original' / j['id'])]
        if missing:
            parser.error(f'{len(missing)} baseline jobs are incomplete; rerun original first')
    print(f"{opts.phase}: {len(manifest['jobs'])} jobs, single seed={manifest['seed']}", flush=True)
    if opts.dry_run:
        for job in manifest['jobs']:
            model = job['model'] if opts.phase == 'original' else 'x' + job['model']
            print(job['id'], '->', model)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive scheduler lock; retained after abrupt OS termination for manual inspection.
    lock = root / '.running.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        parser.error('A scheduler lock exists; confirm no old process is running before removing it')
    try:
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        failures = 0
        for index, job in enumerate(manifest['jobs'], 1):
            directory = root / opts.phase / job['id']
            if successful(directory):
                print(f'[{index}] SKIP {job["id"]}', flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            attempt_number = 1
            while (directory / f'attempt_{attempt_number:03d}').exists():
                attempt_number += 1
            attempt = directory / f'attempt_{attempt_number:03d}'
            attempt.mkdir()
            params = dict(job['params'])
            if opts.phase == 'item_only':
                params['gnn'] = 'x' + job['model']
            params.update(gpu_id=opts.gpu_id, cuda=not opts.cpu,
                          run_dir=str(attempt), out_dir=str(attempt))
            command = [sys.executable, '-u', str(ROOT / 'main.py'), *cli(params)]
            write_json(attempt / 'command.json', command)
            state = dict(status='running', attempt=attempt.name, started=time.time())
            write_json(directory / 'status.json', state)
            write_json(attempt / 'status.json', state)
            print(f'[{index}/{len(manifest["jobs"])}] START {job["id"]} -> {attempt / "train.log"}', flush=True)
            process = None
            try:
                with (attempt / 'train.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    code = process.wait()
                result = attempt / 'result.json'
                ok = code == 0 and result.exists() and read_json(result).get('status') == 'success'
                state.update(status='success' if ok else 'failed', returncode=code, ended=time.time())
            except KeyboardInterrupt:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                state.update(status='interrupted', ended=time.time())
                write_json(directory / 'status.json', state)
                write_json(attempt / 'status.json', state)
                summarize(root, manifest)
                raise
            except OSError as exc:
                state.update(status='failed', error=str(exc), ended=time.time())
            write_json(directory / 'status.json', state)
            write_json(attempt / 'status.json', state)
            summarize(root, manifest)
            failures += state['status'] != 'success'
            print(f'[{index}] {state["status"].upper()} {job["id"]}', flush=True)
        summarize(root, manifest)
        return 1 if failures else 0
    finally:
        lock.unlink()


if __name__ == '__main__':
    sys.exit(main())
