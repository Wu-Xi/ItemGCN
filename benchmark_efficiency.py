"""Benchmark selected recipes from scratch in isolated, sequential processes."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import run_experiments as runner
from run_best_checkpoints import select_jobs
from utils.checkpoint import data_fingerprints, sha256

ROOT = Path(__file__).resolve().parent
MODELS = ('rns', 'mixgcf', 'ahns', 'stabcf', 'directau', 'graphau', 'simgcl', 'sgl', 'recdcl')
DATASETS = ('ali', 'amazon', 'yelp2018')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def benchmark_digest():
    return fingerprint([runner.code_digest(), sha256(Path(__file__)),
                        sha256(ROOT / 'run_best_checkpoints.py')])


def environment(cpu, gpu_id):
    # No torch imports before this point, including in the scheduling process.
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if cpu else str(gpu_id)
    import torch
    import numpy
    import scipy
    import prettytable
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Activate the GPU training environment; '
                           '--cpu is only for smoke testing.')
    result = dict(python=platform.python_version(), torch=str(torch.__version__),
                  numpy=numpy.__version__, scipy=scipy.__version__,
                  prettytable=prettytable.__version__, cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), host=platform.node(),
                  platform=platform.platform(), processor=platform.processor(),
                  torch_threads=torch.get_num_threads(),
                  thread_environment={k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                  device='cpu' if cpu else 'cuda', gpu_id=None if cpu else gpu_id)
    if not cpu:
        props = torch.cuda.get_device_properties(0)
        result.update(gpu_name=props.name, gpu_total_bytes=props.total_memory,
                      gpu_capability=[props.major, props.minor],
                      gpu_uuid=str(getattr(props, 'uuid', 'unavailable')))
    return result


def train_epochs(model, optimizer, train_cf, users, sizes, args, protocol):
    """Same batches/updates as main.py, without evaluation or checkpoint IO.

    Like main.py, shuffle happens before the timed block. The epoch hook,
    negative/history sampling, device transfers and every optimizer step are timed.
    """
    import numpy as np
    import torch
    import main as training
    training.args, training.device = args, torch.device('cuda:0' if args.cuda else 'cpu')
    training.n_users, training.n_items = sizes['n_users'], sizes['n_items']
    negatives = getattr(model, 'requires_negative_sampling', True)
    negative_count = getattr(model, 'negative_sample_count', args.n_negs)
    histories = getattr(model, 'requires_history', False)
    train_sets = training.prepare_train_sets(users['train_user_set'], sizes['n_items']) if negatives else None
    if args.cuda:
        torch.cuda.synchronize()
        # Current model/graph allocations remain counted. Include the first Adam
        # step and all warm-up epochs in the overall training memory peak.
        torch.cuda.reset_peak_memory_stats()
    records = []
    expected_batches = math.ceil(len(train_cf) / args.batch_size)
    for epoch in range(protocol['warmup_epochs'] + protocol['measure_epochs']):
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        shuffled = train_cf[index]
        model.train()
        loss = torch.zeros((), device=training.device)
        if args.cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        if hasattr(model, 'on_train_epoch_start'):
            model.on_train_epoch_start()
        batches = 0
        for offset in range(0, len(shuffled), args.batch_size):
            batch = training.get_feed_dictv4(
                shuffled, users['train_user_set'], train_sets, offset, offset + args.batch_size,
                negative_count, negatives, histories)
            optimizer.zero_grad(set_to_none=True)
            batch_loss, bpr_loss, reg_loss = model(batch, epoch)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError('Non-finite loss at epoch ' + str(epoch))
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.detach()
            batches += 1
        loss_value = loss.item()
        if args.cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if batches != expected_batches or not math.isfinite(loss_value):
            raise RuntimeError('Incomplete epoch or non-finite accumulated loss')
        record = dict(epoch=epoch, measured=epoch >= protocol['warmup_epochs'],
                      seconds=elapsed, loss_sum=loss_value, batches=batches,
                      interactions=len(train_cf))
        records.append(record)
        print(f"epoch={epoch} measured={record['measured']} time={elapsed:.4f}s batches={batches}", flush=True)
    times = [r['seconds'] for r in records if r['measured']]
    peak = torch.cuda.max_memory_allocated() if args.cuda else None
    reserved = torch.cuda.max_memory_reserved() if args.cuda else None
    return dict(epoch_records=records, time_epoch_mean_s=statistics.mean(times),
                time_epoch_std_s=statistics.stdev(times) if len(times) > 1 else 0.,
                time_epoch_median_s=statistics.median(times),
                peak_allocated_bytes=peak, peak_allocated_gib=peak / 1024**3 if peak is not None else None,
                peak_reserved_bytes=reserved,
                peak_reserved_gib=reserved / 1024**3 if reserved is not None else None)


def worker(request_path):
    request = runner.read_json(request_path)
    protocol = request['protocol']
    observed_env = environment(protocol['cpu'], protocol['gpu_id'])
    if observed_env != request['environment'] or benchmark_digest() != request['code_digest']:
        raise RuntimeError('Benchmark code or environment changed; use a new output root')
    import random
    import numpy as np
    import torch
    from utils.data_loader import load_data
    from utils.model_factory import build_model
    args = runner.parse_args(runner.cli(request['params']))
    if data_fingerprints(args) != request['data_files']:
        raise ValueError('Data files changed before training')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start = time.perf_counter()
    train_cf, users, matrices, sizes, adj, b, *_ = load_data(args)
    train_cf = torch.as_tensor(np.asarray(train_cf), dtype=torch.long)
    if len(train_cf) == 0 or args.batch_size < 1:
        raise ValueError('Require nonempty training data and positive batch size')
    model = build_model(sizes, args, adj, b, matrices['train_sp_mat']).to(
        torch.device('cuda:0' if args.cuda else 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.cuda:
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start
    result = train_epochs(model, optimizer, train_cf, users, sizes, args, protocol)
    result.update(status='success', request_digest=fingerprint(request),
                  dataset=request['job']['dataset'], model=request['job']['model'],
                  phase=request['job']['phase'], B=request['job'].get('B'),
                  trainable_params=count, params_m=count / 1e6, n_params=sizes,
                  setup_seconds=setup_seconds, initialization='random',
                  environment=observed_env, protocol=protocol, config=vars(args),
                  notes=dict(time='Full epoch: hook, sampling, transfers, forward, backward, Adam. '
                                  'Excludes shuffle, setup, evaluation and saving; synchronized GPU boundaries.',
                             memory='Peak torch allocated bytes over warm-up AND measured training. '
                                    'Includes live model/graph/optimizer tensors. Not total nvidia-smi memory.',
                             std='Sample standard deviation across consecutive measured epochs; not seeds.'))
    runner.write_json(request_path.with_name('result.json'), result)
    print(f"SUCCESS params={count} mean_s={result['time_epoch_mean_s']:.4f} "
          f"peak_GiB={result['peak_allocated_gib']}", flush=True)


def completed_result(directory, request):
    try:
        state = runner.read_json(directory / 'status.json')
        if state['status'] != 'success':
            return None
        attempt = directory / runner.identifier(state['attempt'])
        result = runner.read_json(attempt / 'result.json')
        if result['status'] != 'success' or result['request_digest'] != fingerprint(request):
            return None
        if result['trainable_params'] <= 0 or not math.isfinite(result['time_epoch_mean_s']):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_summary(output):
    fields = ['dataset', 'model', 'phase', 'status', 'B', 'trainable_params', 'params_m',
              'peak_allocated_gib', 'peak_reserved_gib', 'time_epoch_mean_s', 'time_epoch_std_s',
              'time_epoch_median_s', 'device', 'gpu_name', 'warmup_epochs', 'measure_epochs', 'attempt']
    rows = []
    for path in sorted(output.glob('*/*/*/status.json')):
        directory = path.parent
        request = runner.read_json(directory / 'manifest.json')
        state = runner.read_json(path)
        result = completed_result(directory, request)
        row = {k: request['job'].get(k) for k in ('dataset', 'model', 'phase', 'B')}
        row.update(status='success' if result else state['status'], attempt=state['attempt'],
                   device=request['environment']['device'], gpu_name=request['environment'].get('gpu_name'),
                   warmup_epochs=request['protocol']['warmup_epochs'],
                   measure_epochs=request['protocol']['measure_epochs'])
        if not result and state['status'] == 'success':
            row['status'] = 'missing_or_invalid_result'
        if result:
            row.update({key: result[key] for key in fields if key in result})
        rows.append(row)
    temporary = output / 'summary.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / 'summary.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'best54_configs.json')
    parser.add_argument('--data-path', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'experiment_results/efficiency54')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--phase', choices=('all', 'original', 'item_only'), default='all')
    parser.add_argument('--gpu-id', '--gpu_id', type=int, default=0)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--measure-epochs', type=int, default=3)
    parser.add_argument('--cpu', action='store_true', help='Smoke tests only; GPU memory is null')
    parser.add_argument('--dry-run', action='store_true', help='Validate plan without training or output directories')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    opts = parser.parse_args(argv)
    if opts.gpu_id < 0 or opts.warmup_epochs < 1 or opts.measure_epochs < 1:
        parser.error('Require GPU ID >= 0, warmup epochs >= 1, measured epochs >= 1')
    if opts.probe:
        print(json.dumps(environment(opts.cpu, opts.gpu_id)))
        return 0
    if opts.worker:
        worker(opts.worker)
        return 0
    config = runner.read_json(opts.config)
    try:
        jobs = select_jobs(config, opts.dataset, opts.phase)
        jobs = [j for j in jobs if opts.model is None or j['model'] == opts.model]
        if not jobs:
            raise ValueError('No jobs match the filters')
        jobs.sort(key=lambda j: (DATASETS.index(j['dataset']), MODELS.index(j['model']),
                                 0 if j['phase'] == 'original' else 1))
    except ValueError as exc:
        parser.error(str(exc))
    protocol = dict(warmup_epochs=opts.warmup_epochs, measure_epochs=opts.measure_epochs,
                    cpu=opts.cpu, gpu_id=opts.gpu_id, initialization='random')
    data_path = str(opts.data_path.resolve()) + '/'
    for dataset in {j['dataset'] for j in jobs}:
        if data_fingerprints(SimpleNamespace(data_path=data_path, dataset=dataset)) != config['data_files'][dataset]:
            parser.error('Data files differ from selected configuration: ' + dataset)
    params = []
    for job in jobs:
        actual = dict(job['params'], data_path=data_path, cuda=not opts.cpu, gpu_id=opts.gpu_id,
                      save=False, run_dir=None, out_dir=None,
                      epoch=opts.warmup_epochs + opts.measure_epochs)
        runner.parse_args(runner.cli(actual))
        params.append(actual)
    print(f"{len(jobs)} jobs; random initialization; {opts.warmup_epochs} warm-up + "
          f"{opts.measure_epochs} measured full epochs; sequential GPU={opts.gpu_id}", flush=True)
    for job in jobs:
        print(f"  {job['dataset']}/{job['model']}/{job['phase']} B={job.get('B') or '-'}", flush=True)
    if opts.dry_run:
        return 0
    # Probe in a short-lived child so the scheduler never holds GPU memory.
    command = [sys.executable, str(Path(__file__).resolve()), '--probe', '--gpu-id', str(opts.gpu_id)]
    if opts.cpu:
        command.append('--cpu')
    probe = subprocess.run(command, capture_output=True, text=True)
    if probe.returncode:
        parser.error('Environment preflight failed:\n' + probe.stderr)
    env = json.loads(probe.stdout)
    output = opts.output_root.resolve()
    root_manifest = dict(version=1, protocol=protocol, environment=env, code_digest=benchmark_digest(),
                         config_sha256=sha256(opts.config), data_path=data_path)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'manifest.json').exists() and any(output.iterdir()):
        parser.error('Output is not an empty benchmark directory; choose another --output-root')
    lock = output / '.benchmark.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), host=platform.node()), stream)
    except FileExistsError:
        parser.error('Output is locked; check the existing process before removing: ' + str(lock))
    try:
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and runner.read_json(manifest_path) != root_manifest:
            parser.error('Code, config, device, environment or protocol changed; use a new output root')
        runner.write_json(manifest_path, root_manifest)
        failures = 0
        for job, actual in zip(jobs, params):
            directory = output / job['dataset'] / job['model'] / job['phase']
            directory.mkdir(parents=True, exist_ok=True)
            request = dict(job=job, params=actual, protocol=protocol, environment=env,
                           code_digest=root_manifest['code_digest'], data_files=config['data_files'][job['dataset']])
            manifest = directory / 'manifest.json'
            if manifest.exists() and runner.read_json(manifest) != request:
                raise ValueError('Job configuration changed: ' + str(directory))
            runner.write_json(manifest, request)
            if completed_result(directory, request):
                print('SKIP ' + str(directory), flush=True)
                write_summary(output)
                continue
            number = 1
            while (directory / f'attempt_{number:03d}').exists():
                number += 1
            attempt = directory / f'attempt_{number:03d}'
            attempt.mkdir()
            request_path = attempt / 'request.json'
            runner.write_json(request_path, request)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(request_path)]
            runner.write_json(attempt / 'command.json', command)
            state = dict(status='running', attempt=attempt.name, started=time.time())
            runner.write_json(directory / 'status.json', state)
            print('START ' + str(attempt), flush=True)
            process = None
            try:
                with (attempt / 'train.log').open('w', encoding='utf-8') as stream:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    code = process.wait()
                state.update(status='success' if code == 0 else 'failed', returncode=code)
                runner.write_json(directory / 'status.json', state)
                if code == 0 and not completed_result(directory, request):
                    state.update(status='failed', error='Missing or invalid benchmark result')
            except KeyboardInterrupt:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                state.update(status='interrupted')
                raise
            except OSError as exc:
                state.update(status='failed', error=str(exc))
            finally:
                state['ended'] = time.time()
                runner.write_json(directory / 'status.json', state)
                runner.write_json(attempt / 'status.json', state)
                write_summary(output)
            failures += state['status'] != 'success'
            print(state['status'].upper() + ' ' + str(attempt), flush=True)
        print('Summary: ' + str(output / 'summary.csv'), flush=True)
        return int(failures > 0)
    finally:
        lock.unlink()


if __name__ == '__main__':
    sys.exit(main())
