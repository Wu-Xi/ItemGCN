"""Evaluate fixed Original/Item-only checkpoints by training user activity."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_best_checkpoints import select_jobs
from run_experiments import cli, code_digest, identifier
from utils.checkpoint import data_fingerprints, read_metadata, sha256

ROOT = Path(__file__).resolve().parent
PHASES = ('original', 'item_only')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def plan(opts):
    from types import SimpleNamespace
    from utils.parser import parse_args
    config = read_json(opts.config)
    jobs = select_jobs(config, opts.dataset)
    if opts.models:
        missing = set(opts.models) - {j['model'] for j in jobs}
        if missing:
            raise ValueError('Unknown models: ' + ', '.join(sorted(missing)))
        jobs = [j for j in jobs if j['model'] in opts.models]
    for dataset in {j['dataset'] for j in jobs}:
        observed = data_fingerprints(SimpleNamespace(dataset=dataset, data_path=opts.data_path))
        if observed != config['data_files'][dataset]:
            raise ValueError('Data differs from frozen configuration: ' + dataset)
    runtime_keys = {'cuda', 'gpu_id', 'data_path', 'out_dir', 'run_dir', 'save'}
    plans = []
    for job in jobs:
        folder = opts.checkpoint_root / job['dataset'] / job['model'] / job['phase']
        status = read_json(folder / 'status.json')
        if status['status'] != 'success':
            raise ValueError('Checkpoint task has not succeeded: ' + str(folder))
        checkpoint = folder / identifier(status['attempt']) / 'model_.ckpt'
        _, meta = read_metadata(checkpoint, fp16_manifest=opts.fp16_manifest)
        if meta['data_files'] != config['data_files'][job['dataset']]:
            raise ValueError('Checkpoint data fingerprint differs: ' + str(checkpoint))
        expected = vars(parse_args(cli(job['params'])))
        actual = vars(parse_args(cli(meta['config'])))
        changed = [key for key in expected if key not in runtime_keys and expected[key] != actual[key]]
        if changed:
            raise ValueError('Checkpoint differs from frozen config: ' + str(checkpoint) + ': ' + str(changed))
        if max(opts.ks) > meta['n_params']['n_items']:
            raise ValueError('K exceeds catalog size')
        plans.append(dict(dataset=job['dataset'], model=job['model'], phase=job['phase'],
                          checkpoint=str(checkpoint.resolve()), sha256=sha256(checkpoint),
                          metadata_sha256=sha256(checkpoint.with_name('checkpoint.json'))))
    return plans


def worker(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    import numpy as np
    import torch
    from utils.checkpoint import load_for_evaluation
    from utils.user_activity import evaluate_users, partition_users, aggregate
    torch.set_num_threads(opts.threads)
    # Evaluation only; initialization is overwritten by the checkpoint.
    torch.manual_seed(2025)
    if not opts.cpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, args, data, meta = load_for_evaluation(
        opts.checkpoint, data_path=opts.data_path, cuda=not opts.cpu,
        gpu_id=opts.gpu_id, fp16_manifest=opts.fp16_manifest)
    user_ids, metrics = evaluate_users(model, data, opts.ks, opts.batch_size)
    groups, degrees, stats = partition_users(user_ids, data[1]['train_user_set'])
    order = np.argsort(user_ids)
    overall = {m: values.mean(axis=0).tolist() for m, values in metrics.items()}
    recorded_ks = ast.literal_eval(meta['config']['Ks'])
    recorded = meta['best']['test']
    drift = {m: [(overall[m][i] - recorded[m][recorded_ks.index(k)]) if k in recorded_ks else None
                 for i, k in enumerate(opts.ks)] for m in metrics}
    comparable = [abs(v) for values in drift.values() for v in values if v is not None]
    max_drift = max(comparable) if comparable else None
    result = dict(checkpoint=str(opts.checkpoint.resolve()), checkpoint_sha256=sha256(opts.checkpoint),
                  epoch=meta['best']['epoch'], data_files=meta['data_files'], ks=opts.ks,
                  partition='test users sorted by (distinct train degree, numeric ID); numpy.array_split(4)',
                  groups=stats, overall=overall, group_metrics=aggregate(metrics, groups),
                  recorded_best=meta['best'], difference_from_source_fp32=drift,
                  max_absolute_recorded_difference=max_drift,
                  review_required=max_drift is None or max_drift > opts.drift_tolerance,
                  drift_tolerance=opts.drift_tolerance,
                  provenance=meta.get('evaluation_provenance', dict(storage='original', integrity='SHA256 verified')),
                  inference_dtype=str(next(model.parameters()).dtype),
                  environment=dict(torch=str(torch.__version__), python=sys.version,
                                   device=str(next(model.parameters()).device), batch_size=opts.batch_size))
    opts.output.mkdir(parents=True, exist_ok=True)
    # JSON is the completion marker and is written only after the numeric artifact.
    temporary = opts.output / 'users.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, user_ids=user_ids[order], groups=groups[order], degrees=degrees[order],
                            **{m: values[order] for m, values in metrics.items()})
    temporary.replace(opts.output / 'users.npz')
    result['users_npz_sha256'] = sha256(opts.output / 'users.npz')
    write_json(opts.output / 'evaluation.json', result)
    print('Saved ' + str(opts.output), flush=True)


def summarize(opts, plans):
    import numpy as np
    from utils.user_activity import METRICS, paired_gains
    rows, summaries, flagged = [], [], []
    for dataset, model in sorted({(p['dataset'], p['model']) for p in plans}):
        arrays, results = {}, {}
        for phase in PHASES:
            directory = opts.output_root / dataset / model / phase
            result = read_json(directory / 'evaluation.json')
            if sha256(directory / 'users.npz') != result['users_npz_sha256']:
                raise ValueError('Numeric artifact checksum mismatch: ' + str(directory))
            with np.load(directory / 'users.npz', allow_pickle=False) as archive:
                arrays[phase] = {key: archive[key] for key in archive.files}
            results[phase] = result
            if result['review_required']:
                flagged.append(f'{dataset}/{model}/{phase}')
        a, b = arrays['original'], arrays['item_only']
        for key in ('user_ids', 'groups', 'degrees'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Paired user partition differs: ' + key)
        if results['original']['data_files'] != results['item_only']['data_files']:
            raise ValueError('Paired data files differ')
        pair_rows = paired_gains(a, b, a['groups'], opts.ks)
        for row in pair_rows:
            degree_stats = results['original']['groups'][row['group'] - 1]
            row.update(dataset=dataset, model=model,
                       degree_min=degree_stats['degree_min'], degree_max=degree_stats['degree_max'],
                       review_required=any(r['review_required'] for r in results.values()))
        rows.extend(pair_rows)
        for metric in METRICS:
            for k in opts.ks:
                selected = [r for r in pair_rows if r['metric'] == metric and r['k'] == k]
                best = max(r['absolute_gain'] for r in selected)
                rel = [r for r in selected if r['relative_gain_pct'] is not None]
                best_relative = max((r['relative_gain_pct'] for r in rel), default=None)
                largest = [r['group'] for r in selected if np.isclose(r['absolute_gain'], best, rtol=0, atol=1e-12)]
                summaries.append(dict(dataset=dataset, model=model, metric=metric, k=k,
                                      largest_absolute_gain_groups=largest,
                                      largest_relative_gain_groups=[r['group'] for r in rel if np.isclose(
                                          r['relative_gain_pct'], best_relative, rtol=0, atol=1e-12)],
                                      any_positive_gain=best > 0,
                                      overall_gain=sum(r['overall_gain_contribution'] for r in selected),
                                      g1_minus_g4_absolute_gain=selected[0]['absolute_gain'] - selected[3]['absolute_gain'],
                                      review_required=selected[0]['review_required']))
    write_json(opts.output_root / 'summary.json', dict(rows=rows, comparisons=summaries,
                                                      checkpoints_needing_review=flagged))
    with (opts.output_root / 'group_gains.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ['# User activity analysis', '',
              'G1 has the fewest distinct training interactions; G4 has the most.',
              'Gains compare Item-only minus Original within the same dataset/backbone.',
              'These are descriptive results for fixed checkpoints, not multi-seed significance claims.',
              'A positive G1−G4 gain difference means a larger absolute benefit for G1.', '',
              '| Dataset | Model | Metric | Largest absolute gain group | Overall gain | G1−G4 gain | Review |',
              '|---|---|---|---|---:|---:|---|']
    for item in summaries:
        if item['k'] != 20 or item['metric'] == 'hit_ratio':
            continue
        best = ', '.join('G' + str(g) for g in item['largest_absolute_gain_groups'])
        if not item['any_positive_gain']:
            best += ' (no positive gain)'
        report.append(f"| {item['dataset']} | {item['model']} | {item['metric']}@20 | {best} | "
                      f"{item['overall_gain']:+.6f} | {item['g1_minus_g4_absolute_gain']:+.6f} | "
                      f"{'YES' if item['review_required'] else ''} |")
    if flagged:
        report += ['', 'Source-checkpoint score differences exceed the chosen review tolerance:',
                   *['- ' + value for value in flagged],
                   '', 'Review these before drawing paper conclusions. The threshold is an audit aid, '
                   'not proof of FP16 equivalence.']
    (opts.output_root / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if opts.plot:
        plot_gains(opts.output_root, rows)
    print(f'Summary saved: {opts.output_root / "REPORT.md"}; checkpoints needing review: {len(flagged)}')


def plot_gains(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in sorted({r['dataset'] for r in rows}):
        models = sorted({r['model'] for r in rows if r['dataset'] == dataset})
        for metric in ('recall', 'ndcg'):
            fig, axes = plt.subplots((len(models) + 2) // 3, 3, figsize=(10, 2.7 * ((len(models) + 2) // 3)),
                                     squeeze=False, constrained_layout=True)
            for axis, model in zip(axes.flat, models):
                part = sorted([r for r in rows if r['dataset'] == dataset and r['model'] == model
                               and r['metric'] == metric and r['k'] == 20], key=lambda r: r['group'])
                if not part:
                    axis.set_visible(False)
                    continue
                axis.bar([r['group'] for r in part], [100 * r['absolute_gain'] for r in part],
                         color=['#3479b2' if r['absolute_gain'] >= 0 else '#c86253' for r in part])
                axis.axhline(0, color='#333333', linewidth=.7)
                axis.set_xticks([1, 2, 3, 4])
                axis.set_xticklabels(['G1\nLow', 'G2', 'G3', 'G4\nHigh'])
                axis.set_title(model + (' *review' if part[0]['review_required'] else ''))
                axis.set_ylabel('Absolute gain (percentage points)')
            for axis in list(axes.flat)[len(models):]:
                axis.set_visible(False)
            fig.suptitle(f'{dataset}: {metric.upper()}@20, Item-only minus Original')
            for suffix in ('png', 'svg'):
                fig.savefig(output / f'{dataset}_{metric}20_user_gain.{suffix}', dpi=220)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'best54_configs.json')
    parser.add_argument('--data-path', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dataset', choices=['ali', 'amazon', 'yelp2018'])
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--fp16-manifest', type=Path, help='Explicit opt-in for the converted sharing package')
    parser.add_argument('--ks', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--drift-tolerance', type=float, default=1e-4,
                        help='Absolute metric difference from source FP32 record triggering review')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Create PNG/SVG gain figures using matplotlib')
    parser.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    opts = parser.parse_args()
    opts.ks = sorted(set(opts.ks))
    if min(opts.ks) < 1 or min(opts.batch_size, opts.threads) < 1 or opts.gpu_id < 0 or opts.drift_tolerance < 0:
        parser.error('Invalid K, batch size, threads, GPU ID, or drift tolerance')
    if opts.checkpoint:
        if opts.output is None:
            parser.error('Worker requires --output')
        worker(opts)
        return
    if opts.checkpoint_root is None or opts.output_root is None:
        parser.error('--checkpoint-root and --output-root are required')
    plans = plan(opts)
    digest = hashlib.sha256((code_digest() + sha256(Path(__file__))).encode()).hexdigest()
    specification = dict(version=1, jobs=plans, ks=opts.ks, batch_size=opts.batch_size,
                         cpu=opts.cpu, threads=opts.threads, drift_tolerance=opts.drift_tolerance,
                         code_sha256=digest, config_sha256=sha256(opts.config),
                         fp16_manifest_sha256=sha256(opts.fp16_manifest) if opts.fp16_manifest else None)
    print(f'Validated {len(plans)} fixed checkpoints and data fingerprints; four user groups.', flush=True)
    if opts.dry_run:
        for p in plans:
            print(f"  {p['dataset']}/{p['model']}/{p['phase']}")
        return
    opts.output_root.mkdir(parents=True, exist_ok=True)
    manifest = opts.output_root / 'evaluation_manifest.json'
    lock = opts.output_root / '.evaluation.lock'
    # Atomic creation prevents two runners from updating the same output root.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        if manifest.exists():
            if read_json(manifest) != specification:
                raise ValueError('Evaluation inputs/settings/code changed; use a new output root')
        elif opts.summarize_only:
            raise ValueError('No existing evaluation manifest to summarize')
        else:
            write_json(manifest, specification)
        for p in plans:
            directory = opts.output_root / p['dataset'] / p['model'] / p['phase']
            complete = directory / 'evaluation.json'
            if complete.exists():
                result = read_json(complete)
                if (result['checkpoint_sha256'] != p['sha256'] or result['ks'] != opts.ks
                        or sha256(directory / 'users.npz') != result['users_npz_sha256']):
                    raise ValueError('Existing result mismatch: ' + str(directory))
                print('SKIP ' + str(directory), flush=True)
                continue
            if opts.summarize_only:
                raise ValueError('Missing evaluation: ' + str(directory))
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--checkpoint', p['checkpoint'],
                       '--data-path', str(opts.data_path.resolve()), '--output', str(directory.resolve()),
                       '--batch-size', str(opts.batch_size), '--gpu-id', str(opts.gpu_id),
                       '--threads', str(opts.threads), '--drift-tolerance', str(opts.drift_tolerance),
                       '--ks', *map(str, opts.ks)]
            if opts.cpu:
                command.append('--cpu')
            if opts.fp16_manifest:
                command += ['--fp16-manifest', str(opts.fp16_manifest.resolve())]
            print('EVALUATE ' + str(directory), flush=True)
            with (directory / 'evaluation.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summarize(opts, plans)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
