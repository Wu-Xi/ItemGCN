"""Sweep B in U=BQ with one fixed base-model recipe per model."""
import argparse
import copy
import math
from pathlib import Path
import run_experiments as runner


def representation_trials(exponents):
    values = list(dict.fromkeys(exponents))
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Exponents must be finite and nonnegative')
    presets = {(0., 0.): 'sum', (1., 0.): 'mean', (.5, 0.): 'sqrt', (.5, .5): 'sym'}
    trials = [dict(id=name, params=dict(b_mode=name, b_a=a, b_b=b))
              for (a, b), name in presets.items()]
    for a in values:
        for b in values:
            if (a, b) not in presets:
                trials.append(dict(id=f'degree_a{a:g}_b{b:g}'.replace('.', 'p'),
                                   params=dict(b_mode='degree', b_a=a, b_b=b)))
    for b in values:
        if b != 0:  # weighted_mean with b=0 is exactly mean
            trials.append(dict(id=f'weighted_mean_b{b:g}'.replace('.', 'p'),
                               params=dict(b_mode='weighted_mean', b_a=0., b_b=b)))
    return trials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=runner.ROOT / 'experiments.json')
    parser.add_argument('--model', choices=('all',) + runner.MODELS, default='all')
    parser.add_argument('--trial', help='Fixed base recipe ID; required if model has multiple recipes')
    parser.add_argument('--exponents', type=float, nargs='+', default=[0., .25, .5, .75, 1.])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = copy.deepcopy(runner.read_json(args.base_config))
    if args.dataset not in config['datasets']:
        parser.error('Dataset is absent from base configuration')
    models = list(config['models']) if args.model == 'all' else [args.model]
    if args.model == 'all' and args.trial:
        parser.error('--trial requires one --model')
    try:
        trials = representation_trials(args.exponents)
    except ValueError as exc:
        parser.error(str(exc))
    specifications = {}
    for model in models:
        if model not in config['models']:
            parser.error('Model is absent from base configuration')
        spec = config['models'][model]
        recipes = runner.expand_trials(spec)
        selected = [t for t in recipes if t['id'] == args.trial] if args.trial else recipes[:1]
        if len(selected) != 1:
            parser.error('Select --trial from: ' + ', '.join(t['id'] for t in recipes))
        shared = dict(spec.get('shared', {}))
        shared.update(selected[0]['params'])
        shared['save'] = False
        specifications[model] = dict(shared=shared, configs=trials,
            source='User representation sweep; fixed base recipe ' + selected[0]['id'])
        print(model, 'fixed recipe:', selected[0]['id'], shared, flush=True)
    config['datasets'] = [args.dataset]
    config['models'] = specifications
    manifest = runner.build_manifest(config, args.seed)
    print(f"User representation: {len(manifest['jobs'])} trials, models={','.join(models)}", flush=True)
    if args.dry_run:
        for trial in trials:
            print(trial['id'], trial['params'])
        return 0
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'user_representation_config.json'
    if path.exists() and runner.read_json(path) != config:
        parser.error('Sweep settings changed; use a new output directory')
    if not path.exists():
        runner.write_json(path, config)
    command = ['--config', str(path), '--dataset', args.dataset, '--phase', 'item_only',
               '--output', str(root), '--gpu_id', str(args.gpu_id), '--seed', str(args.seed)]
    if args.cpu:
        command.append('--cpu')
    return runner.main(command)


if __name__ == '__main__':
    raise SystemExit(main())
