"""Joint sweep: all model recipes x 28 B constructions (sym excluded by default)."""
import argparse
import copy
import sys
from pathlib import Path

import run_experiments as runner

B_KEYS = {'b_mode', 'b_a', 'b_b'}


def representation_trials(include_sym=False):
    """24 non-sym degree points + 4 weighted means; optionally restore sym."""
    values = [0., .25, .5, .75, 1.]
    presets = {(0., 0.): 'sum', (1., 0.): 'mean', (.5, 0.): 'sqrt', (.5, .5): 'sym'}
    trials = [dict(id=name, params=dict(b_mode=name, b_a=a, b_b=b))
              for (a, b), name in presets.items()]
    for a in values:
        for b in values:
            if (a, b) not in presets:
                trials.append(dict(id=f'degree_a{a:g}_b{b:g}'.replace('.', 'p'),
                                   params=dict(b_mode='degree', b_a=a, b_b=b)))
    for b in values[1:]:
        trials.append(dict(id=f'weighted_mean_b{b:g}'.replace('.', 'p'),
                           params=dict(b_mode='weighted_mean', b_a=0., b_b=b)))
    return trials if include_sym else [trial for trial in trials if trial['id'] != 'sym']


def build_joint_config(base, dataset, model='all', include_sym=False, models=None):
    """Keep every base recipe, and nest the B sweep inside each recipe."""
    if dataset not in base['datasets']:
        raise ValueError('Dataset is absent from base configuration: ' + dataset)
    runner.identifier(dataset)
    if models is not None:
        if model != 'all':
            raise ValueError('Use either model or models, not both')
        if not models or len(set(models)) != len(models):
            raise ValueError('Model list must be nonempty and contain no duplicates')
        unknown = set(models) - set(base['models'])
        if unknown:
            raise ValueError('Models absent from base configuration: ' + ', '.join(sorted(unknown)))
        # Preserve base-config order so an equivalent model list resumes identically.
        selected_models = [name for name in base['models'] if name in models]
    else:
        selected_models = list(base['models']) if model == 'all' else [model]
    b_trials = representation_trials(include_sym=include_sym)
    config = copy.deepcopy(base)
    config['datasets'] = [dataset]
    config['models'] = {}
    counts = {}
    for name in selected_models:
        if name not in base['models'] or name not in runner.MODELS:
            raise ValueError('Model is absent or unsupported: ' + name)
        spec = base['models'][name]
        if B_KEYS.intersection(spec.get('grid', {})):
            raise ValueError('Base grid must vary model parameters, not B: ' + name)
        recipes = runner.expand_trials(spec)
        trials = []
        for recipe in recipes:
            base_id = runner.identifier(recipe['id'])
            for b_trial in b_trials:
                params = copy.deepcopy(recipe['params'])
                params.update(b_trial['params'])
                params['save'] = False
                trials.append(dict(id=base_id + '__B_' + b_trial['id'], params=params,
                    source='Joint sweep; base recipe=' + base_id + '; B=' + b_trial['id']))
        if not trials or len({trial['id'] for trial in trials}) != len(trials):
            raise ValueError('Empty or duplicate base recipe IDs: ' + name)
        config['models'][name] = dict(shared=copy.deepcopy(spec.get('shared', {})), configs=trials)
        counts[name] = (len(recipes), len(trials))
    config['joint_sweep'] = dict(version=1, loop_order=['model', 'base_recipe', 'B'],
                                representations=b_trials)
    return config, counts


def freeze_config(path, config):
    """Freeze the generated recipe list before handing control to the runner."""
    # This short-lived lock protects initial config creation from parallel launchers.
    lock = path.parent / '.joint-config.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError('Config creation is locked: ' + str(lock))
    try:
        if path.exists():
            if runner.read_json(path) != config:
                raise ValueError('Joint sweep settings changed; use a new output directory')
        else:
            # Refuse to attach a new recipe list to an unrelated existing experiment.
            if (path.parent / 'manifest.json').exists():
                raise ValueError('Existing manifest is not from this joint sweep; use a new output directory')
            runner.write_json(path, config)
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='One dataset per scheduler/output directory')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=runner.ROOT / 'experiments.json')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--model', choices=('all',) + runner.MODELS, default='all')
    selection.add_argument('--models', nargs='+', choices=runner.MODELS,
                           help='Run a model subset in one sequential scheduler')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--include-sym', action='store_true',
                        help='Also train sym, restoring all 29 B constructions (default: 28)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate all jobs; do not create files or train')
    parser.add_argument('--list-jobs', action='store_true', help='Print every joint recipe during dry-run')
    args = parser.parse_args(argv)
    if args.gpu_id < 0:
        parser.error('--gpu_id must be nonnegative')
    if args.list_jobs and not args.dry_run:
        parser.error('--list-jobs requires --dry-run')
    try:
        config, counts = build_joint_config(runner.read_json(args.base_config), args.dataset, args.model,
                                            include_sym=args.include_sym, models=args.models)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    print(f'Joint Item-only sweep: dataset={args.dataset}, gpu={args.gpu_id}, seed={args.seed}', flush=True)
    b_count = len(config['joint_sweep']['representations'])
    print('sym: ' + ('included' if args.include_sym else 'excluded (reuse the earlier sym sweep separately)'), flush=True)
    for model, (base_count, total) in counts.items():
        print(f'  {model}: {base_count} base recipes x {b_count} B constructions = {total} jobs', flush=True)
    print(f'Total: {sum(total for _, total in counts.values())} jobs. Order: model -> base recipe -> B.', flush=True)
    print('Output: ' + str(args.output.resolve()), flush=True)
    if args.dry_run:
        # Resolve and validate every CLI parameter without importing torch or training.
        manifest = runner.build_manifest(config, args.seed)
        if args.list_jobs:
            for job in manifest['jobs']:
                print(job['id'], '->', 'x' + job['params']['gnn'])
        print(f'DRY RUN: validated {len(manifest["jobs"])} jobs; no files created.', flush=True)
        return 0

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'joint_user_representation_config.json'
    try:
        freeze_config(path, config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    command = ['--config', str(path), '--dataset', args.dataset, '--phase', 'item_only',
               '--output', str(root), '--gpu_id', str(args.gpu_id), '--seed', str(args.seed)]
    if args.cpu:
        command.append('--cpu')
    return runner.main(command)


if __name__ == '__main__':
    sys.exit(main())
