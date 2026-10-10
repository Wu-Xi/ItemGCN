"""Portable inference checkpoints, with split and weight integrity checks."""
import hashlib
import json
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprints(args):
    folder = Path(args.data_path) / args.dataset
    names = ['train.txt', 'test.txt']
    if args.dataset not in ('yelp2018', 'gowalla'):
        names.append('valid.txt')
    return {name: {'sha256': sha256(folder / name), 'bytes': (folder / name).stat().st_size}
            for name in names}


class BestCheckpoint:
    def __init__(self, args, n_params):
        import torch
        from run_experiments import code_digest
        self.directory = Path(args.out_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base = dict(format_version=1, config=dict(vars(args)), n_params=dict(n_params),
                         data_files=data_fingerprints(args), code_digest=code_digest(),
                         environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                          cuda_version=torch.version.cuda),
                         purpose='evaluation; optimizer state is not saved')

    def save(self, model, epoch, validation, test, selection_split):
        import torch
        from .experiment_log import write_json
        weights = self.directory / 'model_.ckpt'
        temporary = self.directory / 'model_.ckpt.tmp'
        # Retain the legacy state_dict format, including persistent buffers (e.g. RecDCL).
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
        temporary.replace(weights)
        metadata = dict(self.base, checkpoint_file=weights.name, checkpoint_sha256=sha256(weights),
                        best=dict(epoch=epoch, validation=validation, test=test, selection_split=selection_split))
        write_json(self.directory / 'checkpoint.json', metadata)


def read_metadata(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'model_.ckpt'
    metadata = json.loads(checkpoint.with_name('checkpoint.json').read_text(encoding='utf-8'))
    if metadata['format_version'] != 1 or sha256(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint metadata/weights mismatch: ' + str(checkpoint))
    return checkpoint, metadata


def load_for_evaluation(checkpoint, *, data_path=None, cuda=False, gpu_id=0):
    """Return (model, args, load_data_tuple, metadata). Never train or reselect an epoch.

    Set CUDA_VISIBLE_DEVICES before importing torch in GPU command-line entrypoints.
    Keep the original train/valid/test files: graph buffers are rebuilt from them.
    """
    import torch
    from .parser import parse_args
    from .data_loader import load_data
    from .model_factory import build_model
    from run_experiments import cli
    checkpoint, metadata = read_metadata(checkpoint)
    config = dict(metadata['config'])
    config.update(cuda=cuda, gpu_id=gpu_id, save=False, run_dir=None)
    if data_path is not None:
        config['data_path'] = str(Path(data_path)) + '/'
    args = parse_args(cli(config))
    if data_fingerprints(args) != metadata['data_files']:
        raise ValueError('Dataset files differ from the checkpoint training/evaluation splits')
    data = load_data(args)
    _, _, matrices, n_params, norm_mat, si_norm_mat, *_ = data
    if n_params != metadata['n_params']:
        raise ValueError('User/item ID space differs from checkpoint')
    device = torch.device('cuda:0' if cuda else 'cpu')
    model = build_model(n_params, args, norm_mat, si_norm_mat, matrices['train_sp_mat']).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    return model, args, data, metadata
