"""Load a saved best model for ordinary evaluation; no training or group analysis."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='model_.ckpt or its attempt directory')
    parser.add_argument('--data-path', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ks', type=int, nargs='+', help='Optional evaluation cutoffs, e.g. 10 20 50')
    parser.add_argument('--output', type=Path, required=True, help='New JSON output path')
    opts = parser.parse_args()
    if opts.output.exists():
        parser.error('Output already exists; choose a new evaluation output path')
    if opts.gpu_id < 0 or (opts.ks and any(k < 1 for k in opts.ks)):
        parser.error('gpu-id must be nonnegative and ks must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    # Import torch only after selecting the visible physical GPU.
    from utils.checkpoint import load_for_evaluation
    from utils.evaluate import test
    from utils.experiment_log import write_json
    model, args, data, metadata = load_for_evaluation(opts.checkpoint, data_path=opts.data_path,
                                                    cuda=not opts.cpu, gpu_id=opts.gpu_id)
    if opts.ks:
        args.Ks = str(opts.ks)
    _, users, matrices, n_params, _, _, valid_pre, test_pre, _ = data
    import ast
    ks = ast.literal_eval(args.Ks)
    if max(ks) > n_params['n_items']:
        parser.error('Requested K exceeds catalog size')
    metrics = test(model, users, matrices, n_params, valid_pre, test_pre, evaluation_args=args)
    result = dict(checkpoint=str(opts.checkpoint.resolve()), epoch=metadata['best']['epoch'],
                  dataset=args.dataset, ks=ks, test=metrics, recorded_best=metadata['best'])
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(opts.output, result)
    print('Saved evaluation: ' + str(opts.output))


if __name__ == '__main__':
    main()
