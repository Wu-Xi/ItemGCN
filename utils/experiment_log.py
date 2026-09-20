"""Optional per-run records; the scheduler owns attempt directories."""
import ast
import csv
import json
import platform
import time
from pathlib import Path

import torch


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class ExperimentLog:
    def __init__(self, args):
        self.directory = Path(args.run_dir) if args.run_dir else None
        self.started = time.perf_counter()
        self.best = None
        self.ks = ast.literal_eval(args.Ks)
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            write_json(self.directory / 'config.json', vars(args))
            write_json(self.directory / 'environment.json', {
                'python': platform.python_version(), 'torch': torch.__version__,
                'platform': platform.platform(), 'cuda_version': torch.version.cuda})

    def record(self, epoch, loss, train_seconds, eval_seconds, valid, test, improved, selection_split):
        if not self.directory:
            return
        if improved:
            self.best = dict(epoch=epoch, validation=valid, test=test, selection_split=selection_split)
        row = dict(epoch=epoch, loss=loss, train_seconds=train_seconds, eval_seconds=eval_seconds)
        for split, metrics in [('validation', valid), ('test', test)]:
            for metric, values in metrics.items():
                for k, value in zip(self.ks, values):
                    row[f'{split}_{metric}@{k}'] = value
        path = self.directory / 'metrics.csv'
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def finish(self, epoch, best_score):
        if self.directory:
            if self.best is None:
                raise RuntimeError('No evaluation completed; this run cannot be marked successful')
            write_json(self.directory / 'result.json', dict(
                status='success', last_epoch=epoch, best=self.best, best_score=best_score,
                selection_k=self.ks[1], ks=self.ks, elapsed_seconds=time.perf_counter() - self.started))
