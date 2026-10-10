#!/usr/bin/env bash
# Six workers: three datasets x two variants; nine selected models per worker.
set -euo pipefail
usage() {
  echo 'Usage: bash start_best54_checkpoints.sh [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5] [--dry-run] [--output-root DIR] [--data-path DIR]'
  echo 'Default GPUs 0 1 2 3 4 5. Order: Ali Original, Ali Item-only, Amazon Original, Amazon Item-only, Yelp Original, Yelp Item-only.'
  echo 'Set PYTHON=/path/to/python to select the training environment.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
gpus=()
while [[ $# -gt 0 && "$1" != --* ]]; do
  if [[ ! "$1" =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
  gpu=$((10#$1))
  for previous in "${gpus[@]}"; do
    if [[ "$gpu" == "$previous" ]]; then echo "Duplicate GPU: $gpu" >&2; exit 2; fi
  done
  gpus+=("$gpu"); shift
done
if [[ ${#gpus[@]} -eq 0 ]]; then gpus=(0 1 2 3 4 5); fi
if [[ ${#gpus[@]} -ne 6 ]]; then echo 'Specify exactly six GPUs.' >&2; exit 2; fi
options=()
dry_run=false
output_root=checkpoints/best54_20261010
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --output-root)
      if [[ $# -lt 2 || -z "$2" ]]; then usage >&2; exit 2; fi
      output_root="$2"; shift 2 ;;
    --data-path|--config)
      if [[ $# -lt 2 || -z "$2" ]]; then usage >&2; exit 2; fi
      options+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
python_bin="${PYTHON:-python}"
datasets=(ali amazon yelp2018)
phases=(original item_only)
for stage in validate launch; do
  for i in 0 1 2 3 4 5; do
    dataset="${datasets[$((i/2))]}"
    phase="${phases[$((i%2))]}"
    command=("$python_bin" -u run_best_checkpoints.py --dataset "$dataset" --phase "$phase"
             --gpu-id "${gpus[$i]}" --output-root "$output_root" "${options[@]}")
    if [[ "$stage" == validate ]]; then
      "${command[@]}" --dry-run
    else
      mkdir -p -- "$output_root/scheduler_logs"
      log="$output_root/scheduler_logs/${dataset}_${phase}.log"
      nohup "${command[@]}" >> "$log" 2>&1 < /dev/null &
      printf 'Started %s %s on GPU %s, PID %s, log %s\n' "$dataset" "$phase" "${gpus[$i]}" "$!" "$log"
    fi
  done
  if $dry_run; then exit 0; fi
done
