#!/usr/bin/env bash
# Two nodes, six GPUs per node, four shards per dataset; leave other GPUs free.
set -euo pipefail

usage() {
  echo 'Usage: bash start_joint_remaining_multinode.sh NODE_RANK [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5] [--dry-run]'
  echo 'NODE_RANK is 0 or 1. Default local GPUs: 0 1 2 3 4 5.'
  echo 'Options: --dry-run --list-jobs --seed N --base-config FILE --output-root DIR'
}
if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == --help ]]; then usage; exit 0; fi
rank="$1"
shift
if [[ "$rank" != 0 && "$rank" != 1 ]]; then echo 'NODE_RANK must be 0 or 1.' >&2; exit 2; fi
gpus=()
while [[ $# -gt 0 && "$1" != --* ]]; do
  if [[ ! "$1" =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
  gpu=$((10#$1))
  for previous in "${gpus[@]}"; do
    if [[ "$gpu" == "$previous" ]]; then echo "Duplicate GPU: $gpu" >&2; exit 2; fi
  done
  gpus+=("$gpu")
  shift
done
if [[ ${#gpus[@]} -eq 0 ]]; then gpus=(0 1 2 3 4 5); fi
if [[ ${#gpus[@]} -ne 6 ]]; then echo 'Specify exactly six local GPUs.' >&2; exit 2; fi
options=()
dry_run=false
list_jobs=false
output_root=experiment_results/joint_user_repr_no_sym/remaining_multinode
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --list-jobs) list_jobs=true; shift ;;
    --seed|--base-config)
      if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
      options+=("$1" "$2"); shift 2 ;;
    --output-root)
      if [[ $# -lt 2 || -z "$2" ]]; then usage >&2; exit 2; fi
      output_root="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if $list_jobs && ! $dry_run; then echo '--list-jobs requires --dry-run.' >&2; exit 2; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
datasets=(ali amazon yelp2018)
models=(rns mixgcf directau graphau simgcl)
printf 'Remaining models: %s; node %s/2; GPUs: %s\n' "${models[*]}" "$rank" "${gpus[*]}"
preview=(--dry-run)
if $list_jobs; then preview+=(--list-jobs); fi
# Validate all six shards before submitting background schedulers.
for stage in validate launch; do
  for i in 0 1 2 3 4 5; do
    dataset="${datasets[$((i/2))]}"
    shard=$((rank*2 + i%2))
    output="$output_root/$dataset/shard_${shard}_of_4"
    args=("$dataset" "${gpus[$i]}" "$output" --models "${models[@]}"
          --num-shards 4 --shard-index "$shard" "${options[@]}")
    if [[ "$stage" == validate ]]; then
      printf 'Validate %s shard %s/4 -> GPU %s -> %s\n' "$dataset" "$shard" "${gpus[$i]}" "$output"
      args+=("${preview[@]}")
    fi
    bash "$script_dir/start_joint_user_representation.sh" "${args[@]}"
  done
  if $dry_run; then exit 0; fi
done
