#!/usr/bin/env bash
# Partition the complete RecDCL sweep across all specified GPUs, one job per GPU.
set -euo pipefail

usage() {
  echo 'Usage: bash start_joint_recdcl_multigpu.sh [GPU_ID ...] [options]'
  echo 'Default: GPUs 0..11; first third Ali, second third Amazon, last third Yelp2018.'
  echo 'Options: --dry-run --list-jobs --include-sym --seed N --base-config FILE --output-root DIR'
  echo 'Multi-node: --num-nodes N --node-rank R (zero-based; same GPU count per node).'
  echo 'At least 3 distinct GPUs are required. Each shard has its own resumable output.'
}

gpus=()
while [[ $# -gt 0 && "$1" != --* ]]; do
  if [[ ! "$1" =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
  gpu=$((10#$1))
  for previous in "${gpus[@]}"; do
    if [[ "$previous" == "$gpu" ]]; then echo "Duplicate GPU: $gpu" >&2; exit 2; fi
  done
  gpus+=("$gpu")
  shift
done
if [[ ${#gpus[@]} -eq 0 ]]; then gpus=(0 1 2 3 4 5 6 7 8 9 10 11); fi
if [[ ${#gpus[@]} -lt 3 ]]; then echo 'Specify at least three GPUs.' >&2; exit 2; fi

options=()
dry_run=false
list_jobs=false
run_folder=joint_user_repr_no_sym
output_root=""
num_nodes=1
node_rank=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --list-jobs) list_jobs=true; shift ;;
    --include-sym) options+=("$1"); run_folder=joint_user_repr; shift ;;
    --num-nodes|--node-rank)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then usage >&2; exit 2; fi
      if [[ "$1" == --num-nodes ]]; then num_nodes=$((10#$2)); else node_rank=$((10#$2)); fi
      shift 2 ;;
    --seed|--base-config)
      if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
      options+=("$1" "$2"); shift 2 ;;
    --output-root)
      if [[ $# -lt 2 || -z "$2" ]]; then usage >&2; exit 2; fi
      output_root="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ $num_nodes -lt 1 || $node_rank -ge $num_nodes ]]; then
  echo 'Require num-nodes >= 1 and 0 <= node-rank < num-nodes.' >&2; exit 2
fi
if $list_jobs && ! $dry_run; then echo '--list-jobs requires --dry-run.' >&2; exit 2; fi
if [[ -z "$output_root" ]]; then output_root="experiment_results/$run_folder/recdcl_multigpu"; fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
printf 'Node %s/%s; local GPUs: %s\n' "$node_rank" "$num_nodes" "${gpus[*]}"
if [[ $num_nodes -gt 1 ]]; then
  printf 'All nodes must use %s GPUs each, identical code/config/seed, and distinct node ranks.\n' "${#gpus[@]}"
fi
datasets=(ali amazon yelp2018)
job_datasets=()
job_shards=()
job_counts=()
job_outputs=()
for d in 0 1 2; do
  count=$((${#gpus[@]} / 3))
  if [[ $d -lt $((${#gpus[@]} % 3)) ]]; then count=$((count+1)); fi
  global_count=$((count * num_nodes))
  for ((shard=0; shard<count; shard++)); do
    global_shard=$((node_rank * count + shard))
    job_datasets+=("${datasets[$d]}")
    job_shards+=("$global_shard")
    job_counts+=("$global_count")
    job_outputs+=("$output_root/${datasets[$d]}/shard_${global_shard}_of_${global_count}")
  done
done

preview=(--dry-run)
if $list_jobs; then preview+=(--list-jobs); fi
# Validate every shard before submitting any background training.
for i in "${!gpus[@]}"; do
  printf 'Validate %s shard %s/%s -> GPU %s -> %s\n' \
    "${job_datasets[$i]}" "${job_shards[$i]}" "${job_counts[$i]}" "${gpus[$i]}" "${job_outputs[$i]}"
  bash "$script_dir/start_joint_user_representation.sh" "${job_datasets[$i]}" "${gpus[$i]}" \
    "${job_outputs[$i]}" --model recdcl --num-shards "${job_counts[$i]}" \
    --shard-index "${job_shards[$i]}" "${options[@]}" "${preview[@]}"
done
if $dry_run; then exit 0; fi
for i in "${!gpus[@]}"; do
  bash "$script_dir/start_joint_user_representation.sh" "${job_datasets[$i]}" "${gpus[$i]}" \
    "${job_outputs[$i]}" --model recdcl --num-shards "${job_counts[$i]}" \
    --shard-index "${job_shards[$i]}" "${options[@]}"
done
