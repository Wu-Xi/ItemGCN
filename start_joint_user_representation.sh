#!/usr/bin/env bash
# One independent background scheduler per dataset/GPU. Run with bash on Linux.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo 'Usage: bash start_joint_user_representation.sh DATASET GPU_ID [OUTPUT_DIR] [Python script options...]' >&2
  echo 'Example: bash start_joint_user_representation.sh ali 0' >&2
  echo 'Preview: bash start_joint_user_representation.sh ali 0 --dry-run' >&2
  exit 2
fi

dataset="$1"
gpu="$2"
shift 2
if [[ ! "$dataset" =~ ^[a-zA-Z0-9_-]+$ || ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo 'Dataset must be an identifier; GPU_ID must be a nonnegative integer.' >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
run_folder="joint_user_repr_no_sym"
for option in "$@"; do
  if [[ "$option" == --include-sym ]]; then
    run_folder="joint_user_repr"
  fi
done
output="experiment_results/${run_folder}/${dataset}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  output="$1"
  shift
fi
python_bin="${PYTHON:-python}"
command -v "$python_bin" >/dev/null
command_args=("$python_bin" -u "$script_dir/run_joint_user_representation.py"
              --dataset "$dataset" --gpu_id "$gpu" --output "$output" "$@")

# Dry-run remains foreground and creates neither output directories nor log files.
for option in "$@"; do
  if [[ "$option" == --dry-run ]]; then
    exec "${command_args[@]}"
  fi
done

mkdir -p -- "$output"
nohup "${command_args[@]}" >> "$output/scheduler.log" 2>&1 < /dev/null &
pid=$!
printf 'Started %s on GPU %s; scheduler PID=%s\n' "$dataset" "$gpu" "$pid"
printf 'Log: %s/scheduler.log\n' "$output"
printf 'Resume with the same command; successful jobs are skipped.\n'
