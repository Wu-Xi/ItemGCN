#!/usr/bin/env bash
# Shared implementation for the four model-group entry points.
set -euo pipefail

usage() {
  echo 'Usage: bash start_joint_<group>.sh [GPU_ALI GPU_AMAZON GPU_YELP2018] [options]'
  echo 'Options: --dry-run --list-jobs --include-sym --seed N --base-config FILE --output-root DIR'
}

if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
group="$1"
shift
case "$group" in
  recdcl)      models=(recdcl); gpus=(0 1 2) ;;
  sgl)         models=(sgl); gpus=(3 4 5) ;;
  stabcf_ahns) models=(stabcf ahns); gpus=(6 7 8) ;;
  remaining)   models=(rns mixgcf directau graphau simgcl); gpus=(9 10 11) ;;
  *) echo "Unknown model group: $group" >&2; exit 2 ;;
esac
if [[ $# -gt 0 && "$1" != --* ]]; then
  if [[ $# -lt 3 ]]; then usage >&2; exit 2; fi
  gpus=("$1" "$2" "$3")
  shift 3
fi
for index in 0 1 2; do
  if [[ ! "${gpus[$index]}" =~ ^[0-9]+$ ]]; then
    echo 'GPU IDs must be nonnegative integers.' >&2; exit 2
  fi
  gpus[$index]=$((10#${gpus[$index]}))
done
if [[ "${gpus[0]}" == "${gpus[1]}" || "${gpus[0]}" == "${gpus[2]}" || "${gpus[1]}" == "${gpus[2]}" ]]; then
  echo 'Use three distinct GPUs for the three dataset processes.' >&2; exit 2
fi

options=()
dry_run=false
list_jobs=false
run_folder="joint_user_repr_no_sym"
output_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --list-jobs) list_jobs=true; shift ;;
    --include-sym) options+=("$1"); run_folder="joint_user_repr"; shift ;;
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
if $list_jobs && ! $dry_run; then
  echo '--list-jobs requires --dry-run.' >&2; exit 2
fi
if [[ -z "$output_root" ]]; then output_root="experiment_results/$run_folder"; fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
datasets=(ali amazon yelp2018)
preview_options=(--dry-run)
if $list_jobs; then preview_options+=(--list-jobs); fi
printf 'Group: %s; models: %s\n' "$group" "${models[*]}"

# Validate every dataset before submitting any background process.
for index in 0 1 2; do
  dataset="${datasets[$index]}"
  output="$output_root/$group/$dataset"
  printf 'Validate %s -> GPU %s -> %s\n' "$dataset" "${gpus[$index]}" "$output"
  bash "$script_dir/start_joint_user_representation.sh" "$dataset" "${gpus[$index]}" "$output" \
    --models "${models[@]}" "${options[@]}" "${preview_options[@]}"
done
if $dry_run; then exit 0; fi

for index in 0 1 2; do
  dataset="${datasets[$index]}"
  bash "$script_dir/start_joint_user_representation.sh" "$dataset" "${gpus[$index]}" \
    "$output_root/$group/$dataset" --models "${models[@]}" "${options[@]}"
done
