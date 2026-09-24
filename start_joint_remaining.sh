#!/usr/bin/env bash
# Ali / Amazon / Yelp2018 default to GPU 9 / 10 / 11.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/start_joint_user_representation_group.sh" remaining "$@"
