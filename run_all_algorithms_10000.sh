#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EPISODES="${1:-10000}"
STEPS="${2:-50}"

echo "Running all five algorithms for ${EPISODES} episodes and ${STEPS} steps per episode"
python3 run_final_convergence.py --episodes "${EPISODES}" --steps "${STEPS}"
