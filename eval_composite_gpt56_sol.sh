#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./activate_env.sh

# Edit the GPU list here, or pass --gpus 0,1,2,3.
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
if [[ ${1:-} == --gpus ]]; then GPUS=${2:?Missing GPU list}; shift 2; fi
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
(($# == 0)) || { echo 'Usage: bash eval_composite_gpt56_sol.sh [--gpus 0,1,2,3] [--dry-run]' >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
OUTPUT=${OUTPUT:-$PWD/runs/composite_main_gpt56_sol}
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID

# Read the 16 seen + 16 unseen task names from the official manifest.
TASKS=$(python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('robots/robocasa/eval/target50.json').read_text())
for split in ('composite_seen', 'composite_unseen'):
    for task in m['splits'][split]['tasks']:
        print(split, task)
PY
)
worker() {
    local index=$1 count=0 split task seed out active='' line
    local -a cmd
    trap '[[ -z "$active" ]] || { kill -TERM -- "-$active" 2>/dev/null || true; wait "$active" 2>/dev/null || true; }; exit 130' INT TERM
    while read -r split task; do
        for seed in {1..5}; do
            if (( count++ % ${#GPU_IDS[@]} != index )); then continue; fi
            out="$OUTPUT/$split/${task}_s$seed"
            cmd=(rpent --robot robocasa --task-name "$task" --split target --seed "$seed"
                --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device "${GPU_IDS[$index]}"
                --planner codex --model gpt-5.6-sol --reasoning-effort xhigh
                --max-turns 100 --planner-timeout-s 3600
                --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_5.5_xhigh
                --output-dir "$out")
            if ((DRY_RUN)); then printf -v line '%q ' "${cmd[@]}"; printf '%s\n' "$line"; continue; fi
            [[ ! -e "$out" ]] || { echo "Skip existing directory: $out"; continue; }
            mkdir -p "$out"
            echo "GPU ${GPU_IDS[$index]}: $task seed=$seed"
            setsid "${cmd[@]}" > "$out/launcher.log" 2>&1 &
            active=$!
            wait "$active" || echo "CLI exited nonzero: $out/launcher.log" >&2
            active=''
        done
    done <<< "$TASKS"
}
pids=()
trap 'trap "" INT TERM; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; wait || true; exit 130' INT TERM
for index in "${!GPU_IDS[@]}"; do worker "$index" & pids+=("$!"); done
wait
