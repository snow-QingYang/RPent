#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export RPENT_REPO_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
if [[ ${1:-} == --gpus ]]; then GPUS=${2:?Missing GPU list}; shift 2; fi
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
(($# == 0)) || { echo 'Usage: bash eval_composite_astra.sh [--gpus 0,1,2,3] [--dry-run]' >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
OUTPUT=${OUTPUT:-$PWD/runs/composite_notaskgt_astra}
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID RPENT_DEPTH_DEVICE
CONFIG_SOURCE=${RPENT_EVAL_CODEX_DIR:-${CODEX_HOME:-$HOME/.codex}}

python - "$GPUS" <<'PY'
import sys
ids = sys.argv[1].split(',')
if not ids or any(not x.isdigit() for x in ids) or len(set(map(int, ids))) != len(ids):
    raise SystemExit('GPU indices must be distinct nonnegative integers.')
PY
TASKS=$(python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('robots/robocasa/eval/target50.json').read_text())
for split in ('composite_seen', 'composite_unseen'):
    for task in m['splits'][split]['tasks']:
        for seed in m['splits'][split]['seeds']:
            print(split, task, seed)
PY
)

if ((!DRY_RUN)); then
    mkdir -p "$OUTPUT"
    OUTPUT=$(realpath "$OUTPUT")
    exec 9>"$OUTPUT/.runner.lock"
    flock -n 9 || { echo "Another runner owns $OUTPUT" >&2; exit 1; }
    python - "$OUTPUT" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 2**30
if free < 30:
    raise SystemExit(f'Output disk has {free:.1f} GiB free; provide an OUTPUT path with at least 30 GiB.')
PY
    [[ -f "$CONFIG_SOURCE/config.toml" && -f "$CONFIG_SOURCE/auth.json" ]]
fi

worker() {
    local index=$1 count=0 split task seed out active='' line code gpu
    local -a cmd
    gpu=${GPU_IDS[$index]}
    trap '[[ -z "$active" ]] || { kill -TERM -- "-$active" 2>/dev/null || true; wait "$active" 2>/dev/null || true; }; exit 130' INT TERM
    if ((!DRY_RUN)); then
        export CODEX_HOME="$OUTPUT/.workers/gpu_$gpu/codex"
        export TMPDIR="$OUTPUT/.workers/gpu_$gpu/tmp"
        mkdir -p -m 700 "$CODEX_HOME" "$TMPDIR"
        install -m 600 "$CONFIG_SOURCE/config.toml" "$CODEX_HOME/config.toml"
        install -m 600 "$CONFIG_SOURCE/auth.json" "$CODEX_HOME/auth.json"
    fi
    while read -r split task seed; do
        if ((count++ % ${#GPU_IDS[@]} != index)); then continue; fi
        out="$OUTPUT/$split/${task}_s$seed"
        cmd=(rpent --robot robocasa --task-name "$task" --split target --seed "$seed"
            --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device "$gpu"
            --planner codex --model gpt-6-astra --reasoning-effort xhigh
            --max-turns 100 --planner-timeout-s 3600
            --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_6_astra_high
            --output-dir "$out")
        if ((DRY_RUN)); then printf -v line '%q ' "${cmd[@]}"; printf '%s\n' "$line"; continue; fi
        if [[ -f "$out/result.json" ]]; then echo "Keep recorded result: $out"; continue; fi
        if [[ -e "$out" ]]; then
            mkdir -p "$OUTPUT/.interrupted/$split"
            mv "$out" "$OUTPUT/.interrupted/$split/${task}_s${seed}_$(date -u +%Y%m%dT%H%M%SZ)_$BASHPID"
        fi
        mkdir -p "$out"
        echo "GPU $gpu: $task seed=$seed"
        setsid "${cmd[@]}" > "$out/launcher.log" 2>&1 &
        active=$!
        code=0
        wait "$active" || code=$?
        active=''
        echo "$code" > "$out/launcher_exit_code.txt"
        echo "GPU $gpu: $task seed=$seed exit=$code"
    done <<< "$TASKS"
}
pids=()
trap 'trap "" INT TERM; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; wait || true; exit 130' INT TERM
for index in "${!GPU_IDS[@]}"; do worker "$index" & pids+=("$!"); done
wait
