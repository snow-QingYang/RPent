#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./activate_env.sh

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
if [[ ${1:-} == --gpus ]]; then GPUS=${2:?Missing GPU list}; shift 2; fi
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
(($# == 0)) || { echo 'Usage: bash eval_composite_astra.sh [--gpus 0,1,2,3] [--dry-run]' >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
OUTPUT=${OUTPUT:-$PWD/runs/composite_nocontactgt_astra}
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID RPENT_DEPTH_DEVICE
CONFIG_SOURCE="$PWD/.local-setup/codex-nocontactgt"

python - "$GPUS" <<'PY'
import sys
ids = sys.argv[1].split(',')
if not ids or any(not x.isdigit() for x in ids) or len(set(map(int, ids))) != len(ids):
    raise SystemExit('GPU indices must be distinct nonnegative integers.')
PY
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
    mkdir -p "$OUTPUT/.runner-sessions"
    SESSION_DIR=$(mktemp -d "$OUTPUT/.runner-sessions/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    STOP_FILE="$SESSION_DIR/STOP"
fi

TASKS=$(python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
m = json.loads(Path('robots/robocasa/eval/target50.json').read_text())
for split in ('composite_seen', 'composite_unseen'):
    for task in m['splits'][split]['tasks']:
        for seed in m['splits'][split]['seeds']:
            path = output / split / f'{task}_s{seed}' / 'result.json'
            try:
                result = json.loads(path.read_text())
            except (OSError, ValueError):
                result = None
            if isinstance(result, dict) and result.get('valid') is True:
                continue
            print(split, task, seed)
PY
)

stop_run() {
    flock 8
    printf '%s\n' "$*" >> "$STOP_FILE"
    flock -u 8
}

worker() {
    local index=$1 count=0 split task seed out active='' line code gpu archive
    local -a cmd
    gpu=${GPU_IDS[$index]}
    trap '[[ -z "$active" ]] || { kill -TERM -- "-$active" 2>/dev/null || true; wait "$active" 2>/dev/null || true; }; exit 130' INT TERM
    if ((!DRY_RUN)); then
        exec 8>"$SESSION_DIR/dispatch.lock"
        trap 'code=$?; if ((code != 0)); then stop_run "Worker exited with status $code" || true; fi' EXIT
        export CODEX_HOME="$OUTPUT/.workers/gpu_$gpu/codex"
        export TMPDIR="$OUTPUT/.workers/gpu_$gpu/tmp"
        mkdir -p -m 700 "$CODEX_HOME" "$TMPDIR"
        install -m 600 "$CONFIG_SOURCE/config.toml" "$CODEX_HOME/config.toml"
        install -m 600 "$CONFIG_SOURCE/auth.json" "$CODEX_HOME/auth.json"
    fi
    while read -r split task seed; do
        [[ -n "$split" ]] || continue
        if ((count++ % ${#GPU_IDS[@]} != index)); then continue; fi
        out="$OUTPUT/$split/${task}_s$seed"
        cmd=(python -m rpent.cli.main --robot robocasa --task-name "$task" --split target --seed "$seed"
            --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device "$gpu"
            --planner codex --model gpt-6-astra --reasoning-effort xhigh
            --max-turns 100 --planner-timeout-s 3600
            --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_6_astra_high
            --output-dir "$out")
        if ((DRY_RUN)); then printf -v line '%q ' "${cmd[@]}"; printf '%s\n' "$line"; continue; fi
        # Serialize task starts with STOP creation; running episodes finish normally.
        flock 8
        if [[ -e "$STOP_FILE" ]]; then flock -u 8; break; fi
        if [[ -e "$out" ]]; then
            mkdir -p "$OUTPUT/.interrupted"
            archive=$(mktemp -d "$OUTPUT/.interrupted/${split}_${task}_s${seed}_XXXXXX")
            mv "$out" "$archive/${task}_s$seed"
            echo "Archived previous attempt: $archive/${task}_s$seed"
        fi
        mkdir -p "$out"
        echo "GPU $gpu: $task seed=$seed"
        setsid "${cmd[@]}" 8>&- 9>&- > "$out/launcher.log" 2>&1 &
        active=$!
        flock -u 8
        code=0
        wait "$active" || code=$?
        active=''
        echo "$code" > "$out/launcher_exit_code.txt"
        echo "GPU $gpu: $task seed=$seed exit=$code"
        if ! python - "$out/result.json" "$code" <<'PY'
import json
import sys
from pathlib import Path

try:
    result = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    result = None
valid = (
    isinstance(result, dict)
    and result.get('valid') is True
    and result.get('termination_reason') != 'infrastructure_error'
    and not result.get('infrastructure_error')
)
# The CLI exits 1 for a planner timeout even when the native environment
# result is valid. Keep that budget-exhausted episode and dispatch the next.
exit_code = int(sys.argv[2])
expected_exit = exit_code == 0 or (
    exit_code == 1 and valid and result.get('termination_reason') == 'planner_timeout'
)
raise SystemExit(0 if valid and expected_exit else 1)
PY
        then
            stop_run "GPU $gpu: $task seed=$seed exit=$code; invalid/missing result or unexpected exit"
            echo "Stopping new tasks after unsuccessful execution: $out" >&2
            return 1
        fi
    done <<< "$TASKS"
}
pids=()
trap 'trap "" INT TERM; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; wait || true; exit 130' INT TERM
for index in "${!GPU_IDS[@]}"; do worker "$index" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((!DRY_RUN)) && [[ -e "$STOP_FILE" ]]; then status=1; fi
exit "$status"
