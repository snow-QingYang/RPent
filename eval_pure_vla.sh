#!/usr/bin/env bash
# Example: bash eval_pure_vla.sh --task-name ScrubCuttingBoard --seed 1 \
#   --checkpoint ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 --output-dir ./runs/smoke
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Activate the intended environment before calling, or set PURE_VLA_PYTHON.
PYTHON_BIN=${PURE_VLA_PYTHON:-python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONHASHSEED=0
unset RLDX_RESET_SEED
exec "$PYTHON_BIN" -m robots.robocasa.eval.pure_vla "$@"
