#!/usr/bin/env bash
# Source this file to use this checkout with the installed RoboCasa dependencies.
export RPENT_REPO_ROOT=/mnt/sdd/cvmi-baoqy/mobile_astra/HarnessVLA-RPent-NoGeomTools
unset PYTHONHOME TRANSFORMERS_CACHE RLDX_RESET_SEED
export PYTHONPATH="$RPENT_REPO_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "$RPENT_REPO_ROOT/.venv/bin/activate"
export ROBOCASA_ASSETS_PATH="$RPENT_REPO_ROOT/resources/assets"
export RLDX_MODEL_PATH="$RPENT_REPO_ROOT/checkpoints/rldx-1-ft-rc365"
export HF_HOME="$RPENT_REPO_ROOT/.local-setup/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export NUMBA_CACHE_DIR="$RPENT_REPO_ROOT/.local-setup/numba-cache"
export NO_ALBUMENTATIONS_UPDATE=1
source /mnt/sda/cvmi-baoqy/mobile_astra_environments_20261028/egl/env.sh
export TRITON_LIBCUDA_PATH="$RPENT_REPO_ROOT/.local-setup/cuda"
export TRITON_CACHE_DIR="$RPENT_REPO_ROOT/.local-setup/triton-cache"
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"
