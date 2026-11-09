# RoboCasa geometry from Depth Anything 3

Branch: **da3**. Upstream experiment base: a744bd5f128ebef3cb8c6736cf7cd98adff681f1.
This branch is independent of no_task_progress.

back_project_batch and query_world_map use depth predicted from saved RGB by
DA3METRIC-LARGE, together with the recorded camera intrinsics and camera-to-world
pose. Both tools share a per-observation depth cache. Metric scaling follows
DA3's processed focal length / 300 convention; prediction failures are returned
as tool errors without falling back to simulator depth.

Camera calibration, robot state, task progress, success and grasp/contact feedback
remain available. Observation capture saves RGB and camera calibration without
generating simulator depth/world maps. Simulator depth APIs and native planner
file access remain available; this branch is not an isolation boundary.
The RLDX policy, action execution and memory are unchanged.

## Depth model

Install the additional packages into the configured RoboCasa environment:

    uv pip install --no-deps -r requirements/da3-runtime.txt

This is an incremental dependency list, not a complete environment lock. The
validated shared stack included Torch 2.7.1+cu126, torchvision 0.22.1+cu126,
NumPy 1.26.4, OpenCV 4.11.0.86, OmegaConf 2.3.0, Hugging Face Hub 0.36.2 and
moviepy 1.0.3. DA3's API imports moviepy and pycolmap even for this depth path.
The tested depth path did not require xformers, open3d or gsplat.

    hf download depth-anything/DA3METRIC-LARGE \
      --revision 4010e39f3634a45bc60553321fb49fb760bd594e \
      --local-dir ./checkpoints/DA3METRIC-LARGE

RPENT_DEPTH_MODEL_PATH overrides the checkpoint directory. In ordinary CLI use,
RPENT_DEPTH_DEVICE can override the inference device. The CLI example below
clears that override so `--cuda-device` selects the inference device.
DA3 source revision is pinned in requirements/da3-runtime.txt.

## Evaluation

Follow the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
and configure a [Codex planner connection](docs/source-en/rst_source/guides/configure_planner.rst).
Activate your prepared environment and install this checkout with
`pip install -e ".[robocasa]"`. Provide simulator assets, the checkpoint at
`checkpoints/rldx-1-ft-rc365`, and the matching Astra memory at
`memory/target50/robocasa/GPT_6_astra_high`. Download the required model files
before enabling offline mode. Credentials, environments, weights, memory and
run outputs are not included in this branch.

Run one episode from the checkout with the recorded model and budgets:

```bash
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID RPENT_DEPTH_DEVICE
export RLDX_MAX_CHUNKS=40 RLDX_ACTION_STEPS_PER_CHUNK=8 RLDX_SETTLE_PATIENCE=999
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

rpent --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local \
  --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/da3/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the `composite_seen` and `composite_unseen` entries in
[target50.json](robots/robocasa/eval/target50.json): 16 tasks in each group,
seeds 1–5, for 160 episodes. Both groups use CLI `--split target`; choose a
separate output directory per task and seed. `--cuda-device` selects the GPU.
Read the native outcome from each `result.json`. Keep valid task failures and
budget-exhausted episodes; retry infrastructure failures without a valid result.

Custom batch scheduling, live-join helpers and machine activation scripts are
not distributed with this branch. The CLI example does not depend on them.

## Validation

    python -m pytest tests/unit_tests/robots/robocasa \
      tests/unit_tests/rpent/tools \
      tests/unit_tests/robots/test_tool_schema_contracts.py -q

The publishing checkout passed 218 CPU unit tests. Earlier simulator/VLA/DA3
runs used the same geometry implementation; no additional GPU evaluation was
started to publish this branch.

## 中文说明

此分支把两个几何工具的数据源换成 DA3 预测深度，保留已知相机内参与位姿，
以及机器人状态、任务进度、成功和抓取接触反馈。它与 `no_task_progress` 独立。
先按官方文档配置 RoboCasa、RLDX、模型与记忆，再安装上面的增量 DA3 依赖。
在自行激活的环境中使用上方 CLI 命令，完整评测为 Seen/Unseen 各 16 个任务、
每任务 seeds 1–5，共 160 局；每局使用独立输出目录。命令清除
`RPENT_DEPTH_DEVICE` 覆盖值，使 `--cuda-device` 选择推理设备。
自定义批量调度、运行中接入和机器激活脚本不发布。
