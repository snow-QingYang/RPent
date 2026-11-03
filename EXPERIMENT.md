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
RPENT_DEPTH_DEVICE can override the inference device. The composite launcher
clears that override so each worker uses its own selected CUDA device.
DA3 source revision is pinned in requirements/da3-runtime.txt.

## Evaluation

First follow the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
for the simulator, RLDX dependencies, assets, checkpoint, planner credentials,
and official task memory. Install this checkout into that environment. The
launcher activates .venv if present; otherwise it uses the active environment.
The checkpoint is expected at checkpoints/rldx-1-ft-rc365 and memory at
memory/target50/robocasa/GPT_6_astra_high.

The launcher uses GPT-6-Astra with xhigh reasoning, 100 turns, a 3600-second
planner timeout, and the original RLDX settings (40 chunks, settle patience
999, 8 actions per chunk). It evaluates 16 seen and 16 unseen tasks, each with
seeds 1–5: 160 episodes. Eight GPUs receive 20 episodes each.

Set RPENT_EVAL_CODEX_DIR to a local Codex configuration directory when needed;
it defaults to CODEX_HOME, or ~/.codex. Configure your own model provider and
authentication there. Model downloads, memory, credentials, environments and
experiment results are not included in this Git branch. The launcher uses
cached Hugging Face files in offline mode, so download them during setup.

    bash eval_composite_da3_astra.sh --dry-run
    OUTPUT=/path/with/free/space/da3 bash eval_composite_da3_astra.sh

Use --gpus 0,1,2,3 to select four cards. The DA3 launcher skips any existing
episode directory, including an incomplete one; choose a fresh OUTPUT for a
clean rerun.

## Validation

    python -m pytest tests/unit_tests/robots/robocasa \
      tests/unit_tests/rpent/tools \
      tests/unit_tests/robots/test_tool_schema_contracts.py -q

The publishing checkout passed 218 CPU unit tests. Earlier simulator/VLA/DA3
runs used the same geometry implementation; no additional GPU evaluation was
started to publish this branch.

## 中文说明

此分支把两个几何工具的数据源换成 DA3 预测深度，保留已知相机内参与位姿，
以及机器人状态、任务进度、成功和抓取接触反馈。它与 no_task_progress 独立。
先按官方文档配置 RoboCasa、RLDX、模型与记忆，再安装上面的增量 DA3 依赖。
启动脚本默认八卡、160 局；已有 episode 目录会被跳过，包括未完成目录。
