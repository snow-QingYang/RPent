# RoboCasa composite baseline launchers

Branch: `baseline_composite`. The experiment code is unchanged from official
RPent commit `a744bd5f128ebef3cb8c6736cf7cd98adff681f1`.

The launchers evaluate the official 16 Seen and 16 Unseen composite tasks with
seeds 1–5: 160 episodes per run. Defaults are eight GPUs, one episode per GPU,
100 planner turns, a 3600-second planner timeout, and RLDX 40 chunks with
8 actions per chunk and settle patience 999.

| Script | Planner | Memory directory under `memory/target50/robocasa/` |
| --- | --- | --- |
| `eval_composite.sh` | GPT-6-Astra / xhigh | `GPT_6_astra_high` |
| `eval_composite_gpt56_sol.sh` | GPT-5.6-Sol / xhigh | `GPT_5.5_xhigh` |
| `eval_composite_gpt55.sh` | GPT-5.5 / xhigh, historical launcher | `GPT_5.5_xhigh` |

The Sol launcher uses the existing GPT-5.5 memory corpus. The GPT-5.5 script
is retained as a launch configuration, not evidence of a completed evaluation.

## Setup and run

Follow the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
for dependencies, simulator assets, the RLDX checkpoint, task memory and planner
authentication. Install this checkout in the environment used by the launcher.

`activate_env.sh` is the original server-specific activation script. On a new
machine, replace its source path with your own environment activation/configuration.
The `.local-setup/env.sh` file it references is private and is not distributed.
Place the checkpoint at `checkpoints/rldx-1-ft-rc365` and the memory at the paths
listed above. Download dependencies before running: the launchers enable offline
Hugging Face mode. Environments, credentials, weights and results are excluded.

```bash
bash eval_composite.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
OUTPUT=/path/to/new/baseline-run bash eval_composite.sh --gpus 0,1,2,3,4,5,6,7
```

Use `--gpus 0,1,2,3` for four GPUs. Work is assigned to fixed GPU workers.
Existing episode directories are skipped, including incomplete ones; use a fresh
output directory for a clean rerun. Resuming does not restore an in-progress
simulator state. The launchers do not implement the shared queue in the later
`no_geometry_tools` branch.

## Recorded results

The latest Astra baseline completed 160 episodes: Seen 34/80, Unseen 21/80,
total 55/160 (34.38%). The separate Sol reference completed 45/160; it uses
different memory and is not a pure planner-model ablation. See the
[experiment page](https://snow-qingyang.github.io/mobile-astra/experiments/baseline/)
for source batches, detailed results and limitations.

All original geometry, task-progress, contact and success feedback remain,
including VLA early stopping on the simulator's success predicate.

## 中文说明

此分支保留原版实验代码，并归档 Astra、5.6-Sol 和历史 5.5 启动脚本。
默认八卡、32 个 composite 任务各 5 个 seed；支持指定四卡。
换机器需要调整 `activate_env.sh`，自行配置环境、资产、权重、记忆和认证。
脚本跳过所有已有 episode 目录，包括未完成的局；完整重跑请使用新的输出目录。
