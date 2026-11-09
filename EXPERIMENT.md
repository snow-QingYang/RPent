# RoboCasa composite baseline

Branch: `baseline_composite`. The experiment code is unchanged from official
RPent commit `a744bd5f128ebef3cb8c6736cf7cd98adff681f1`.

The recorded evaluations cover the official 16 Seen and 16 Unseen composite
tasks with seeds 1–5: 160 episodes. The original batches used eight GPUs with
one episode per GPU, 100 planner turns, a 3600-second planner timeout, and
RLDX 40 chunks with 8 actions per chunk and settle patience 999.

| Planner | Memory directory under `memory/target50/robocasa/` |
| --- | --- |
| GPT-6-Astra / xhigh | `GPT_6_astra_high` |
| GPT-5.6-Sol / xhigh | `GPT_5.5_xhigh` |
| GPT-5.5 / xhigh, historical configuration | `GPT_5.5_xhigh` |

The Sol reference uses the existing GPT-5.5 memory corpus. The GPT-5.5
configuration is not evidence of a completed evaluation. For the Sol reference,
change the example's `--model` to `gpt-5.6-sol` and its memory directory to
`memory/target50/robocasa/GPT_5.5_xhigh`.

## Setup and run

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
  --output-dir ./runs/baseline_composite/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the `composite_seen` and `composite_unseen` entries in
[target50.json](robots/robocasa/eval/target50.json): 16 tasks in each group,
seeds 1–5, for 160 episodes. Both groups use CLI `--split target`; choose a
separate output directory per task and seed. `--cuda-device` selects the GPU.
Read the native outcome from each `result.json`. Keep valid task failures and
budget-exhausted episodes; retry infrastructure failures without a valid result.

Custom batch scheduling, live-join helpers and machine activation scripts are
not distributed with this branch. The CLI example does not depend on them.

## Recorded results

The latest Astra baseline completed 160 episodes: Seen 34/80, Unseen 21/80,
total 55/160 (34.38%). The separate Sol reference completed 45/160; it uses
different memory and is not a pure planner-model ablation. See the
[experiment page](https://snow-qingyang.github.io/mobile-astra/experiments/baseline/)
for source batches, detailed results and limitations.

All original geometry, task-progress, contact and success feedback remain,
including VLA early stopping on the simulator's success predicate.

## 中文说明

此分支保留原版运行代码，记录 Astra、5.6-Sol 和历史 5.5 配置。
按上方官方 CLI 示例，在自行配置并激活的环境中运行；相同任务和 seed 使用
独立输出目录。完整评测为 Seen/Unseen 各 16 个任务、每任务 seeds 1–5，共 160 局。
Astra 使用 `GPT_6_astra_high` 记忆；Sol 参考使用 `GPT_5.5_xhigh`，因此两者
不是仅改变规划器模型的消融。自定义批量调度、运行中接入和机器激活脚本不发布。
