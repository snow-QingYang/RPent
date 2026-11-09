# RoboCasa composite: No task progress

Branch: **no_task_progress**. Upstream experiment base:
0e406bb96d670d0c18a4cea9a6ea047119661d19. This branch is independent of da3.

view_env_state omits task_progress from Agent-facing observations. Action tools
use that same observation path. Evaluation and exploration prompts no longer
instruct the Agent to inspect task_progress. The simulator and internal episode
records retain the original task-progress computation.

read_text_file accepts only files under the configured memory directory that
also pass the existing per-run memory policy. Reading success_criteria.md,
states.json or simulator source through this tool is denied. This change is in
the shared CommonTools class and applies to other robots on this branch too.

## Information still available

- success, robocasa_terminated and the RLDX success status / early-stop behavior.
- Grasp/contact feedback, including grasp_contact, grasp_obj and grasped.
- Original simulator geometry, camera calibration and robot state; DA3 is not used.
- Original task and global memory, which can contain historical predicate names.

This is a tool-level observation/file-read restriction, not an operating-system
sandbox. Native planner tools are not disabled. Use evaluation mode for this
experiment: exploration still requests success criteria in its workflow, which
conflicts with the restricted file reader and has not been validated here.

## Evaluation revision

The completed E03 evaluation used runtime commit
`b7333814fcb1178442057b231e267d4dcfba4554`, archived on `no_task_progress_eval`.
That version generated `success_criteria.md`; all 160 recorded episodes retain
this file. Its `read_text_file` was already restricted to selected memory,
while native planner shell access remained available.

This branch includes the later change
`26d8665d3d394dc77ce90a68d266d370d2384f98`, which stops generating
`success_criteria.md`. The published E03 results do not evaluate that later
change. Runtime and test files are unchanged from that commit by this
publication cleanup.

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
  --output-dir ./runs/no_task_progress/composite_seen/ScrubCuttingBoard_s1
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
      tests/unit_tests/rpent/memory tests/unit_tests/rpent/tools \
      tests/unit_tests/robots/test_tool_schema_contracts.py -q

The publishing checkout passed 250 CPU unit tests. Before publication, a real
SetUpCuttingStation smoke run produced 23 observations and 48 completed tool
calls with no live task_progress fields in Agent-facing tool results; the
native episode outcome was failure. The historical 160-episode evaluation used runtime commit
`b7333814fcb1178442057b231e267d4dcfba4554`.

## 中文说明

此分支屏蔽 Agent 观测中的 `task_progress`，并将 `read_text_file` 限制在本次
允许读取的记忆内。success、终止标志、抓取接触、官方几何和原有记忆仍保留。
限制位于工具层，不是系统沙箱；当前实验使用 evaluation，不使用 exploration。
完整 E03 评测使用 `b7333814fcb1178442057b231e267d4dcfba4554`，归档于
`no_task_progress_eval`；该版本仍生成 `success_criteria.md`，160 局均保留此文件。
文件工具不能读取它，但原生 shell 仍可访问。后续 `no_task_progress` 的
`26d8665` 才停止生成此文件，已发布的 E03 结果不代表后续版本的评测。
按上方官方 CLI 示例在自行配置的环境中运行，每局使用独立输出目录。
自定义批量调度、运行中接入和机器激活脚本不发布。
