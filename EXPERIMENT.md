# RoboCasa evaluation without task_progress observations

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

    bash eval_composite_astra.sh --dry-run
    OUTPUT=/path/with/at/least/30GiB/no_task_progress bash eval_composite_astra.sh

Use --gpus 0,1,2,3 to select four cards. Each worker copies the local Codex
config.toml and auth.json into its own private directory under OUTPUT/.workers;
that directory contains credentials and must not be published.

The launcher locks OUTPUT against duplicate runners. It keeps any episode with
result.json, including an invalid result, and moves unfinished directories to
.interrupted before restarting those episodes. Resume granularity is one
episode, not an in-progress simulator state.

## Validation

    python -m pytest tests/unit_tests/robots/robocasa \
      tests/unit_tests/rpent/memory tests/unit_tests/rpent/tools \
      tests/unit_tests/robots/test_tool_schema_contracts.py -q

The publishing checkout passed 250 CPU unit tests. Before publication, a real
SetUpCuttingStation smoke run produced 23 observations and 48 completed tool
calls with no live task_progress fields in Agent-facing tool results; the
native episode outcome was failure. The subsequent 160-episode evaluation
uses this same implementation.

## 中文说明

此分支屏蔽 Agent 观测中的 task_progress，并将 read_text_file 限制在本次允许
读取的 memory 内。success、终止标志、抓取接触、官方几何和原有记忆仍保留。
限制位于工具层，不是系统沙箱；当前实验使用 evaluation，不使用 exploration。
启动脚本默认八卡、160 局，要求输出盘至少有 30 GiB 可用空间。可按整局续跑；
已有 result.json 的局会保留，包括 invalid 结果。OUTPUT/.workers 含认证信息，
不要上传该目录。
