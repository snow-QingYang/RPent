# RoboCasa without geometry query tools

This evaluation removes `back_project_batch` and `query_world_map` from the
planner's registered tools. Calls to either name return `unknown tool`. The
evaluation prompt and `view_env_state` description no longer direct the planner
to use pixel back-projection or world-map queries.

The implementation starts from official `RLinf/RPent` commit
`a744bd5f128ebef3cb8c6736cf7cd98adff681f1`, on branch `no_geometry_tools`.
It is independent of the DA3, NoTaskGT, and NoContactGT variants.

## Scope

- The runtime tool list contains 16 tools, with evaluation reset still disabled.
- Task progress, including numeric distances, success, contact feedback, robot
  proprioception, and all motion commands retain their baseline behavior.
- RLDX inputs, frame history, control logic, budgets, and memory selection retain
  their baseline behavior. The Astra memory corpus has 148 unchanged files.
- Simulator depth/world-map artifacts and ordinary file access are retained.
  This is an ablation of the two geometry query tools, not isolation from every
  possible geometry information source.
- The prompt changes cover evaluation. Exploration's separate instructions
  have not been adapted or validated for this setting.

## Evaluation

Follow the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
and install this checkout into the configured environment. The included
`activate_env.sh` preserves this server's checkout and EGL setup paths; update
those paths for another machine. Provide simulator assets, the RLDX checkpoint
at `checkpoints/rldx-1-ft-rc365`, and memory at
`memory/target50/robocasa/GPT_6_astra_high`. The launcher reads a private Codex
profile from `.local-setup/codex-nogeomtools/{config.toml,auth.json}`. Set up your
own provider and credentials there. These files, environments, model weights
and run outputs are excluded from Git.

`eval_composite_astra.sh` runs the official 32 composite tasks, seeds 1–5:
80 Seen and 80 Unseen episodes. It uses GPT-6-Astra with `xhigh`, 100 planner
turns, a 3600-second planner timeout, and RLDX 40 chunks × 8 action steps,
with settle patience 999. Each GPU runs one episode at a time.

```bash
OUTPUT=/path/on/result-volume/composite_nogeomtools_astra \
  bash eval_composite_astra.sh --gpus 0,1,2,3,4,5,6,7
```

The launcher preserves valid successes and valid task failures. It archives
invalid attempts before rerunning them, and stops dispatching new episodes
after an infrastructure failure. In-flight episodes can finish.

Workers share a pending-episode queue, so a GPU claims the next task after its
current episode finishes. `--gpus 0,1,2,3` selects four GPUs; the default is
0–7. `--dry-run` prints the pending commands without executing them and displays
an approximate round-robin assignment. Private worker authentication copies live
under `OUTPUT/.workers`; publish only reviewed result exports.

The launcher records dispatch state under `OUTPUT/.runner-sessions`. An
infrastructure failure creates a session `STOP` file and halts new dispatch.
The earlier server-specific job that waited for NoContactGT completion is not
part of this launcher and is not needed for an independent run.

## Completed evaluation

The 160-episode run completed on 2026-11-06 at 00:51 Hong Kong time:
Seen **22/80**, Unseen **17/80**, total **39/160 (24.38%)**. The paired Astra
baseline scored 55/160. All 160 recorded planner streams were checked; the
5,263 MCP calls contain zero calls to the two removed tools. This recorded-call
audit is not a guarantee of filesystem or native-execution isolation.

See the [experiment page](https://snow-qingyang.github.io/mobile-astra/experiments/geometry-removed/)
for the full task/seed comparison, source hashes and audit scope.

## Validation

The RoboCasa and tool-schema unit tests verify the runtime tool list, rejected
calls to removed tools, prompt consistency, and unchanged declared functions.
Launcher and queue checks exercise resume behavior, predecessor completion,
GPU selection, cancellation, and failure handling without running an experiment.
The completed smoke checked real VLA execution and recorded planner calls before
the full batch; its file-access audit is a trace check, not a sandbox.

## 中文说明

此分支移除两个主动三维查询工具及评测提示中的调用要求，保留内部深度、
任务进度、接触和成功反馈。它与 DA3、NoTaskGT、NoContactGT 分支独立。
默认八卡共享待办队列，可指定四卡；支持按整局续跑，保留有效成功和有效失败。
换机器需要调整激活脚本的本地路径，并自行配置权重、记忆、环境和认证。
