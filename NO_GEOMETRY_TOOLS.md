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
  --output-dir ./runs/no_geometry_tools/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the `composite_seen` and `composite_unseen` entries in
[target50.json](robots/robocasa/eval/target50.json): 16 tasks in each group,
seeds 1–5, for 160 episodes. Both groups use CLI `--split target`; choose a
separate output directory per task and seed. `--cuda-device` selects the GPU.
Read the native outcome from each `result.json`. Keep valid task failures and
budget-exhausted episodes; retry infrastructure failures without a valid result.

Custom batch scheduling, live-join helpers and machine activation scripts are
not distributed with this branch. The CLI example does not depend on them.

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
The completed smoke checked real VLA execution and recorded planner calls before
the full batch; its file-access audit is a trace check, not a sandbox.

## 中文说明

此分支移除两个主动三维查询工具及评测提示中的调用要求，保留内部深度、
任务进度、接触和成功反馈。它与 DA3、NoTaskGT、NoContactGT 分支独立。
按上方官方 CLI 示例，在自行配置并激活的环境中运行。完整评测为 Seen/Unseen
各 16 个任务、每任务 seeds 1–5，共 160 局；每局使用独立输出目录。
保留有效成功和有效失败结果；仅重试无有效结果的基础设施故障。
自定义批量调度、运行中接入和机器激活脚本不发布。
