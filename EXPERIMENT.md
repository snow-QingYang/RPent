# RoboCasa composite without historical memory

Branch `no_memory` evaluates the official RPent baseline without historical
task or global memory. Its base is
`a744bd5f128ebef3cb8c6736cf7cd98adff681f1`. It is independent of the geometry,
contact, task-progress and completion-feedback ablations.

## Scope

- The evaluation prompt states that historical memory is unavailable and does
  not list memory files.
- The memory loader returns an empty selection. RPent file tools deny access
  to historical memory under the configured corpus root, including symlink
  aliases. Result metadata records `policy="none"`, empty selected/read files
  and the memory audit status.
- Corpus download and internal validation remain unchanged. A valid memory
  corpus is still required at startup; its contents are not supplied to the
  Agent. This is an evaluation branch, not an exploration-memory redesign.
- Current-episode dialogue, observations and action history remain. Geometry
  queries, contact feedback, `task_progress`, success signals, success-rule
  source access, VLA inputs/control and native-success early stopping retain
  baseline behavior.

The file-tool boundary is not an operating-system sandbox. Native planner shell
access remains available; absence of Agent memory reads must also be checked in
recorded trajectories. Source code and current-episode artifact reads remain
allowed.

Production changes are limited to `robots/robocasa/memory.py`,
`robots/robocasa/prompts/evaluate.py` and
`robots/robocasa/eval/validate_target50.py`.

## Run an episode

Follow the [RoboCasa installation guide](docs/source-en/rst_source/simulators/robocasa.rst)
for this checkout and install `.[robocasa]`. Configure the
[Codex planner](docs/source-en/rst_source/guides/configure_planner.rst) with
access to `gpt-6-astra`. Download simulator assets, the RLDX-1-FT-RC365 checkpoint,
its base-model support files and the official `GPT_6_astra_high` memory corpus.
The memory corpus remains a startup dependency even though its Agent-visible
selection is empty. Resources, credentials and local scheduler scripts are not
included in this branch.

From this checkout in the prepared environment:

```bash
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID RPENT_DEPTH_DEVICE
export RLDX_MAX_CHUNKS=40 RLDX_ACTION_STEPS_PER_CHUNK=8 RLDX_SETTLE_PATIENCE=999
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Enable offline mode after downloading all required resources.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

rpent --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local \
  --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/no_memory/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the 32 tasks and seeds 1–5 in
[the experiment manifest](experiments/no_memory_composite.json). Seen and Unseen
are evaluation groups; both use CLI `--split target`. Store each episode under
`composite_seen/<Task>_s<seed>` or `composite_unseen/<Task>_s<seed>`.
Validate the completed batch with its matching manifest:

```bash
python -m robots.robocasa.eval.validate_target50 ./runs/no_memory \
  --manifest ./experiments/no_memory_composite.json
```

## Validation and completed experiment

The focused offline contract suite covers empty selection, file access and
aliases, retained current-episode access, rendered prompts and result validation:

```bash
pytest tests/unit_tests/robots/robocasa/test_no_memory_contracts.py -q
```

All 10 cases passed. The inherited upstream tests that require a nonempty memory
selection still describe the baseline setting; they have not been weakened or
rewritten to claim that the full upstream test suite passes on this ablation.

A real `StackBowlsCabinet`, target seed 1, smoke run completed successfully and
called `finish`, with no historical memory reads. It is excluded from the formal
160-episode batch, which completed on 2026-11-09:

| Split | NoMemory | Baseline |
| --- | ---: | ---: |
| Seen | 24/80 | 34/80 |
| Unseen | 17/80 | 21/80 |
| Total | 41/160 | 55/160 |

All 160 results pass the NoMemory validator. An audit of all 160 recorded
trajectories, including 364 native shell commands across 84 episodes, found no
Agent attempts to read historical task/global memory. Source/AGENTS and
current-episode artifact reads did occur. This audit does not establish OS
isolation or zero internal corpus reads by the harness.

Results, paired outcomes, initial-state comparison limits and source hashes are
published on [Mobile Astra](https://snow-qingyang.github.io/mobile-astra/experiments/no-memory/).
The runtime source files in this branch match the recorded experiment snapshot;
the manifest, documentation and test accompany that code for reproduction.

## 中文说明

`no_memory` 从官方 baseline 移除历史任务／全局记忆。Prompt 不再列出记忆文件，
记忆选择为空，文件工具禁止读取对应记忆目录。下载和内部校验保留，因此仍需准备
有效的官方记忆语料，但内容不会提供给 Agent。本局交互历史、几何、接触、任务进度、
成功反馈、成功规则源码及 VLA 原生成功早停均保留，没有叠加其他去真值实验。

按[中文环境说明](docs/source-zh/rst_source/simulators/robocasa.rst)准备依赖和资源，
使用上方 CLI 与实验 manifest。32 个 composite 任务、每任务 seeds 1–5，共 160 局；
Seen / Unseen 均使用 `--split target`。完整结果为 Seen 24/80、Unseen 17/80，
合计 41/160。10 项离线测试和真实单局冒烟通过；全量轨迹未观察到 Agent 读取历史
记忆，但发现了允许的源码与本局文件读取。文件工具限制不等于系统沙箱。
