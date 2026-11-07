# RoboCasa365: Agent Check Success

Branch `agent_check_success` builds on `no_task_progress` commit `26d8665`.
The outer planner judges completion from task language and observations and
reports its decision through `finish`. RLDX remains an action policy; it has no
learned termination output.

## Behavior and Scoring

- Native task success no longer stops an RLDX call early or returns
  `status="success"`. Action limits, settling behavior, frame history, robot
  state, geometry and contact feedback remain unchanged.
- Initial, post-action and historical tool observations omit live `success`,
  `robocasa_terminated` and `task_progress`, including replayed action results.
  Primitive execution `ok` remains available. Native predicates still execute
  privately at their original cadence because they can update task bookkeeping.
- The inherited NoTaskGT behavior retains private `task_progress` in saved
  states, restricts `read_text_file` to selected memory, and does not generate
  `success_criteria.md`. Historical memory can still contain success-rule
  descriptions; this experiment removes live task feedback, not prior knowledge.
- After the planner returns, the evaluator reads and caches the native final
  verdict. `result.json` stores `success` with
  `success_source="env.check_success_after_planner"`, separately from
  `agent_finish.called`, `agent_finish.status` and `agent_finish.summary`.
  This measures the final state, not whether the task ever succeeded or the
  state at the instant of `finish`.
- `termination_reason` distinguishes `agent_finish`,
  `agent_return_without_finish`, timeouts, truncated responses and infrastructure
  errors. A native success cannot conceal a planner error. Missing `finish`
  does not by itself invalidate an available native score.

The serialized identifier remains `experiment_variant="agent_stop"` for
compatibility with the evaluation already in progress. The branch validator
requires this identifier, the score source and consistent finish evidence;
it rejects baseline records. `agent_check_success` is the experiment and branch
name, not a new result-schema value.

This branch is **evaluation only**. Exploration guards and recipe export have
not been adapted. The existing `finish` handler and shared planner flow are
unchanged: the prompt asks for no further robot commands after `finish`, but
there is no additional actuator guard. File-tool restrictions are not an
operating-system sandbox; native planner shell access remains available. This
branch retains simulator geometry and contacts and does not combine the DA3,
NoContactGT or NoGeomTools ablations.

## Run an Episode

Prepare an environment using the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
and install this checkout with `pip install -e ".[robocasa]"`. Configure a
working [Codex planner connection](docs/source-en/rst_source/guides/configure_planner.rst)
with access to `gpt-6-astra`. Download the simulator assets, RLDX-1-FT-RC365
checkpoint and required base-model support files before enabling offline mode.
Place the checkpoint at `checkpoints/rldx-1-ft-rc365` and the matching Astra task
memory at `memory/target50/robocasa/GPT_6_astra_high`. These resources and private
provider credentials are not included in the branch.

Run from this checkout in the prepared environment. This example uses the
experiment's model, memory selection and action/time budgets:

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
  --output-dir ./runs/agent_check_success/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the `composite_seen` and `composite_unseen` task/seed entries in
[target50.json](robots/robocasa/eval/target50.json): 16 Seen and 16 Unseen tasks,
each with seeds 1–5, for 160 episodes. Both evaluation groups use CLI
`--split target`; give each task/seed a separate output directory. Read native
success and the agent's declaration separately in each `result.json`.
Custom batch scheduling, live-join helpers and server activation scripts are
not part of this published branch; the CLI command does not require them.

## Validation Evidence

The relevant offline CPU suite passed **270 tests** before publication,
including native-predicate cadence, observation filtering, RLDX stopping and
result/validator contracts. Focused lint and formatting checks passed.

A real GPT-6-Astra/xhigh smoke run on `ScrubCuttingBoard`, target seed 1,
passed the execution-chain and transcript audit. **The task itself failed:**
the agent called `finish(status="success")`, while the evaluator recorded
`success=false`, `valid=true`. The run took 350.8 seconds and included 9 tool
calls, 4 observations and 3 RLDX calls. Each RLDX call reached 40 chunks / 320
actions with `status="cap"`, for 960 actions total. Audited tool outputs omitted
live task-success/progress fields; there were no native shell calls or actions
after `finish` in this run. This validates the execution and scoring separation,
not benchmark success or a sandbox guarantee.

## 中文说明

`agent_check_success` 分支基于 `no_task_progress` 的 `26d8665`。外层规划器根据
任务文字和观测判断是否完成，并调用 `finish` 报告判断；RLDX 仍是动作策略，
没有学习得到的终止输出。

取消原生成功判据触发的 RLDX 提前结束和 `status="success"`，保留动作预算、
静止判据、帧历史、机器人状态、几何及接触反馈。初始、动作后和历史工具观测
不提供实时 `success`、`robocasa_terminated`、`task_progress`，回放动作结果也会
过滤这些字段；动作执行的 `ok` 仍可见。原生判据按原节奏在内部执行，以保留
可能依赖这些调用更新的任务状态。

继承的 NoTaskGT 设置仍在内部保存 `task_progress`，将 `read_text_file` 限制于
选中的记忆文件，并且不生成 `success_criteria.md`。历史记忆可能包含成功规则的
描述，因此本实验移除的是实时任务反馈，不是已有任务知识。

规划器返回后，评测器独立读取并缓存原生最终判定，在 `result.json` 中保存
`success` 和 `success_source="env.check_success_after_planner"`；Agent 的声明
另存为 `agent_finish.called/status/summary`。评分针对规划器返回后的最终状态，
不是“曾经成功”，也不是调用 `finish` 时的状态快照。`termination_reason` 区分
主动结束、未调用 `finish`、超时、响应截断和基础设施错误；原生成功不能掩盖
规划器错误，未调用 `finish` 本身也不会使可用的原生评分失效。

为兼容正在进行的评测，结果仍使用 `experiment_variant="agent_stop"`。
本分支验证器要求该标签、评分来源及一致的结束声明，不能混入 baseline 记录；
`agent_check_success` 是实验和分支名称，并非新的结果字段值。

本分支仅支持 **evaluation**，未适配 exploration 的成功保护和策略导出。
`finish` 处理函数与共享规划器流程未改动；prompt 要求结束后不再发出机器人命令，
但没有额外的动作拦截器。文件工具限制不等于系统沙箱，规划器原生 shell 仍可
访问文件。本实验保留模拟器几何和接触信息，没有叠加 DA3、NoContactGT 或
NoGeomTools。

按[中文 RoboCasa 配置说明](docs/source-zh/rst_source/simulators/robocasa.rst)
安装本分支及 `.[robocasa]`，配置能够调用 `gpt-6-astra` 的
[Codex 连接](docs/source-zh/rst_source/guides/configure_planner.rst)，提前下载资产、
RLDX 权重、基础模型支持文件及匹配的 Astra 记忆，再使用上方同一条 CLI 命令。
资源路径、离线设置、线程设置和预算均与英文示例一致；资源与私有凭据不随分支
发布。完整评测按 `target50.json` 运行 Seen/Unseen 各 16 个任务、每任务 seeds
1–5，共 160 局；两组均使用 `--split target`，每局使用独立输出目录。自定义
批量调度、运行中接入及服务器激活脚本不随分支发布，CLI 不依赖这些脚本。

发布前相关离线 CPU 套件 **270 项通过**，覆盖原生判据调用节奏、观测过滤、
RLDX 停止条件及结果/验证器约定，相关 lint 和格式检查也通过。真实
GPT-6-Astra/xhigh 单局冒烟（`ScrubCuttingBoard`，target seed 1）的执行链路和
记录审核通过，但 **任务原生判定失败**：Agent 声明成功，独立评分仍为
`success=false`、`valid=true`。该局耗时 350.8 秒，9 次工具调用、4 次观测；
3 次 RLDX 各运行 40 块 / 320 步后返回 `cap`，合计 960 步。审核未发现工具输出
包含实时任务成功/进度字段，该局也未调用原生 shell 或在 `finish` 后继续执行
动作。这验证了执行与评分分离，不代表完整基准表现或系统沙箱保证。
