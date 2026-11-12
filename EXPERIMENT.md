# Combined RoboCasa ablation with Planner robot pose and simulator success

Branch: `no_privileged_info_with_planner_pose`. Official runtime base:
`a744bd5f128ebef3cb8c6736cf7cd98adff681f1`, matching the previous composite baseline.
This evaluation combines the geometry-tool, historical-memory, task-progress,
and contact-feedback ablations. The Planner receives robot end-effector and
base positions and orientations. Simulator success remains available.

| Information | Evaluation behavior |
| --- | --- |
| Geometry queries | `back_project_batch` and `query_world_map` are not registered. No scene-graph tool or DA3 model is added. |
| Historical task/global memory | Empty selection, no memory paths in the prompt, and `policy="none"`. |
| Task progress | Not queried by the observation builder; absent from current, saved and historical observations. |
| Contact/grasp labels | VLA does not query object contact or return `grasp_contact`, `grasp_obj`, `grasped`, or `grasp_detected`. |
| Success-rule source | `success_criteria.md` is not generated; RPent text-file tools cannot read source or other text files. |
| Native success | `success`, `robocasa_terminated`, successful VLA early stopping and final native scoring are retained. |
| Planner robot pose | End-effector and base positions and orientations remain visible to the Agent. |
| Observations/control | RGB, base/arm/gripper proprioception, VLA inputs/history, manual actions and current-episode interaction history are retained. |

`held_apart` remains a command/finger-width heuristic, not a verified grasp.
Coordinate-based actions still accept coordinates; they do not reveal object
coordinates. The prompt uses images and robot observations for intermediate
decisions and directs the Agent to finish when the simulator reports success.

This combines the earlier **Agent tool/output** ablations. The simulator is not
modified and internal depth/world-map artifacts still exist. Native planner shell
access is inherited from baseline: file-tool restrictions are not an OS sandbox.
Recorded trajectories must also be audited for attempts to recover hidden data
through other interfaces. This branch targets evaluation; exploration and memory
generation are outside this experiment.

## Run

Use the official RoboCasa environment, assets and RLDX-1-FT-RC365 checkpoint.
Corpus download/validation remains a startup dependency, as in NoMemory;
none of the historical corpus contents is selected for the Agent.

```bash
export RLDX_MAX_CHUNKS=40 RLDX_ACTION_STEPS_PER_CHUNK=8 RLDX_SETTLE_PATIENCE=999
rpent --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/no_privileged_info/composite_seen/ScrubCuttingBoard_s1
```

[The manifest](experiments/no_privileged_info_composite.json) defines 32 composite
tasks with seeds 1–5: 80 Seen and 80 Unseen episodes. Both use `--split target`.
Keep machine-specific queue scripts and generated results outside this checkout.

```bash
python -m robots.robocasa.eval.validate_target50 ./runs/no_privileged_info \
  --manifest ./experiments/no_privileged_info_composite.json
```

## Validation

```bash
pytest tests/unit_tests/robots/robocasa tests/unit_tests/robots/test_tool_schema_contracts.py -q
```

The combined contract tests cover blocked geometry dispatch, denied memory/text
reads including aliases, current/historical/saved output, retained robot state,
VLA history continuity, native-success early stopping, and absent criteria export.
A bounded scripted smoke checks real simulator/VLA execution and `finish`;
it is not a benchmark success-rate measurement.

On 2026-11-09, all 155 tests above passed; the 14 combined contract cases also
passed independently. Ruff and diff checks passed. A real ScrubCuttingBoard,
target seed 1, smoke executed two VLA chunks (four simulator actions) and called
`finish(status="stuck")` at its scripted limit. Its tool responses and saved
states contained none of the removed fields, memory reads were zero, and no
success-rule file was generated. Native-success early stopping is covered by
the unit tests; the bounded smoke did not establish task success.

## 中文说明

分支 `no_privileged_info_with_planner_pose` 保留 Agent 可见的机器人末端与底盘位姿。
本分支合并此前四项消融：禁用两个主动三维查询工具、历史 memory、task_progress、
模拟器接触及接触派生的抓取提示；同时取消生成成功判据源码，并限制文本文件读取。
保留 simulator success、成功早停、原生最终评分，以及相机图像、底盘／末端／夹爪
自身状态、VLA 输入与历史、动作接口和本局交互历史。

这是 Agent 工具／输出层的组合消融。内部深度和 world map 仍生成，原生 Planner
shell 也不是系统级隔离；正式评测需审计实际调用，确认没有从其他接口获取隐藏信息。
本分支用于评测，不重设计探索与记忆生成流程。32 个 composite 任务各 5 个 seed，
共 160 局；目前仅准备组合版本与验证，不代表已完成正式评测。
155 项相关测试及真实模拟器／VLA 短冒烟通过；冒烟仅执行四个模拟器动作后按脚本
调用 finish，验证调用链与信息边界，不作为任务成功率结果。
