# Combined RoboCasa ablation with RGB and native success feedback

The Planner assesses intermediate progress from RGB and receives native
simulator success for task completion. Robot-state measurements and normal
action-execution summaries are hidden. The VLA and controllers retain their
internal proprioception, control logic and history.

Branch: `no_privileged_info_no_planner_pose`. Parent combined-ablation commit:
`5e612e916c011bcc76431efd1775040f86563fef`. Official runtime base:
`a744bd5f128ebef3cb8c6736cf7cd98adff681f1`.

| Information | Planner input and saved observations | Internal behavior |
| --- | --- | --- |
| Arm/base pose | No measured position or orientation, including current and historical pose fields. | RLDX and controllers retain their original proprioception, pose tracking and action execution. |
| Gripper and grasp-width heuristic | No `gripper_qpos`, `robot0_gripper_qpos`, or `held_apart`. | Gripper control and the VLA's gripper input remain available internally. |
| Movement measurements | No `final_dist`, `base_moved`, `moved`, `peak_lift`, or `base_drift`. | Controllers still compute measurements needed to execute and stop actions. |
| Execution diagnostics | No action `ok`, `stuck`, `status`, executed `steps` / `chunks` / action counts, `vla_desync`, or action duration. Normal `state` and `log.result` are empty objects. | Internal success/failure checks, execution limits, VLA history invalidation and framework timing continue to operate. |
| Geometry queries | `back_project_batch` and `query_world_map` are not registered. No SceneGraph or DA3 replacement. | Simulator depth/world-map artifacts remain internal. |
| Historical task/global memory | Empty selection, no memory paths in the prompt, `policy="none"`. | Existing corpus download/validation remains a startup dependency. |
| Task progress | Not queried by the observation builder; absent from current, saved and historical observations. | Simulator task subconditions and native success checks still run. |
| Contact/grasp labels | No `grasp_contact`, `grasp_obj`, `grasped`, or `grasp_detected`. | VLA wrapper does not query the object-contact report; simulator contact physics remains. |
| Success-rule source | No generated `success_criteria.md`; RPent text-file tools deny all text reads. | Runtime source remains on disk. |
| Native success | `success` and `robocasa_terminated` remain. | Native success early stopping and final scoring are unchanged; VLA action `status` is hidden from the Planner. |
| RGB and action history | RGB, camera/role labels, observation `step`, task language and Agent-issued commands remain. | VLA images, manual controls, budgets and current-episode interaction history remain. |

The output restriction applies to initial, current, historical and post-action
observations, saved state/action results and `view_env_state().log.result`.
`elapsed_s` is also hidden from returned and saved action logs; the framework
can still measure runtime internally. Observation `step` is an image/history
index, not an executed simulator-step count.
Action-result stage/reset markers, repeated VLA prompt/configuration fields and
video paths are also omitted. The manifest lists all 43 hidden feedback fields;
the initial observation can have `log.result=null` because no action preceded it.

Command arguments are preserved as inputs. For example, `move_base(steps=10)`
can still appear in `log.command`; it does not report how many steps actually
executed. Likewise, the Agent can call `finish(status="success")` or
`finish(status="stuck")`. These are completion declarations, distinct from
hidden action-status measurements. Coordinate targets and gripper commands
remain available. Internal primitive results needed by sequences such as
`scripted_grasp` still guide execution without being exposed to the Planner.

The prompt uses images to assess reach and spatial changes. Small relative
adjustments remain available through `move_delta` and `move_base`; absolute
coordinate actions accept targets chosen by the Planner. The complete live
task-language strategy, consecutive VLA-call rules and the fact that manual
commands reset VLA frame history remain in place. The prompt does not ask the
Planner to read gripper width, action status, `cap`, or `vla_desync`.

This is not an all-ground-truth removal. Simulator success and fixed prompt
priors such as counter height ≈0.9 m and arm reach ≈0.8 m remain. The simulator
is not modified. Internal geometry artifacts still exist; depth/world-map NPZ
files retain the original 25-step cleanup window, while RGB and camera metadata
are not removed by that window. Native Planner shell access is inherited from
the baseline, so file-tool restrictions are not an OS sandbox. Recorded
trajectories must be audited for attempts to recover hidden data through other
interfaces. This branch targets evaluation; exploration and memory generation
are outside this experiment.

## Run

Use the official RoboCasa environment, assets and RLDX-1-FT-RC365 checkpoint.
The historical memory corpus must be available for startup validation, although
none of its contents is selected for the Planner.

```bash
export RLDX_MAX_CHUNKS=40 RLDX_ACTION_STEPS_PER_CHUNK=8 RLDX_SETTLE_PATIENCE=999
rpent --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/no_privileged_info_no_planner_pose/composite_seen/ScrubCuttingBoard_s1
```

[The manifest](experiments/no_privileged_info_no_planner_pose_composite.json)
preserves 32 composite tasks and seeds 1–5: 80 Seen and 80 Unseen episodes.
Both groups use `--split target`. Keep machine-specific queue scripts and
generated results outside this checkout.

```bash
python -m robots.robocasa.eval.validate_target50 ./runs/no_privileged_info_no_planner_pose \
  --manifest ./experiments/no_privileged_info_no_planner_pose_composite.json
```

## Validation Status

The current expanded output restriction passed 209 RoboCasa and tool-schema
tests on 2026-11-10. These include native-versus-dispatched action comparisons,
scripted-grasp failure paths, consecutive VLA history and manual reseeding,
success early stopping, and filtering when state capture fails. Changed Python
files passed Ruff checks. All 29 native primitive methods other than the
observation builder match the preceding smoke's implementation when docstrings
are excluded; the RLDX skill implementation is byte-identical.

Nine reads of the preceding real run's archived observations through the new
production view tool passed: empty state, empty normal action results, unchanged
native success and command history, and byte-identical RGB images. The original
archive was not modified. This is replay of existing observations, not a new
live Agent run; the unit tests separately inject all 43 restricted fields.

A new real Agent smoke was attempted but its launcher exited before starting
the simulator, VLA or Planner: no idle GPU had enough free memory. The expanded
interface has not completed a new real Agent smoke or the full 160-episode
evaluation. Previous validation alone does not establish correctness of the
additional gripper/execution-output removals.

The earlier 2026-11-10 smoke lasted 353.2 seconds and checked only the pose and
movement-output version. It completed five VLA calls, `move_delta`, `move_base`,
and the initial observation. That version still returned gripper measurements,
`held_apart`, action `ok` / `status`, execution counts and `vla_desync`. Its
182 passing tests and 9 offline historical-view checks belong to that earlier
interface. The smoke exhausted its 20-response budget with native success false
and no `finish` call; it is neither validation of the current interface nor a
benchmark success-rate measurement.

Run the focused contracts before a real simulator/VLA smoke:

```bash
pytest tests/unit_tests/robots/robocasa tests/unit_tests/robots/test_tool_schema_contracts.py -q
```

Verify empty robot state and normal action-result objects across initial,
current, historical and saved observations. Check that Agent command parameters
remain intact and that no executed-step counts or durations leak through another
wrapper. Also verify internal RLDX inputs/history, controller and scripted
sequence behavior, native success and early stopping. Record any actual
`finish` call separately from a budget stop.

## 中文说明

本分支在组合信息移除基础上，隐藏 Planner 可见的末端／底盘位姿、夹爪开度、
`held_apart`、移动距离、动作 `ok`／`stuck`／`status`、实际执行步数／chunk 数、
`vla_desync` 和动作耗时。初始、当前、历史、动作后观测以及保存状态和动作结果
遵循同一边界；正常 `state`、`log.result` 为空对象，初始无动作时 `log.result` 可为 null。
动作结果中的阶段／重置标志、VLA 重复指令／配置和视频路径也不返回；清单列出完整
43 个隐藏字段。框架内部仍可计时，VLA／控制器
仍使用原有 proprioception、历史和执行判定，`scripted_grasp` 等内部控制链继续运行。

Agent 依据 RGB 判断中间进度，依据保留的 `success`／`robocasa_terminated` 判断
任务完成。保留图像索引 `step`、任务文字和自己的 command 历史；输入的 `steps`、
目标坐标、夹爪命令，以及 `finish(status=...)` 都是 Agent 参数，不是执行测量反馈。
Prompt 不再要求读取夹爪宽度、动作状态、cap 或 vla_desync；完整任务 VLA 策略、
连续调用规则、手动动作清除 VLA 历史的规则及夹爪动作参数含义保持原样。

原生 success、成功早停、最终评分与台面约 0.9 m／机械臂可达约 0.8 m 的固定先验
仍保留，因此不能称为全部无真值。此前禁用的两个主动几何工具、历史 memory、
task_progress、接触抓取标签和成功判据源码入口继续禁用。内部语料和几何产物仍存在；
深度／world-map NPZ 沿用最近 25 步保留窗口，RGB 与相机 metadata 不按此窗口清理。
文件工具限制不是系统级沙箱，正式评测仍需核查实际轨迹；不改探索和记忆生成。

当前扩展版本通过 209 项 RoboCasa／工具 schema 测试与 Ruff 检查，包括原生和
工具派发动作序列对比、抓取失败分支、连续 VLA 历史、手动动作后的历史重建、
成功早停与状态捕获失败时的返回过滤。除观测构造函数外，29 个底层方法去除
docstring 后与上一版一致；RLDX 技能文件完全一致。

用新生产观测接口回读旧真实归档的 8 步及 latest，共 9 次，全部返回空 state／
正常空动作结果，success、command 和 RGB 保持一致，原归档未改变。这是旧数据
的离线回读，不是新真实 Agent 运行；43 字段逐一注入的覆盖由单测提供。

新真实 Agent 冒烟的启动检查因没有显存充足的空闲 GPU 退出，未启动模拟器、VLA
或 Planner；尚未完成新真实冒烟或 32 任务 × 5 seeds 的正式评测。之前 353.2 秒
的冒烟及 182 项测试属于仍保留夹爪／执行反馈的旧接口，不能单独证明本轮新增移除已验证。
该旧冒烟在 20 个响应预算用尽后停止，原生任务未成功，也没有 Agent finish。
