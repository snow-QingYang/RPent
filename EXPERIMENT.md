# Pure RLDX-1 on RoboCasa composite

状态：**评测入口和独立环境已建立。真实 VLA 的 32 步动作链已通过验证；正式队列在空卡上完成最终冒烟验收后自动开始 160 局。**
分支：`pure_vla`。新目录：`HarnessVLA-RPent-PureVLA`。

本分支从已评测 baseline 提交 `eebdfcd79b6f292003093be5dd1ce8fe639a4f82`
创建；其 RPent 运行代码等同官方 `a744bd5f128ebef3cb8c6736cf7cd98adff681f1`。
本分支新增独立评测入口，不修改 Planner、VLA、模拟器或现有实验的实现。

## 1. 实验要回答什么

相同 RLDX-1-FT-RC365 checkpoint 直接执行完整任务、没有外层 Planner 和手工
工具介入时，在现有 32 个 composite 任务上能达到什么成功率。

这是一组采用官方 RLDX 任务预算的纯 VLA 基线。现有网站里的“仅 VLA 动作成功”
仍然经过 Planner 选择、续调工具，不能代替这组实验。与原 HarnessVLA 结果的
对照需要同时报告预算、初始化和成功判定的差异，不能直接解释成 Planner 的
严格单变量因果收益。

## 2. 固定任务、模型和输入

| 项目 | 设置 |
| --- | --- |
| 任务 | 与 `robots/robocasa/eval/target50.json` 一致的 composite_seen 16 个、composite_unseen 16 个；不跑 atomic。 |
| 局数 | 每任务显式 seeds 1–5，共 160 局；每个 task + seed 是独立单元。 |
| 环境 split | `target`。Seen / Unseen 是统计分组。 |
| 模型 | `RLWRLD/RLDX-1-FT-RC365`，revision `587e9ecdcc5e7184fcc17f58713908edff5af041`，沿用 baseline checkpoint。 |
| 模型支持文件 | `RLWRLD/RLDX-1-VLM`，revision `4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7`。 |
| RLDX 实现 | 锁定 baseline 环境中的 `RLinf/RLDX-1@rpent` commit `ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a`；不因上游更新而换模型或实现。 |
| 图像 | 左、右、腕部三路 256×256 RGB；相机键和预处理沿用官方实现。 |
| 任务文字 | 当前场景的完整原生 task language，整局不拆解、不改写为子任务。 |
| 本体状态 | 保留模型所需的末端相对底盘位置／姿态、夹爪位置、底盘位置／姿态。纯 VLA 不等于仅 RGB。 |
| 时间历史 | 从 checkpoint modality 配置读取；当前视频索引为 `[-6,-4,-2,0]`，state 为 `[0]`。 |
| 动作 | 官方动作映射和全身控制；当前 checkpoint 预测 16 步、每次执行前 8 步再观察。允许 VLA 自己移动底盘。 |

不启动 GPT/Codex/其他 Planner，不建立 MCP 工具调用，不使用 `move_to`、
`scripted_grasp`、`rldx_arm` 等由 Planner 调度的技能，不读任务／全局 memory、
评分源码、三维查询结果、接触标签或 task_progress 来决定动作。

保留 **RLDX 模型内部的 temporal memory / RTC 和视频历史**。每局重置 policy
session，局内连续运行；不能把移除 Planner 文件 memory 错做成每个 chunk 重置模型。

## 3. 复用官方评测，最小适配

官方来源：

- [RoboCasa365 评测 shell](https://github.com/RLinf/RLDX-1/blob/ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a/run_scripts/eval/robocasa_365/eval_robocasa365.sh)
- [任务分组与 horizon](https://github.com/RLinf/RLDX-1/blob/ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a/run_scripts/eval/robocasa_365/task_sets.yaml)
- [官方 rollout 和 policy 构造](https://github.com/RLinf/RLDX-1/blob/ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a/rldx/eval/rollout_policy.py)

直接复用 `create_rldx_sim_policy`、`run_rollout_gymnasium_policy`、
`MultiStepWrapper`、原生 RoboCasa Gym 观测／动作转换，不自己重写模型推理循环。
新增入口负责指定任务、seed、预算、环境兼容和结果保存。

只在本实验入口提供环境 factory 适配，不修改共享 site-packages 或其他实验仓库。
实现前已识别两处需要处理的兼容问题：

1. 当前 RLDX factory 还导入已安装 RoboCasa 中不存在的
   `robocasa.utils.gym_utils.GrootRoboCasaEnv`；当前原生接口是
   `robocasa.wrappers.gym_wrapper.RoboCasaGymEnv`。使用真实原生接口，核对其
   controller、camera、初始化和 reset 行为，保留官方 policy 与动作转换。
2. 官方高层 `run_rldx_sim_policy(video_dir=None)` 仍会自动创建录像目录，开启
   `VideoRecordingWrapper`，继而改变 reset seed。因此使用较低层
   `run_rollout_gymnasium_policy`，显式令 `WrapperConfigs.video.video_dir=None`。
   另外保存每次模型调用已收到的 RGB 拼图和终态图像，不增加 render、reset 或录像层 seed 重写。

同一个 task / seed 下，官方 Gym factory 和 RPent 原初始化还可能在全局随机数
设定、构造期 reset 次数上不同。冒烟阶段核对这些差异并明确记录；不能仅关闭
录像 seed 重写就宣称与历史 baseline 的完整初始场景相同。

## 4. 预算、成功和停止

主实验使用官方 `task_sets.yaml` 中每任务的 horizon，范围 800–2900 个
**原生 simulator action steps**，不根据运行结果事后调整。

Baseline 的 `40 chunks × 8 actions = 320` 是一次 VLA 工具调用的上限，Planner
可以继续调用，不能拿它作为整局纯 VLA 的预算。这里的官方 horizon 与已有
100-turn HarnessVLA 设置不等预算；同时公开真实动作数、VLA 推理次数、模拟时间
和实际耗时。需要严格等预算比较时，应另设计双方采用同一动作上限的实验。

- 主评分与停止沿用官方 rollout：环境在执行步中产生的原生 success 累计为
  `any_success`，在官方 chunk 返回边界按成功或 horizon 结束。
- success 仅供评测器评分和停止，不进入 VLA 的 observation，不用于改写指令、
  规划下一动作或调用额外工具。
- 另外保存终态原生 `final_state_success`、首次成功的原生 action step、停止原因。
  当前 RPent 技能在 chunk 末检查当前 success，和官方累计步内 success 有差别，
  报告必须分列，不能混为同一指标。
- 终态在 termination / vector autoreset **之前**缓存；不能等官方 rollout 返回后
  再查询，因为官方函数收尾还会 reset / close 环境。
- 不增加“卡住就手动修复”“闭爪就算抓到”“40 chunks 后重新启动任务”等规则。
- 主预算是 simulator steps；manifest 固定每局 7200 秒进程保护上限，基础设施失败最多尝试三次。
  超限未解决的单元标记 blocked，不作为任务失败，也不静默移出 160 局计划。

官方 CSV 的 `steps` 是外层 chunk 计数，不是原生动作数；保存时另计实际
`env.step` 次数，不能简单沿用这个字段名称或一律乘 8。

## 5. Seed、初态和结果有效性

每个 task + seed 单独运行，`n_envs=1`、`n_episodes=1`。不会把同一默认 seed 下
连续生成五局，标成 seeds 1–5。

原 HarnessVLA 将 CLI seed 传给环境构造，清空 `RLDX_RESET_SEED` 后正常 reset。
官方录像 wrapper 则有 `base_seed * 100000 + episode_id` 的覆盖逻辑；该版本首次
使用递增前的 episode_id=-1，例如 nominal seed 1 会变成 99999。本实验不启用
这条覆盖路径。

环境 factory 保留原生 Gym 构造时的首次 reset 结果，交给官方 rollout 的首次 reset 使用，避免额外采样一局。收尾 reset 返回缓存终态，不改写评分。
记录 nominal seed、实际传入的 constructor/reset seeds、reset 次数、依赖提交、
初始任务文字、三路 RGB、本体状态和可用的完整初始化记录／哈希。实际差异必须
显示在对照报告里；相同 seed 或少数观测相同不证明全部物体初态相同。

预先固定 VLA 采样 seed 的生成方式，独立于任务采样 RNG；记录依赖、checkpoint
和采样参数。多 GPU 分配顺序不能决定一局的 seed。保留模型原有随机采样，不
为了得到相同动作而改成另一种推理方法，也不承诺跨硬件逐位确定性。

有效任务失败照常计入分母。OOM、服务异常等基础设施失败单独记录并重试同一
单元，旧尝试归档；不重跑有效失败挑最好一次。已完成单元按完整、可校验的结果
恢复，存在目录或 MP4 文件不等于已经完成。

## 6. 本分支新增的文件

评测实现集中在下列独立文件：

| 文件 | 用途 |
| --- | --- |
| `robots/robocasa/eval/pure_vla.py` | 薄入口：调用官方 policy/rollout，适配环境构造、seed 与终态记录，输出单局结果。 |
| `experiments/pure_vla_composite.json` | 固定 32 任务 × 5 seeds、官方 horizons、依赖版本、评分与重试协议。 |
| `eval_pure_vla.sh` | 简短、可移植的单任务／单卡调用入口，不携带服务器排队逻辑。 |
| `robots/robocasa/eval/summarize_pure_vla.py` | 校验 160 个唯一单元，汇总 Seen/Unseen、两种 success、逐 seed 矩阵、动作数及耗时。 |
| `tests/unit_tests/robots/robocasa/test_pure_vla_eval.py` | 覆盖 seed/reset、评分时机、horizon 和有效结果判定等必要契约。 |

不修改原 `rldx_skill.py`、`primitives.py`、Agent prompt、MCP 注册或 Planner 框架。
原 baseline 的启动脚本随分支基底保留，不作为纯 VLA 启动入口。

## 7. 验证、队列与复现

运行环境独立复制自 baseline，复用下载好的模型与资产，没有修改原环境。
CPU 合约测试覆盖任务列表、seed、局内 memory、原生步数、瞬时成功与终态、
损坏结果拒绝恢复，以及模型输入白名单。真实冒烟使用 ScrubCuttingBoard seed 1，
4 次推理执行 32 个原生动作，视频历史持续更新，完整任务指令保持不变。
冒烟结果不计入正式 160 局。

正式队列先在空卡完成最终版本的 32 步验收，通过后自动派发任务。最多八张卡，
每张卡独立加载一个模型、同时执行一个 episode；当前实现每局重新启动进程，
不复用上一局的模型状态。共享队列动态领取下一未完成单元，每分钟检查空卡。
若发现外来计算进程共卡，停止并隔离本次尝试；正常的任务失败不重跑。
进程被中断时从该局开头重试，已通过校验的完整结果不会重复计分。
服务器专用调度脚本和运行记录在仓库外，不纳入分支。

可移植的单局入口（先激活 RoboCasa 环境并设置资产路径）：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONHASHSEED=0 \
  bash eval_pure_vla.sh --task-name ScrubCuttingBoard --seed 1 \
  --checkpoint ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --output-dir ./runs/pure_vla/composite_seen/ScrubCuttingBoard_s1
```

`--cuda-device 0` 指可见 CUDA 设备内的编号；物理卡通过 `CUDA_VISIBLE_DEVICES`
选择，EGL 使用其物理编号。加 `--smoke-max-native-steps 32` 只用于冒烟。
正式单局从 manifest 自动读取本任务 horizon，不接受正式预算临时改写。

汇总整个运行目录：

```bash
python -m robots.robocasa.eval.summarize_pure_vla ./runs/pure_vla \
  --output ./runs/pure_vla/summary.json
```

每局保存 `result.json`、`policy_calls.jsonl`、`native_steps.jsonl`、初始 RGB／
本体状态／模拟器状态与 XML、每次模型调用的 RGB 拼图和终态图像。
模拟器状态和原生 success 是评测端审计记录，未放入模型 observation。
模型输入在每次推理前按九个允许键校验。

### 初态对照的实际限制

冒烟中，同一个 task/seed 的原生 Gym 和 baseline 探针出现过不同厨房布局。
诊断确认模拟器的物体放置失败会在一次 reset 内重新抽样布局，因此只检查 reset
次数不足以证明场景一致。上游 `robocasa/models/fixtures/counter.py` 中的
`valid_geoms = list(set(valid_geoms))` 还使用 XML 元素对象的集合顺序，可能影响
候选台面区域与后续放置；`PYTHONHASHSEED=0` 不能固定这种对象身份顺序。

本实验固定 hash seed 并保存每局初态，但不修改官方模拟器来消除这些差异，
也不宣称与历史 baseline 按 seed 严格配对。报告同时说明官方动作 horizon 与
原 HarnessVLA turn 预算、逐动作累计 success 与 chunk 末当前 success 的差异。
完成后在网站上公布 Seen/Unseen、逐 seed 矩阵和案例，并与已有实验对照。

## 8. 官方任务 horizon

单位：原生 simulator action steps。数值取自上述固定 RLDX 提交。

| Seen task | Horizon | Unseen task | Horizon |
| --- | ---: | --- | ---: |
| ScrubCuttingBoard | 800 | ArrangeBreadBasket | 2900 |
| StackBowlsCabinet | 1400 | ArrangeTea | 1500 |
| WashLettuce | 1100 | BreadSelection | 1300 |
| RinseSinkBasin | 900 | CategorizeCondiments | 1100 |
| PreSoakPan | 1600 | CuttingToolSelection | 800 |
| StirVegetables | 1600 | GarnishPancake | 1800 |
| LoadDishwasher | 1200 | GatherTableware | 1500 |
| SteamInMicrowave | 1400 | HeatKebabSandwich | 1800 |
| SetUpCuttingStation | 1600 | MakeIceLemonade | 2000 |
| GetToastedBread | 2000 | PanTransfer | 1200 |
| DeliverStraw | 1700 | PortionHotDogs | 1500 |
| KettleBoiling | 1000 | RecycleBottlesByType | 1900 |
| PrepareCoffee | 1200 | SeparateFreezerRack | 1600 |
| StoreLeftoversInBowl | 1700 | WaffleReheat | 2700 |
| SearingMeat | 2900 | WashFruitColander | 2100 |
| PackIdenticalLunches | 2600 | WeighIngredients | 2000 |

若每局都跑满 horizon，160 局总上限为 262,000 个原生动作；成功早停会减少实际
动作数。该上限不能直接换算为 GPU 小时，耗时需由后续真实冒烟测量。
