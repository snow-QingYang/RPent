# Pure RLDX-1 on RoboCasa composite — experiment plan

状态：**方案已建立，尚未实现评测入口、配置新环境或启动实验。**
分支：`pure_vla`。新目录：`HarnessVLA-RPent-PureVLA`。

本分支从已评测 baseline 提交 `eebdfcd79b6f292003093be5dd1ce8fe639a4f82`
创建；其 RPent 运行代码等同官方 `a744bd5f128ebef3cb8c6736cf7cd98adff681f1`。
本次仅记录方案，不修改 Planner、VLA、模拟器或现有实验。

## 1. 实验要回答什么

相同 RLDX-1-FT-RC365 checkpoint 直接执行完整任务、没有外层 Planner 和手工
工具介入时，在现有 32 个 composite 任务上能达到什么成功率。

这是一组采用官方 RLDX 任务预算的纯 VLA 基线。现有网站里的“仅 VLA 动作成功”
仍然经过 Planner 选择、续调工具，不能代替这组实验。与原 HarnessVLA 结果的
对照需要同时报告预算、初始化和成功判定的差异，不能直接解释成 Planner 的
严格单变量因果收益。

## 2. 固定任务、模型和输入

| 项目 | 计划 |
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

计划直接复用 `create_rldx_sim_policy`、`run_rollout_gymnasium_policy`、
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
   如需录像，在不改变 seed、reset 次数、动作和成功判定的记录层单独保存。

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
- 主预算是 simulator steps；独立进程保护超时如需设置，在正式运行前写入 manifest，
  并按事先定义的规则分类，不能把慢任务静默移出分母。

官方 CSV 的 `steps` 是外层 chunk 计数，不是原生动作数；保存时另计实际
`env.step` 次数，不能简单沿用这个字段名称或一律乘 8。

## 5. Seed、初态和结果有效性

每个 task + seed 单独运行，`n_envs=1`、`n_episodes=1`。不会把同一默认 seed 下
连续生成五局，标成 seeds 1–5。

原 HarnessVLA 将 CLI seed 传给环境构造，清空 `RLDX_RESET_SEED` 后正常 reset。
官方录像 wrapper 则有 `base_seed * 100000 + episode_id` 的覆盖逻辑；该版本首次
使用递增前的 episode_id=-1，例如 nominal seed 1 会变成 99999。本实验不启用
这条覆盖路径。

计划通过环境 factory 对齐现有 RPent 的构造／reset 协议，并在冒烟时核验。
记录 nominal seed、实际传入的 constructor/reset seeds、reset 次数、依赖提交、
初始任务文字、三路 RGB、本体状态和可用的完整初始化记录／哈希。实际差异必须
显示在对照报告里；相同 seed 或少数观测相同不证明全部物体初态相同。

预先固定 VLA 采样 seed 的生成方式，独立于任务采样 RNG；记录依赖、checkpoint
和采样参数。多 GPU 分配顺序不能决定一局的 seed。保留模型原有随机采样，不
为了得到相同动作而改成另一种推理方法，也不承诺跨硬件逐位确定性。

有效任务失败照常计入分母。OOM、服务异常等基础设施失败单独记录并重试同一
单元，旧尝试归档；不重跑有效失败挑最好一次。已完成单元按完整、可校验的结果
恢复，存在目录或 MP4 文件不等于已经完成。

## 6. 后续预计涉及的文件

当前只修改本文。实现阶段预计增加下列独立文件，审阅时保持改动范围明确：

| 文件 | 用途 |
| --- | --- |
| `robots/robocasa/eval/pure_vla.py` | 薄入口：调用官方 policy/rollout，适配环境构造、seed 与终态记录，输出单局结果。 |
| `experiments/pure_vla_composite.json` | 固定 32 任务 × 5 seeds、官方 horizons、依赖版本、评分与重试协议。 |
| `eval_pure_vla.sh` | 简短、可移植的单任务／单卡调用入口，不携带服务器排队逻辑。 |
| `robots/robocasa/eval/summarize_pure_vla.py` | 校验 160 个唯一单元，汇总 Seen/Unseen、两种 success、动作预算及 baseline 对照。 |
| `tests/unit_tests/robots/robocasa/test_pure_vla_eval.py` | 覆盖 seed/reset、评分时机、horizon 和有效结果判定等必要契约。 |

不修改原 `rldx_skill.py`、`primitives.py`、Agent prompt、MCP 注册或 Planner 框架。
原 baseline 的启动脚本随分支基底保留，不作为纯 VLA 启动入口。

## 7. 验证与运行顺序

1. 建立与 baseline 版本一致的独立运行环境，复用已下载权重／资产并校验版本，
   不改共享环境的代码。完成兼容入口和 manifest，先检查任务集合及预算。
2. CPU 检查 seed 不被录像层重写、每局 policy reset、终态在自动 reset 前缓存、
   原生动作计数和失败分类；确认没有 Planner/MCP/memory 的启动或读取路径。
3. 经后续授权运行单任务冒烟，记录全部模型输入键、任务指令、动作与评分。
   核对三路图像、本体输入、历史连续、全身动作、8 步节奏与初始状态。
   断点恢复也要验证不会因残留目录误跳过或重复计分。
4. 冒烟通过后，按后续安排启动正式 160 局。八卡时每卡一个 VLA worker、
   同时一个 episode；每局独立环境和模型状态，可在局间复用模型权重。
   空闲 worker 从共享队列领下一个未完成单元，服务器调度脚本放仓库外。
5. 报告 Seen/Unseen/总成功率、两种 success、逐 seed 矩阵、步数和耗时，并与
   E01/E08/E09 对照。说明旧 HarnessVLA 的动作预算、判定与初态限制；不把
   raw 成功率差值直接写成严格的 Planner 因果增益。完成核验后更新网站。

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
