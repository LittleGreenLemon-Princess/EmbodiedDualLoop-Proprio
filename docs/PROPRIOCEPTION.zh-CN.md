# 本体反馈与技能库融合

新增条件为 `dual-loop-proprio-v1`。运行协议仍为 `dual-loop-v1`，原始复现和
默认 delegate 输入保持原状。本功能只在显式启用时注入本体反馈。

## 设计来源

基于 EmbodiedDualLoop `5a5861e180c484e5bd345e9d2d2a9337152ed116`。
参考 ASPIRE `codex/reusable-knowledge` 分支
`b2cc044ad795da90aaaff58f037a7dcb51cc4733`：

- `aspire/sim/cap/envs/simulators/libero.py`：统一自身状态观测。
- `aspire/sim/cap/integrations/trace_logger.py`：执行后回读实测状态。
- `aspire/real/cap/saved_scripts/yam_runtime/gripper.py`：目标、实测和证据分离。
- `aspire/sim/cap/skills/library.py`：可复用技能的元信息与来源。
- `aspire/sim/cap/knowledge/{models,retrieval}.py`：适用范围、版本锁定和确定性检索。

这里为独立实现，没有复制 ASPIRE 源码，也不引入其机器人驱动、模型依赖或
跨试次学习行为。原有归属、配方、技能快照和许可文件保持不变；没有给继承代码
重新指定项目级许可证。

## 本体状态怎么进入双循环

`ProgramBackend` 从现有 `public_observation().state` 读取双臂状态：

| 字段 | 来源 | 语义 |
| --- | --- | --- |
| `ee_pose_world_wxyz` | `*_ee_pose` | 世界坐标米、wxyz 单位四元数 |
| `joint_position_rad` | `*_arm_joint_state` | 关节角，环境未提供则为 null |
| `gripper_opening_normalized` | `*_ee_joint_state[0]` | ARX 约定，0 闭合、1 打开 |

每个字段都有有效性标记。非有限值、错误形状、无效四元数和超范围夹爪量标为缺失。
不猜测关节速度、力矩、限位或接触力；后续接入需先确定公开接口的单位和映射。
`acquired_at_monotonic_s` 是宿主读取时间，不能用来证明传感器数据新鲜。

每次真实动作前后记录状态和变化：关节差、夹爪差、末端平移差、四元数测地角差。
仅观察不累计动作样本。生成代码失败但已消耗动作时，仍尝试记录实测效果。
反馈同时写入选择器输入、动作事件和最终 `latest` 报告；canonical pose
输入变换也保留该反馈。原始 RGB-D 和原生成功判断继续保留。

夹爪停止、末端到位、代码成功均不构成抓取成功。没有将 ASPIRE 的启发式
`contact_detected` 或机器人特定阈值直接搬过来。

## 使用

宿主和 CPU 验证使用 Linux / WSL，Python 3.11+。依赖沿用根目录安装说明。

```bash
python -m scripts.smoke_dual_loop --run-dir runs/202610231800_proprio_cpu --proprioception
python -m scripts.smoke_dual_loop --run-dir runs/202610231801_proprio_stuck --proprioception --stuck
python -m unittest discover -s tests -v
```

每次换一个以 YYYYMMDDHHMM 开头的新目录。CPU 演示不模拟接触或物体。
在正常 agent workspace 中，使用已有客户端：

```bash
python scripts/env.py delegate submission/rim_pick.json --selector jev --observation-profile proprioception-v1
```

API 请求等价于在 delegate 请求中增加 `"observation_profile": "proprioception-v1"`。
不传此字段即 baseline。当前固定整任务控制器仍沿用原条件；未自动更改其封存身份。

## 技能库

`integrations/skills/library.json` 是版本锁定的只读清单。首个条目
`thin-rim-pick@1.0.0` 复用现有四个 helper，标为 supplied，而非新学到或已验证技能。
清单包含 embodiment、能力标签、所需公开信号、物理前提、来源和源码哈希。
第二个条目 `ee-offset-correction@1.0.0` 封装已有左臂自由空间位置微调技能，
可用 `hover-correction` 能力标签检索；参数需包含当前公开目标 `target_pose`。
哈希按 UTF-8、统一 LF 的源码计算，Windows/Linux 一致。

`SkillLibrary.search()` 按 embodiment 与能力检索；`compile()` 检查固定版本、
机器人类型、声明可用信号和源码哈希，产生正常 `TaskBrief`。它不在宿主执行技能代码。
物理前提由调用方根据当前观察确认，编译成功不能替代前提核验。

在源码仓库中可以生成完整委托文件：

```bash
python -m scripts.compile_library_skill --signals left_ee_pose right_ee_pose left_ee_joint_state right_ee_joint_state --parameters examples/dual_loop/rim_parameters.json --task-id bowl-pick --goal "Lift the bowl and verify with fresh object evidence" --output work/rim_pick.json
```

示例参数的 detection 故意是 `ok: false`；需要替换为当前公开 RGB-D 的检测结果。
生成文件包含所有源码，可放入 agent workspace 的 `submission/` 后 delegate。
CLI 不覆盖已有文件。

本版提供显式检索与任务编译，没有自动从 Test 提取、晋升、注入或写回技能。
准备阶段可将确认过的 helper 和清单放进 `skills/`；只有 `skills/`、`lessons/`
按原协议冻结并转移。不要让 agent 访问宿主仓库或把清单放在可跨 Test 写入的位置。
若调整 helper，必须显式更新版本和哈希，再重新冻结；不得自动接受哈希变化。

## 验证范围

新增 CPU 测试覆盖字段缺失与无效值、四元数正负等价、动作前后反馈、
canonical transform 后的选择器输入、报告和事件证据，以及技能适用范围、
信号缺失、版本与源码篡改。实际 GPU 仿真、Jev 调用、真机抓取和性能需要独立实验。
