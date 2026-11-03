# 固定基础任务程序，遇到问题才启动 System 2

新配置是 `configs/dual-loop-frozen`，运行条件为
`fixed-base-with-local-system2-repair-v1`。旧的标准复现、诊断 pilot、每场在线开发
配置都保留；新条件不能与它们混合计算成功率。

## 开发一次，固定完整程序

每个原生任务在 Playground 由实际 System 2 开发完整参数化任务程序，包括现场感知、
流程状态、可选参数、停止条件与物体证据。入口保存为 `skills/task_program.py`，
`skills/task_program.json` 列出原生任务、代码依赖、参数和预算。每个入口仍为
skill_init/context/options/tick，初始化和观察零动作，每个 tick 最多一次原生动作。

当前场景的对象位置应从新观察重新取得，不能依赖 Playground 的 Python 变量、
scratch 或固定场景坐标。常量、几何假设与范围可以作为参数。完整程序放在 skills/
中，因此仍只通过原有 skills/lessons 冻结通道转移，不转移整个开发现场。

开发阶段先把 manifest 的 task 对象写入 submission/development_task.json，再调用
原有 delegate。最终导出必须与一次实际委托过的代码和参数一致；host 在 finish
以及冻结时检查。文件格式正确不等于任务成功，开发失败必须照实保留。

冻结后生成宿主文件 frozen-task.json，记录程序哈希和开发报告。所有 Test 使用
相同初始哈希、新模型无关的程序状态和新仿真；在采样 Test 场景之前完成冻结检查。
同一原生任务的目标物体/颜色也可能变化；宿主每场从公开 get_instruction() 绑定 Jev
目标，程序初始化也重新读取该指令，不沿用 Playground 目标物体的文字。

## 新场景的正常执行

`orchestrator.frozen_task` 是普通控制程序，直接把固定任务交给 System 1。
它不会提前启动负责写代码的 System 2。Jev 每次从当前测量和有界选项选择参数，
原生动作上限、任务目标和成功检查不变。

如果得到原生任务成功，直接结束。模型说 done、代码结束或末端到位不会替代原生成功。
失败、停滞、置信度不足、缺少物体证据、决策/动作/时间阈值等会返回交接报告。
原生 episode 已结束或全局预算耗尽时，不再启动一个无法执行修复的模型。

## System 2 按需局部修复

manifest 中 `max_system2_interventions` 默认为 2，可设 0..3。每次介入启动新的实际
System 2 会话，给它当前原始报告、状态和剩余预算。它只允许零动作观察/几何分析，
并在 submission/ 写一份局部修复提案；不能自己 delegate、reset 或 finish。
宿主使用工作区外的控制令牌限制动作委托和结束操作，令牌不注入修复模型的环境，
不写入公开请求日志。当前 local 模式不构成文件系统安全隔离，不能把此边界说成容器隔离。

控制程序核对冻结文件未改、原生目标未改、置信度不降低、预算不扩大，再把修复交给
Jev。只改任务名、版本号或预算的同一程序会被拒绝；语义上是否真正修复仍需结合证据
审查，不能保证检测所有通过参数计数器绕行的情况。连续闭合/保持仍须在一个循环内完成。

修复从当前物理状态继续，没有重新起场或退还已用动作。修复代码与原始失败都保存。
该场结束后丢弃场景内修复；下一场重新使用固定基础程序，不能积累 Test 经验。
这叫“固定基础程序 + 允许局部修复”，不是 ASPIRE 的严格 frozen-code 条件。

### 缓存与夹爪反馈（2026-10-21 修补）

本地模型进程的 Python 子进程默认 `PYTHONDONTWRITEBYTECODE=1`，不生成导入缓存。
固定程序控制器核对 skills/lessons 时只忽略 `__pycache__/` 内的 `.pyc` 文件；
源码、参数、其他文件的增删改和符号链接仍被检查。原有全局冻结清单与历史结果不改写。
修复进程返回或异常时先保存耗时，再检查基础程序，避免报错后漏记模型耗时。

固定程序模式（开发与 Test）由宿主向 `context.measured_grippers` 注入公开控制状态：
RoboDojo 的 `*_ee_joint_state[0]` 来自 `prev_control` 中最近控制目标的归一化值，
控制器可能根据当前关节位置限制目标变化幅度，它不是独立测得的夹爪开度。
为保持兼容，`measured_opening`、`initial_opening` 等原字段名和值保留；动作前后差值、
开/闭/静止标签和连续静止次数均描述控制设定的变化，不代表夹爪实际运动或收敛。
ARX 的 1 表示打开目标、0 表示闭合目标；缺失为 null/unknown。
单纯重新观察不增加动作或静止计数，最后一步变化会保留到下一次决策。
这些字段同时进入 Jev 输入、逐步反馈日志及 System 2 交接报告。原有其他模式保持原输入。

本接口没有提供独立夹爪开度测量，不能从目标变化或不变推断真实夹爪运动、接触或抓取。
闭合/保持动作仍须有界，结合公开物体视觉证据和原生判分判断进展。
System 2 提示词同时要求有界闭合/保持、检查当前状态后续接，不盲目重新打开夹爪。
本修补不改写之前冻结的任务代码，也不改变其阶段逻辑或直接替模型选择动作。

## 入口与结果

```sh
python -m orchestrator run --task configs/dual-loop-frozen \
  --env-task general_pickup --test-scenes 2 --gpu 0 \
  --run-dir runs/202610211200_fixed_task_smoke
```

替换实际时间前缀；每次开发使用新目录。批量脚本默认也改为此配置；传入旧配置可继续
原流程。一个任务内还是先完成开发与冻结，然后评测多个新场景，不是每个场景开发一遍。

每场的 fixed-task/result.json 与 events.jsonl 记录基础哈希、首次结果、介入次数、
每次触发原因与修订、修复模型进程耗时和最终原生结果；repairs/ 保存各次实际模型日志。
test.json 的 fixed_task_summary 区分 primary_successes、recovered_successes、
system2_interventions 和 repair_agent_seconds。后者含模型工具等待，不等于纯推理时间。
统计缺少测量或无效场景时不编造完整成功率。

默认保留原运行证据和代码清单，不额外重跑仿真回放；也有 4 GiB 批量场景启动前检查。
原生任务预算不变。当前修复进程只实现已配置的 local Claude CLI 路由；要求 Docker
隔离时明确拒绝，不静默降级。完整 CPU 检查和实际仿真运行应分别报告。

## 改造前的备份

之前部署版本为 `df4eb6a`；当时全部未提交代码已保存到 `322f504`，并打标签
`snapshot/pre-frozen-task-20261021`，已推送 GitHub。它保存的是改造前状态，包含
先前通用批量/反馈草稿，不能把那时尚未进行的物理验证算作已完成。
