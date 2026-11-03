# 视觉尖端运动诊断辅助代码

`integrations/skills/visual_motion_guard.py` 提供不发动作的函数
`visual_motion_consistency(before, after, **limits)`。它检查一次已完成运动的
前后公开测量是否出现“EE 明显移动、视觉尖端近乎不动”的证据。
返回值不授权纠偏，不把 EE 当成尖端，不判断抓取或原生任务成功。

这是新增的 **supplied diagnostic**，不是 Playground 学到的技能，也没有机器人
验证。本次修改不改动旧实验、旧冻结知识或其分数，不包含场景/seed 常量。

## 输入与前提

每个样本需要以下字段。所有向量是普通列表/元组，NumPy 数组应先转成列表。

|字段|含义|
|---|---|
|`ee_position_m`, `tip_position_m`|测得的三维位置，单位米，不是目标指令或图像像素|
|`ee_orientation_wxyz`|测得的单位四元数，允许 q 与 -q 等价|
|`ee_arm`, `tip_arm`|相同手臂身份，前后必须保持|
|`tip_id`|前后一致的实体尖端/指垫身份，不能用未经匹配的深色连通域冒充|
|`ee_frame`, `tip_frame`|各信号的固定坐标系标识，前后不得变化|
|`ee_capture_time_s`, `tip_capture_time_s`|传感器采集时间，不能用 Python 读取时间或模型完成时间代替|
|`clock_id`|采集时间与 `now_s` 使用的同一个时钟域|
|`units`|必须为 `m`；其他单位应在有依据的适配中转换|
|`timestamps_verified`|采集时间的来源、时钟对齐确有依据才填 true|
|`stationary_frames`|两个坐标系在该区间保持固定，且长度尺度已校准才填 true|
|`rigid_tip`|该尖端与 EE 的相对几何在区间内固定才填 true|

例如移动的腕相机、未知外参更新、开合中的指尖、跟踪对象切换均不满足上述前提。
两个固定坐标系可以有不同原点和轴方向，因为只比较位移长度；它们不能有未知比例
或非刚性畸变。三项 true 是调用方提供的证据声明，函数不能凭数值自己证明它们。
现有 `acquired_at_monotonic_s` 只代表宿主读取时间，**不能**直接填充采集时间字段。
缺少可信采集时间或刚性条件时应保留 unknown，不补造时间戳。

所有 `limits` 都显式传入，不使用来自失败 Test 场景的默认阈值：

|参数|需要确定的边界|
|---|---|
|`now_s`, `clock_id`|当前评估时刻与其时钟域|
|`max_age_s`, `max_pair_skew_s`, `max_interval_s`|最大数据年龄、同一端点的两信号时间差、前后间隔|
|`min_ee_motion_m`, `max_static_tip_motion_m`|足够激励的 EE 位移、近乎静止的视觉位移|
|`position_error_bound_m`|每个位置端点的误差上界，包含标定误差与时间对齐误差|
|`orientation_error_bound_rad`|每个 EE 姿态端点的角误差上界|
|`tip_offset_bound_m`|EE 到该刚性尖端的距离上界，不是将二者视为同一点|
|`max_rotation_rad`|允许用此诊断的姿态变化上界|
|`max_position_norm_m`|两个坐标系中位置长度的合理上界，用于拒绝错误尺度|

同一端点的时间差在阈值内仍不足以自动证明同步：必须有共时采样，或用已知运动
上界将该时间差造成的位移误差计入 `position_error_bound_m`。无法给出误差/几何
上界时，不应调用此诊断得出静止判断。参数不合法抛出 ValueError；缺失、非有限、
形状/单位/尺度不合法或前提未验证的观测返回 unknown。

## 判断范围

若测得姿态变化角为 θ，每端姿态角误差上界为 εq，则真实角变化上界为
`min(π, θ + 2εq)`。超过允许姿态变化时返回 unknown。即使未超过，长度为 L 的
EE–尖端刚性偏置也可能产生最多 `2L sin(θ上界 / 2)` 的旋转位移。

从 EE 位移减去这个旋转项及四个位置端点的误差上界后，若仍能要求尖端发生明显
移动，而视觉位移低于静止阈值，才返回 `suspected_static_visual_tip`。
不足够的平移激励、过大误差或可能由旋转抵消的位移都返回 unknown。

`no_static_tip_evidence` 只表示本次没有命中这个静止检测，不验证运动方向、视觉
检测身份、目标误差、接触安全或成功。此函数没有 `allow_visual_correction` 字段。
所有返回值的 `task_success_verified` 都为 false。

## 新 Playground 中接入

`stage_workspace()` 会把 helper 复制到新输入工作区的 `skills/`，并在
`skills/SUPPLIED.md` 标明其来源和边界。模型必须在新 Playground 中明确包含、
执行和验证自己的集成，随后按原协议冻结。没有自动注入旧任务状态机。

任务清单可以使用：

```json
{"task": {"includes": ["skills/visual_motion_guard.py"], "entry": "skills/task_program.py"}}
```

以下是自有任务包装代码里的最小接入位置示意。`previous_sample`、`current_sample`
和 `limits` 必须由当前公开观测与校准构建，不应把示例值固化进 Test。
在下一次动作之前复核，而不是等纠偏多次后才报告。

```python
def check_visual_motion(state, previous_sample, current_sample, limits):
    evidence = visual_motion_consistency(previous_sample, current_sample, **limits)
    state['visual_motion_guard'] = evidence
    if evidence['status'] in ('unknown', 'suspected_static_visual_tip'):
        state['status'] = 'needs_system2'
        state['reason'] = 'visual_motion_guard:' + evidence['reason']
    return evidence

# 在 skill_context 返回状态和上述 evidence；暂停状态下 skill_options 返回 {}。
# 在 skill_tick 的动作分支前重新检查当前状态，避免缓存动作绕过诊断：
if state['status'] != 'active':
    return {'diagnostic': state.get('visual_motion_guard'), 'action_applied': False}
```

运行时已经会将 `context.status != active` 交回 System 2；该路径无需再调用选择器
来决定是否停。`no_static_tip_evidence` 不得清除既有失败状态或作为允许动作的依据，
原有目标、预算、接触及观测验证仍需满足。首个样本或缺少前后配对时为 unknown；
该辅助代码不能为获得诊断数据而自行制造动作或重放不确定动作。

没有向 `library.json` 添加虚假的可编译条目：现有库的每项必须具有完整四入口，
而此文件是无动作的组合 helper。未来任务可以显式 include 它；staging、原协议的
source/frozen 哈希会保存实际使用版本。不得把 helper 的可用性当成调用证据。

## CPU 验证

```sh
python -m unittest discover -s tests -p test_visual_motion_guard.py -v
```

测试涵盖缺失条件、手臂/帧/跟踪身份变化、采集时间、单位尺度、非有限值、四元数
正负等价、旋转抵消、误差过大及输入不变，并将 staged 源码放进真实代码沙箱验证
其可执行且没有机器人原语。测试不调用模型、GPU 或仿真，不证明真实视觉检测可靠。
