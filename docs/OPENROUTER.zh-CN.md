# 可选 OpenRouter 双循环适配

`agent.preset = "openrouter"` 让 System 2 通过宿主 OpenRouter API 编写程序，
`selector.kind = "openrouter"` 让 System 1 通过同一服务选择合法选项。
这是一项新增、显式选择的 `openrouter-only-dual-loop-adaptation-v1` 条件，
不是原生 Jev 复现。既有 Jev、Claude、Codex 配置和默认 supplied helpers 保持不变。

本入口使用现有 Linux 宿主、Docker 隔离和原生 RoboDojo 环境。
它使用 direct runner；OpenRouter 配置会拒绝另设的 XPolicyLab/SIF 执行后端。
仓库不包含操作人员的 Windows/WSL 启动器、机器配置、认证或历史实验队列。
CPU 测试验证协议和故障处理；它们不证明真实网络长期可用或机器人完成任务。

## 配置与启动

先按 [安装说明](INSTALL.md) 准备仓库自身的 RoboDojo 资源和模拟器解释器，
并构建 agent 镜像：

```sh
docker build -t embodiedevo-agent:latest docker/agent
```

模型请求由宿主发出。把认证 JSON 放在仓库外，内容包含 `OPENAI_API_KEY`，
然后只向宿主设置其路径；可选代理也由操作人员自行配置：

```sh
export EMBODIEDEVO_OPENROUTER_AUTH_FILE=/absolute/private/openrouter-auth.json
# Optional: export EMBODIEDEVO_OPENROUTER_PROXY_URL=http://proxy-host:port
```

不要把认证、代理密码或本机路径写进 task 配置或提交到 Git。
离线 shell 容器只挂载工作区和会话 socket；没有 API 密钥、模拟器源码或 GPU。
Test 中 `skills/` 和 `lessons/` 只读，场景修复不能传入下一场。

示例 `configs/dual-loop-openrouter-cover-blocks/task.toml` 为两个循环指定
`deepseek/deepseek-v4.1-flash`、`high`，completion 上限均为 16384；
Playground 和每场 Test 均为 100 次代码执行、2400 秒，原生动作上限由原任务决定。
每次实验应记录源码、配置、实际镜像摘要、资产和模型回执。

```sh
python -m orchestrator explore --task configs/dual-loop-openrouter-cover-blocks \
  --run-dir runs/YYYYMMDDHHMM_openrouter_cover_blocks --gpu 0
# 完成 Playground 并独立核对导出与冻结记录之后，才使用同一个 run-dir：
python -m orchestrator test --task configs/dual-loop-openrouter-cover-blocks \
  --run-dir runs/YYYYMMDDHHMM_openrouter_cover_blocks --gpu 0
```

将占位符换成新时间戳。不要复用已有失败场景来重新执行动作。该入口保留原协议的
预算、任务目标、反馈和评分；传输重试只发生在尚未执行本次模型工具/动作之前。
发生不确定的 shell/session 操作时检查已有记录，不能重放该操作。
Test 布局来自有限资产池：不同 seed 不一定对应不同布局内容，报告时应核对内容哈希。

## 导出与审计

OpenRouter Playground 使用单独的 provider 提示，要求实际委托完整的最终程序后再冻结。
manifest 中显式记录 `selector` 和 `observation_profile`；工作区 CLI 也应使用同一设置：
宿主在委托前检查 provider/model/effort，并记录实际 selector/profile。
OpenRouter 导出必须与这些记录完全一致，且至少有一次原生动作；
仅初始化或以 selector/program/runtime error 结束的匹配尝试不能建立可冻结的执行证据。

```sh
python scripts/env.py delegate submission/development_task.json \
  --selector openrouter --model deepseek/deepseek-v4.1-flash --effort high \
  --timeout 120 --observation-profile proprioception-v1
```

普通 Test 直接执行冻结程序；只有有界场内失败才启动 System 2 修复。
修复必须遵守原有 task/entry contract，并继承冻结的 selector 配置。
程序运行、模型自报完成和记录完整都不能代替原生成功检查。

S1 的 API 回执保存在委托 `system1-api.jsonl`，S2 在 `agent/openrouter-session.jsonl`
或场内 `repairs/*/agent/` 下保留回执、错误及 usage。费用应按实际回执分类，
不能把未知费用算成零；截断或不符合协议的已付费响应也要保留。
输出被截断、模型字段无效、完整 API 错误与传输响应不完整应分别归因。

传输恢复有总时限：S1 默认最多 2 次请求，共享原 120 秒请求预算与委托剩余时间；
S2 单次逻辑请求最多 8 次尝试、600 秒恢复窗口，并受整场剩余时间约束。
退避和 `Retry-After` 消耗同一预算，不增加代码执行或原生动作额度。
证书校验错误、认证错误、错误配置和不完整/不符合协议的模型决策不会变成可执行动作。
到达时限后返回的响应不能再应用工具；底层读取仍受现有 socket 与 session watchdog 管理，
不把读取超时当作确定的操作回滚。本修复不承诺消除代理或服务端的 TLS 故障。

```sh
python -m unittest discover -s tests -v
```
