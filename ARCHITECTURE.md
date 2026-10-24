# CodeWorld 开发与验收

本 Git 仓库是开发、测试和发布的唯一源码。压缩包由本仓库的
`python -m robodojo_benchmark bundle --output /path/CodeWorld.tar.gz` 生成；不从旧研究目录覆盖源码。

| 模块 | 职责 |
|---|---|
| `robodojo_benchmark` | 配对样本、策略配置、调度、准备、原生结果接收、汇总与资产导出 |
| `codeworld_robot` | 机器人接口、运动与碰撞检查、模拟器生命周期、独立 CLI |
| `codeworld_workspace` | 可编程世界、几何查询、观测证据与状态更新、决策记录 |
| `reconstruction_assist` | 重建测量、反馈与纠错 |
| `agentic_mesh_world` | 重建和几何生成工具 |
| `vendor` | 固定版本的上游实现 |

依赖方向：实验 harness → robot → workspace；重建辅助 → 几何工具。
机器人层通过配置中的 `episode_preparer` 和 `instruction_transform` 接入实验逻辑，
不导入实验包。RGB-D 证据处理位于 `codeworld_workspace.observation_evidence`；
`codeworld_robot.grasp` 仅保留旧接口兼容。

RoboDojo 初始化与原版 baseline 生命周期在 `robodojo_runtime.py`；文件请求队列在
`request_loop.py`；动作分发在 `robodojo_dispatch.py`，闭环状态与运动在 `robodojo_server.py`。

官方 `_result.json` 的有效评分优先于 Agent 退出和后处理状态。已评分失败不能重跑，
视频可单独重新导出。旧版误标的记录使用 `recover` 恢复官方评分；不覆盖旧 attempt。

每个 attempt 的 `source_versions.json` 保存 CodeWorld commit、未提交修改标志、实际源码标识、
上游实际版本与内容检查。实际源码归档复用在 `campaigns/<campaign>/sources/`，以源码标识命名。
Git commit 可能为空（解压包），此时仍保留完整源码快照。上游内容标识用于发现交付后的改动，
并非独立证明其官方来源。凭据与本机 runtime 文件不进入源码归档。

CI 运行四个第一方测试目录，无需 GPU 或 API。真实仿真验收独立进行：接收机器先通过
`doctor` 和 native probe，再对四策略分别完成一个新 episode，核对官方结果、动作预算、
Agent 日志和三类视频。CPU 测试不证明 Isaac Sim、GPU、API 或任务成功。
现有实机证据及未完成项见 `experiments/robodojo/VALIDATION.md`。
