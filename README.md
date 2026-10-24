# CodeWorldv1 实验运行说明

## 提前准备

Linux、NVIDIA 驱动及能运行 **Isaac Sim 5.1 相机渲染**的 GPU/环境（仅知道是 H 卡不足以确认兼容）、`git`、Git LFS、`uv`、Node.js/npm，以及可下载依赖和 RoboDojo Assets 的网络。推荐复用已验证的 RoboDojo/IsaacLab Python；Assets 不包含在源码包中。

在仓库根目录（本 README 所在目录）中执行：

```bash
bash experiments/robodojo/install.sh --sim-python /你的RoboDojo环境/bin/python --assets /你的Assets目录
# 没有现成环境时，安装器也支持 --bootstrap-sim；省略 --assets 会下载官方资产。
```

安装器准备独立的仿真、控制、重建环境、固定版本 Codex CLI 和重建依赖，生成 `experiments/robodojo/runtime.local.json`。视频导出依赖 OpenCV、Pillow 和 imageio-ffmpeg，随控制环境安装。

## 接入 API

需要支持所配置模型、图片输入及 Responses API 的服务。默认模型为 `gpt-6-astra`、推理等级 `high`；具体模型是否开放取决于服务方。只有 Codex 登录额度不能直接运行原版 RoboProbe API baseline。

```bash
export OPENAI_API_KEY='你的API密钥'  # 仅在运行机器设置，不写入实验包
.venv-benchmark/bin/python experiments/robodojo/configure_api.py --base-url https://你的服务/v1
```

该命令同时配置 Agent、重建和 RoboProbe，密钥只从环境变量读取。自定义变量可传 `--key-env NAME`。服务地址不包含 `/responses`。设置依据：[Codex 配置文档](https://developers.openai.com/codex/config-reference/)。

## 开始跑

```bash
PY=.venv-benchmark/bin/python
$PY -m robodojo_benchmark init --output campaigns/main
$PY -m robodojo_benchmark doctor --campaign campaigns/main --runtime experiments/robodojo/runtime.local.json

# 指定策略、任务、样本（1–50）、GPU
bash experiments/robodojo/run_one.sh ours build_tower 1 0

# 批量领取尚未运行的实验；同一 GPU 使用一个 worker
$PY -m robodojo_benchmark worker --campaign campaigns/main --runtime experiments/robodojo/runtime.local.json --gpu 0
$PY -m robodojo_benchmark summary --campaign campaigns/main
```

默认 **14 × 4 × 50 = 2,800 次**。修改 `experiments/robodojo/suite.json` 的 `tasks`、`samples`、`strategies` 后初始化**新 campaign**即可增减任务/样本；已有 campaign 保留冻结配置。worker 可通过 `--strategy`、`--task`、`--sample` 筛选；每台机器独立输出时，应分配不重叠的任务范围。

## 任务和 baseline

|任务 ID|任务|
|---|---|
|`insert_tubes`|Insert Tubes|
|`build_tower`|Build Tower|
|`pack_objects_into_box`|Pack Objects Into Box|
|`hang_mugs`|Hang Mugs|
|`fill_egg_holder`|Fill Egg Holder|
|`organize_table`|Organize Table|
|`arrange_largest_number`|Arrange Largest Number|
|`swap_T`|Swap T|
|`store_tools_in_toolbox`|Store Tools In Toolbox|
|`sweep_blocks`|Sweep Blocks|
|`play_stacking_toy`|Play Stacking Toy|
|`fill_pen_holder`|Fill Pen Holder|
|`stack_bowls`|Stack Bowls|
|`deposit_coin`|Deposit Coin|

|策略 ID|含义|
|---|---|
|`roboprobe_l3`|官方 RoboProbe-L3，原生 RGB＋机器人状态|
|`ours_no_codeworld`|Ours − CodeWorld，RGB-D，无持久三维世界|
|`ours`|Ours，RGB-D 重建＋CodeWorld|
|`gt_codeworld`|Ours＋GT CodeWorld，原生几何；观测时更新真实位姿|

沿用官方布局、动作预算和评分；RGB-D 是明确披露的观测增强。GT 默认 `live_state`，包含几何、身份和状态信息优势。

## 产出资产和目录

每次实验位于 `campaigns/main/runs/<任务>__<样本三位数>__<策略>/attempt_001/`（重试产生新 attempt）。以下路径均相对此目录：

|产出|位置|
|---|---|
|CodeWorld 生成代码|`episode/source/`，递归重建中间代码在 `episode/preparation/runtime/runs/`|
|CodeWorld 描述及状态|`episode/world.json`、`episode/objects/`、`episode/state/`|
|3D 资产|`episode/geometry/`；Ours 重建导出在 `episode/preparation/export/`|
|Agent 操作与模型记录|`agent/events.jsonl`、`agent/final.txt`、`episode/robot_io/requests/` 与 `responses/`；RoboProbe 模型记录在 `trace/`|
|任务视频|`artifacts/task.mp4`；原始录像 `episode/robot_io/continuous.mp4`|
|CodeWorld＋任务同步视频|`artifacts/codeworld_task_sync.mp4`|
|Agent 操作＋任务视角同步视频|`artifacts/agent_task_sync.mp4`|
|逐帧同步依据和世界快照|`episode/robot_io/timeline.jsonl`、`episode/robot_io/world_snapshots/`|
|官方评分、运行结果、产物状态|`evaluator/`、`result.json`、`artifacts/manifest.json`|

视频在实验结束后自动导出。CodeWorld 画面展示该帧最近记录的世界状态，观测间保持不变；Agent 视频展示实际操作请求/原生动作，完整记录保存在日志中。视频按任务帧同步，不包含模型等待时长。无 CodeWorld 和 RoboProbe 分支不生成虚构的 CodeWorld 代码、3D 资产或对比视频；GT 使用原生导出世界和通用代码接口（`episode/tools/codeworld_workspace/`），没有重建生成程序。CodeWorld 视频省略不会随关节更新的初始机器人网格。

单独补导出（不会重跑实验）：

```bash
.venv-benchmark/bin/python -m robodojo_benchmark.artifacts --attempt campaigns/main/runs/build_tower__001__ours/attempt_001
```

`manifest.json` 为 `COMPLETE` 才表示视频导出完成。缺少新时间轴的历史运行无法补造同步视频；仿真启动前失败也不会有任务录像。官方任务成功以 `result.json` 的 `official_success` 为准，与视频导出状态独立。
