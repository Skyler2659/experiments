# CodeWorld

以下命令均在仓库根目录执行；首次使用先完成文末的依赖安装。

## 接入 API Key

配置支持图片输入和 Responses API 的服务，默认模型为 `gpt-6-astra` / `high`。服务地址不带 `/responses`；Codex 登录额度不能代替 RoboProbe 所需的 API Key。

```bash
export OPENAI_API_KEY='你的API密钥'
.venv-benchmark/bin/python experiments/robodojo/configure_api.py --base-url https://你的服务/v1
```

## 启动实验

首次初始化并检查环境，然后选择单个实验或批量运行。单个实验参数依次为：策略、任务、样本（1–50）、GPU。

```bash
PY=.venv-benchmark/bin/python
$PY -m robodojo_benchmark init --output campaigns/main  # 仅首次执行
$PY -m robodojo_benchmark doctor --campaign campaigns/main --runtime experiments/robodojo/runtime.local.json

# 单个实验
bash experiments/robodojo/run_one.sh ours build_tower 1 0

# 或：批量运行尚未完成的队列（每块 GPU 一个 worker）
$PY -m robodojo_benchmark worker --campaign campaigns/main --runtime experiments/robodojo/runtime.local.json --gpu 0
```

## 实验结果

结果位于 `campaigns/main/runs/<任务>__<样本三位数>__<策略>/attempt_001/`。查看汇总：

```bash
.venv-benchmark/bin/python -m robodojo_benchmark summary --campaign campaigns/main
```

```text
result.json                         运行结果与官方评分
agent/、episode/robot_io/            Agent 记录、动作请求和回执；RoboProbe 记录在 trace/
episode/source/、episode/geometry/   CodeWorld 生成代码与 3D 资产
artifacts/task.mp4                   任务视频
artifacts/codeworld_task_sync.mp4    CodeWorld 与任务同步视频
artifacts/agent_task_sync.mp4        Agent 操作与任务同步视频
artifacts/manifest.json              视频导出状态
```

无 CodeWorld 的策略不生成世界资产及其对比视频；GT 使用原生几何和通用世界接口，没有重建生成代码。同步视频中的世界状态随观测更新。

## 依赖与安装

需要 Linux、NVIDIA 驱动、支持 Isaac Sim 5.1 相机渲染的 GPU、Git/Git LFS、uv、Node.js/npm 和下载网络；H 卡兼容性需实机确认。安装器准备 Python 环境、Codex CLI、重建和视频依赖，并生成 `runtime.local.json`。

```bash
# 已有 RoboDojo / Isaac Sim 5.1 / IsaacLab 环境和官方 Assets
bash experiments/robodojo/install.sh --sim-python /你的环境/bin/python --assets /你的Assets目录

# 或：自动安装仿真环境并下载资产
bash experiments/robodojo/install.sh --bootstrap-sim
```

## 实验与策略

RoboDojo **14 个任务 × 4 种策略 × 50 个样本，共 2,800 次**。配置在 `experiments/robodojo/suite.json`；修改任务、样本或策略后初始化新的 campaign。

```text
insert_tubes          build_tower           pack_objects_into_box
hang_mugs             fill_egg_holder       organize_table
arrange_largest_number  swap_T              store_tools_in_toolbox
sweep_blocks          play_stacking_toy     fill_pen_holder
stack_bowls           deposit_coin
```

| 策略 ID | 内容 |
|---|---|
| `roboprobe_l3` | RoboProbe-L3：原生 RGB＋机器人状态 |
| `ours_no_codeworld` | Ours − CodeWorld：RGB-D，无持久三维世界 |
| `ours` | Ours：RGB-D 重建＋CodeWorld |
| `gt_codeworld` | Ours＋GT CodeWorld：原生几何，观测时更新真实位姿 |

四组使用相同的公开任务 recipe，沿用官方布局、动作预算和评分；Ours 系列使用增强 RGB-D 观测，GT 额外获得真实几何与状态。
