# WorldRUA

Ours + PyRUA：让机器人 Agent 使用可查询、可更新的 CodeWorld。

## 1. API 密钥配置

在实验服务器安装 Codex CLI 后，使用独立认证目录（[官方认证说明](https://learn.chatgpt.com/docs/auth)）：

```bash
export WORLD_RUA_AUTH_HOME="$HOME/.worldrua-auth"
export OPENAI_API_KEY="你的 API key"
printenv OPENAI_API_KEY | CODEX_HOME="$WORLD_RUA_AUTH_HOME" codex -c 'cli_auth_credentials_store="file"' login --with-api-key
```

运行脚本只清理各样本的认证副本，不退出原账号。

## 2. 环境部署与实验运行

Linux + NVIDIA，推荐 Python 3.12。先按[环境准备](docs/SETUP.md)安装 RoboDojo/Isaac Sim、Isaac Lab、任务资产及 Codex CLI；Agent 与仿真器使用各自的 Python 环境。首次获取代码：

```bash
git clone https://github.com/Skyler2659/experiments.git WorldRUA
cd WorldRUA
```

在仓库根目录运行（CUDA 12.8 的已用版本组合）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints-tested-linux.txt -e '.[dev,collision]'
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/install_sam3.py
worldrua setup-runtime ./runtime --python python3.12
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_PYTHON_BIN=/path/to/robodojo-env/bin
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export WORLD_RUA_RUNTIME="$PWD/runtime"
cp configs/robodojo11.example.json configs/batch.local.json
worldrua doctor configs/batch.local.json
worldrua run configs/batch.local.json --detach
worldrua dashboard runs/robodojo11 --port 5191
```

SAM3 权重下载、Chromium 系统库及仿真环境配置见[环境准备](docs/SETUP.md)；三个 `/path/to/…` 必须替换为实际路径。`doctor` 检查依赖、CUDA、任务布局和浏览器渲染，通过后再运行；模型权限与完整仿真启动仍需在目标机器确认。

默认三张 L20、每卡两个样本，共六路并行。服务器本地查看 `http://127.0.0.1:5191/`，远程访问需 SSH 转发。模型断连会保留存活的仿真器续跑；队列进程退出后执行 `worldrua resume runs/robodojo11 --detach`。仿真器已死亡则不能原地恢复，见[续跑说明](docs/RECOVERY.md)。

## 3. 实验产物与导出路径

每个样本保存在 `runs/robodojo11/<task>_s<sample>/`。例如：

```bash
worldrua export runs/robodojo11/build_tower_s0 --bundle --videos
```

| 内容 | 样本目录内路径 |
|---|---|
| 带材质的当前 3D 场景，隐藏机器人与相机 | `exports/current_scene.glb` |
| 初始场景（存在时）、世界代码/资产、Agent/CodeWorld 日志及观测 | `exports/initial_scene.glb`、`exports/artifacts.tar.gz` |
| 连续任务视频 | `episode/bench/eval_result/`，导出时复制到 `exports/` |
| 真实画面 + 同步历史 CodeWorld 视频 | `videos/comparison.mp4`（观测序列，2 fps） |
| 评分、耗时和原始 Agent 记录 | `episode/result.json`、`episode/codex_events.jsonl` |

任务 + Agent 日志的合成视频尚未实现；日志已导出。[完整资产说明](docs/ARTIFACTS.md)。

## 4. 实验任务清单

按成绩 Excel 的 11 类任务配置，见 [robodojo11.example.json](configs/robodojo11.example.json)：

| 任务 | 配置名称 |
|---|---|
| 搭塔 | `build_tower` |
| 插管 | `insert_tubes` |
| 装箱 | `pack_objects_into_box` |
| 按大小排列套娃 | `sort_nesting_dolls_by_size` |
| 叠碗 | `stack_bowls` |
| 交换积木 | `swap_blocks` |
| 排列最大数字 | `arrange_largest_number` |
| 求解方程 | `solve_equation` |
| 套柱玩具 | `play_stacking_toy` |
| 挂杯子 | `hang_mugs` |
| 做吐司 | `make_toast` |

示例每类 `samples: [0,1,2]`，共 33 次；正式样本数修改配置。交换积木使用修复按钮复位的环境，补丁及说明见[环境准备](docs/SETUP.md#qualification)。

## 5. 实验耗时与显存占用

未发现中断记录的阶段均值：重建 **32.4 分钟**，策略推理/执行 **46.1 分钟**（含动作与反馈），全新重建总耗时 **76.6 分钟**；显存平均 **11.60 GiB**、采样最大 **14.39 GiB**（搭塔/挂杯子复用资产运行的整卡占用估计）。
