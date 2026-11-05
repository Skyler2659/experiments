# WorldRUA 交接说明

WorldRUA 是最新 Ours + PyRUA 的项目名。代码以服务器最后一批已验证的 RoboDojo
版本为基线，保留 CodeWorld 状态维护、碰撞检查、提前放弃审查和模型断连续跑。
已撤回的槌头校正补丁没有包含进来。没有混入本地另一个方向的 LIBERO 改动。

## 学长需要准备的环境

- Linux + NVIDIA GPU；之前验证过三张 L20、每卡两个样本。
- 已能独立启动的 RoboDojo、XPolicyLab、Isaac Sim 和官方资产。
- Python 3.12 的 Agent 环境，以及独立的 Node/Chromium 重建运行环境。
- 在服务器上单独登录的 Codex 账号；登录目录放在仓库和实验输出之外。

先按英文 [安装说明](SETUP.md) 安装，再复制 `configs/robodojo.example.json` 为
`configs/batch.local.json`，填写路径、任务和样本编号。任务名称必须与 RoboDojo 配置文件
一致；样本编号表示 layout，`eval_seed` 是布局集合编号。

```bash
worldrua doctor configs/batch.local.json
worldrua stage configs/batch.local.json
worldrua resume runs/robodojo_batch --detach
worldrua dashboard runs/robodojo_batch --port 5191
```

`stage` 不启动仿真器。也可以用 `run CONFIG --detach` 一次完成新建和启动。
批次目录已经存在时用 `resume`，不要再次 `run` 覆盖它。

浏览器通过 SSH 隧道查看服务器的 localhost 端口。需要实时世界对比图时，另一个终端运行：

```bash
worldrua comparisons runs/robodojo_batch
```

## 续跑与结果

断网后不重启机器人，先看进程和 `recovery/`。`resume` 接管原进程；模型连接失败会
等待后恢复同一个会话。仿真器本身被杀死时不能无损恢复，代码会标记中断，不自动
把物理任务重跑一遍。`FINISHED` 是正常结束，不等于成功，成功看 `success` 和原生评测。

每卡三个并行尚未压测。普通任务可先按重建 20–45 分钟、执行 10–50 分钟估算，
长序列任务单独预留时间。输出目录包含实测计时，历史等待不要混入有效耗时。

## 交付产物

```bash
worldrua export runs/robodojo_batch/stack_bowls_s0 --bundle --videos
```

会导出带材质的当前场景 GLB、原生任务录像、日志和世界状态归档，以及逐观测帧的
任务/CodeWorld 对比视频。重建画面隐藏机器人和相机模型。任务＋Agent 日志的视频合成
尚未实现，原始 Agent 日志已经保存。细节和文件布局见 [产物说明](ARTIFACTS.md)。

本仓库的安装、单测和导出检查不等于在新服务器上已经完成机器人实验。
首次迁移应先做小规模环境验证，再扩展完整批次。已有实验目录不会被本项目改动。
