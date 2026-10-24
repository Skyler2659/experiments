## 当前 14 任务交接版本

默认配置已改为 14 任务 × 50 样本 × 4 策略，共 2800 次；模型 gpt-6-astra / high。新增逐帧时间轴、CodeWorld 状态快照及两种同步视频自动导出。四策略离线视频测试覆盖帧数、分辨率、实际请求关联、无世界分支及历史缺失时间轴的处理。新录像链路尚未完成真实 Isaac Sim 端到端验收，以下原生验证仅针对之前的版本，不能替代此次验收。未启动正式 2800 次实验。

## 2026-10-24 全局接口修复补充

固定 X5 底座与桌面的安装接触现由 RoboDojo 共享碰撞初始化处理，覆盖所有任务、Ours 和 GT，以及基准和旧版直接运行入口。活动部件、夹爪、物体、自碰撞和双臂碰撞检查保留。实际保存的 Ours/GT 初始几何离线复查均由底座 COLLISION 变为 CLEAR，没有执行机器人动作。

公开任务说明漏传的补丁也是共享加载逻辑；全部 42 个发布任务及 `_random` 映射与 RoboProbe 原始加载器一致，当前选定的十个任务均覆盖。相关回归检查 46 项通过、2 项因独立测试环境缺少 USD 模块跳过。本次修复不改变已启动 episode 的源码快照，也不把离线检查当作任务成功。

以下为此前交接验证记录，其测试计数和临时服务状态是当时的记录。

# 验证记录 — 2026-10-24

正式计划为 **500 个配对样本、2000 个 episode，全部待运行**。以下是独立验证 campaign 的工程结果，不是正式实验成功率。机器为 **NVIDIA L20 / Isaac Sim 5.1 / Codex CLI 0.160.0**；模型为 `gpt-6-astra`、`medium`。马跃的 H 卡尚未验收。

## 已完成检查

| 检查 | 结果 |
|---|---|
| 本地机器人、CodeWorld、调度回归 | 219 passed，13 subtests passed |
| RoboProbe 原版测试 | 358 passed，85 skipped，9 subtests passed |
| 独立解压包：控制环境测试 | 217 passed，2 skipped，13 subtests passed |
| 独立解压包：重建模块测试 | 57 passed，2 subtests passed |
| 10 个任务的 14 个原生任务/变体探针 | 全部获得原生结果，均为 0 次任务动作 |
| 独立解压包实际安装 | 使用已有 Isaac Sim 和官方 Assets，安装模拟器依赖隔离层和控制环境；doctor 与真实 GT 场景探针通过 |
| RoboProbe 原生接口 | 3 张 RGB、14 维动作绑定、原版 EEF planner 导入和原生评分通过；未调用模型 |

Linux 控制环境跳过的两项合成 USD 测试已在本地 `usd-core` 环境通过；原生模拟器另实际验证了 USD 导出。14 个探针覆盖全部 10 个任务，以及 Pack Objects、Hang Mugs、Arrange Largest Number、Sweep Blocks 的随机变体。Fill Egg Holder 实际导出了独立的盒身和活动盖子连杆。探针只证明初始化、接口、几何导出与评分通路，不计入任务成功率。

官方资产固定到 `35efbc7dedfdbeeb6e95fb749bd885d73d483e41`。RoboDojo 的 `env/src/task/utils/env_cfg` 与 RoboProbe 的 `policy/utils/model_client` 已与各自固定提交比较，原版评分及策略源码未改。

## 独立 CLI 的真实短时验证

三项均为新的 Build Tower 发布样本 1，官方动作上限保持 **1050**。监督会话只管理生命周期，任务程序和动作由独立 CLI 生成。

| 策略 | 执行窗口 | 官方动作 | 官方分数 | 官方成功 |
|---|---:|---:|---:|---|
| Ours − CodeWorld | 600 秒 | 748 / 1050 | 10 / 100 | false |
| GT CodeWorld（live_state） | 600 秒 | 548 / 1050 | 0 / 100 | false |
| Ours（最终新采集验证 r06） | 360 秒 | 188 / 1050 | 0 / 100 | false |

三项均在预先声明的验证墙钟预算结束后取得原生评分。它们证明实际执行和评分链路能运行，**不证明任务成功或方法有效性**。正式 suite 的执行窗口为 14400 秒，Ours 重建另有 14400 秒；短时验证未占用正式样本结果。

最终 Ours 完整经过：同一次官方 reset → 21 视角 RGB-D 采集（0 次官方动作）→ 8 个目标对象的 RGB 标注/SAM2/测量 → 完整递归重建与自动掩码纠错 → CodeWorld 导出 → 独立 CLI 动作与状态更新 → 官方评分。重建保留 depth=4、每节点 8 cycles 和自适应反馈/纠错机制；没有使用历史重建资产或替代简化方法。内部 FIT 检查通过，10 个导出对象的源程序/几何绑定检查通过。

最终动作审计：188 条原生动作编号连续；12 次运动请求均绑定 World 决策；有 5 次 `validated_rgbd_update`；监督者唯一请求是生命周期 `finish`；公开动作回执没有 `official_success`。模型有 3 次夹爪模式/位姿格式错误，接口返回错误后仍可处理后续合法请求，未将这些请求当作成功动作。内部 FIT 与绑定检查不等于独立几何精度或机器人任务成功。

GT 验证记录了 24 次 oracle 状态刷新，实际物体位姿发生变化，29 次运动请求均绑定 World 决策。无 CodeWorld 验证没有 world.json 或目标几何目录，125 次接口请求来自独立 CLI。

## 已修复的问题

- 使用发布 slot、标准/随机变体及不稳定布局替补；不再把样本号误当统一的布局 0–49。Fill Egg Holder 的布局 50–53 已保留。
- 官方 `run_eval` 接管评分和视频收尾，区分原生动作、接口请求与物理 tick，隐藏内部成功标志。
- 分开 CUDA 逻辑设备与 Vulkan 物理 GPU 编号，修复非零 GPU 上的渲染设备选择。
- 加入每 episode 的新采集、新重建和独立 CLI 上下文；补齐可移植的测量工具源码。
- 修复 Codex 多图参数吞掉 prompt、导出进程未继承 Playwright 浏览器路径、重建子进程超时残留、临时认证刷新丢失等问题。
- GT 保留原生凹形表面和孔洞，密集三角面使用批量处理；关节连杆独立导出并更新当前位姿及历史。
- 无 CodeWorld 策略使用直接 RGB-D 接口，不再被 CodeWorld 完成契约阻塞；保留机器人自身和双臂碰撞检查。
- 修复解压安装时 NumPy 被升到 2.x 引起 Isaac Sim 启动失败；模拟器层固定到官方 Docker 的 NumPy/SciPy/OpenCV 版本，并增加诊断。
- 固定 manifest、互斥领取、多 GPU 锁、明确的基础设施重试和独立探针标记，保留每次尝试，不覆盖已评分失败。

## 交接后的现场验收

1. **H 卡型号、驱动和真实渲染。** 目前仅在 L20 验证，不能据此确认未知 H 卡兼容。
2. **RoboProbe 模型 episode。** 原版 L3 需要 Responses API 地址和 Key；Codex 登录额度不能直接作为这个 Key。接口探针通过不等于 API 模型实跑通过。
3. **空白机器安装。** 已实际验证 `--sim-python ... --assets ... --skip-reconstruction` 路径；`--bootstrap-sim` 和从空白环境安装全部重建依赖尚未完整验收。完整 Ours 已在实际 Linux 重建环境运行。

早期验证发现的失败日志保留原状，均未被当作正式实验结果。网络盘上的 bubblewrap 权限问题在验证机通过显式 `reconstruction_sandbox: danger-full-access` 配置解决；交付默认仍为 `workspace-write`。早期探针及 GT/无 CodeWorld 短时验证因临时模型代理出现在线机器人材质加载警告；直连材质后的解压安装探针和最终 Ours r06 均没有 `USD_MDL` 错误。正式 RGB 对比应使用材质完整的配置。

结构化证据见 [validation-evidence.json](validation-evidence.json)。原始日志和审计保存在监督工作区的 `reports/robodojo_handoff_20261024/` 与隔离的 Linux 验证目录。临时认证快照已删除，模型代理和 SSH 转发已停止；交接包不含认证文件、本机 runtime 配置、历史 episode 或重建资产。
