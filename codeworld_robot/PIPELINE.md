# 当前执行流程

重建资产 → CodeWorld 文件与几何 API → 独立 Codex CLI 编程决策 → 原生机器人控制器 → 图像与机器人反馈 → Codex 按需要更新 CodeWorld。

机器人控制由 capabilities 明确区分：

- LIBERO 默认保留 OSC 末端控制；此模式不承诺完整关节路径碰撞检查。
- LIBERO-PRO Panda 可显式选择 joint_position，使用原生 JointPositionController 的绝对 set_qpos 目标和原生物理动力学。
- FetchBench FR3 使用 franka_r3_cvx 的七关节原生位置控制；不依赖旧 CaP-X 桥接。
- RoboDojo X5 保留原有接口和可选路径检查。

关节模式共享机器人模型绑定、IK 候选、轨迹生成和离散采样碰撞检查。检查当前重建场景、机械臂、自碰撞、夹爪扫掠及显式声明的携带物。它不能证明连续碰撞安全，也不会自动识别被抓住的物体。接触例外必须明确指定物体或连杆对。

Panda 从实际加载的公开机器人模型导出几何，TCP 遵循 LIBERO 的位置／方向观测定义；FR3 读取公开原生 URDF。初始化时 FK 必须与原生机器人观测一致，否则停止启动。

步数须连同计数域报告：Panda 为 native_policy_cycle；FetchBench 为 physics_tick，不能直接称为论文中的策略动作次数。check_ik、check_motion 不推进仿真。

Codex 自主选择计算、观测、动作及世界更新。工具不包含跨任务经验积累，也不复用历史任务动作脚本。刚体位姿和局部形状更新仍由 Agent 根据观测决定，接口返回观测不会自动触发重建。
