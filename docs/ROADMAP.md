# 学习与开发路线图

总时间 1 周。先用 2~3 天做三个基础练习（三个任务都要用到），再选定任务集中实现。
每完成一项就打勾、提交一次 git。

## 阶段 0：环境 ✅
- [x] 安装 Isaac Sim 6.1，首次启动
- [x] `scripts/00_hello_franka.py`：场景 + Panda 摆姿态 + 抛球（认识 Stage / Prim / Articulation / RigidPrim / 主循环）
- [x] git + GitHub 仓库

## 阶段 1：三个基础练习（2~3 天）

### 练习 1：抛体轨迹预测（半天）→ `scripts/01_trajectory.py`
- [x] 实现 `predict_position`（已知 p0、v0、t → 位置）
- [x] 实现 `launch_velocity`（已知 p0、目标点、T → v0）
- [x] 跑通：连续公式误差随 t 线性增长（≈8 cm/s）；按半隐式欧拉离散积分补偿 ½·g·dt 后误差降到 0.00 cm
- [ ] 思考题：
  1. ✅ 误差为什么不是 0？→ PhysX 用半隐式欧拉（先更新速度再更新位置），每秒多掉 ½·g·dt；由误差斜率反推出 dt = 1/60 s
  2. 分别用 `--T 0.4` 和 `--T 1.0` 跑。T 变小时速度和弧线最高点怎么变？接球 / 投篮时 T 选大还是选小？
  3. 附加：不知道 v0、只看得到前 5 帧的位置，能否把 v0 估出来？（接球任务的第一步）

学到的：**轨迹预测**。正着用是接球（预测球会经过哪里），反着用是投篮（算出手速度）。

### 练习 2：逆运动学 IK，让末端到达指定位置（1 天）
- [ ] 运行官方示例 `D:\isaacsim\standalone_examples\api\isaacsim.core.experimental.api\control_robot_numpy.py`
      （Franka + 纯 numpy 写的微分 IK，逻辑全部可见。注：`tutorials\manipulation\tutorial_9_follow_target.py` 用的是 UR10e，不是 Franka）
- [ ] 读懂它是怎么把"末端目标位置"变成 7 个关节角的（雅可比矩阵 + 阻尼最小二乘，每帧迭代一小步）
- [ ] （可选）拖动版：`D:\isaacsim\standalone_examples\api\isaacsim.robot_motion.examples\manipulation\follow_target.py --robot franka`（cuMotion RMPflow，黑盒但带避障）
- [ ] 写 `scripts/02_ik_reach.py`：让夹爪移动到练习 1 的绿点，并打印末端的实际位置和误差
- [ ] 测一下：从 ready 姿态移动到绿点要多久？（接球任务的时间预算）

学到的：**末端到达指定位置**。三个任务都要用。

### 练习 3：夹爪开合（半天）
- [ ] 运行 `tutorial_9_gripper_control.py`，读懂手指的 2 个自由度怎么控制
- [ ] 写 `scripts/03_gripper.py`：夹爪移到球的位置 → 合上 → 抬起，确认球被夹住没有滑掉

学到的：**抓住与松开的时机**。接球（合上）和投篮（松开）都需要。

（可选）`tutorial_9_arm_trajectory.py`：让关节沿平滑轨迹运动。做投篮时会用到。

### 🔀 决策点：选定任务
做完三个练习后，根据手感和剩余时间选定一个任务，并在这里写下选择的理由。

## 阶段 2：实现任务（约 3 天）
- [ ] 搭场景（篮子 / 瓶子 / 抛球器），并加入随机初始条件
- [ ] 核心逻辑（接球：观测 → 预测拦截点 → IK 移动 → 合爪；投篮：抓球 → 挥臂 → 按时松开；滚球：推球 → 瞄准瓶子）
- [ ] 定义"成功"的判定条件，并在代码里自动判定

## 阶段 3：评估（约 1 天）
- [ ] `--headless` 批量跑 N 次随机实验，统计成功率
- [ ] 归类失败原因（够不着 / 来不及 / 夹滑了 / ……），每类附一两个例子

## 阶段 4：交付
- [ ] 录演示视频（成功案例 + 典型失败案例）
- [ ] README：方法、成功率、失败分析、如何运行
