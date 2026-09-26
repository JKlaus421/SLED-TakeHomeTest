"""04_swing.py —— 投篮第 1 步：空手挥臂，测手能甩多快

做的事情：
  1. 手臂从 READY 摆到"后摆"姿态 WINDUP_POSE，等它停稳
  2. 把肩、肘、腕（q[1]、q[3]、q[5]）的目标一次性设成 FOLLOW_POSE，让关节电机全速转过去
     —— 关节空间控制：不算 IK，不管手走什么路线，只管关节转到哪
  3. 每一帧记录两指中心的位置，速度 = 位置差 / DT，找出最大速度、出现的时刻、位置和方向

运行：
  run.bat scripts\\04_swing.py
  run.bat scripts\\04_swing.py --headless
"""

from __future__ import annotations

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
from franka_utils import DT, READY_POSE, make_scene, to_np

# ======================================================================
#  TODO：你来填这两组关节角（单位：弧度，9 个数 = 7 个手臂关节 + 2 根手指）
#
#  只改 q[1] 肩、q[3] 肘、q[5] 腕，其余保持 READY 的值：
#    q[0] = 0      （不转身，正对 +x 方向扔）
#    q[2] = q[4] = 0（不拧上臂、小臂 → 手臂保持在 x-z 竖直平面里）
#    q[6] = 0.785、手指 = 0.04（先不管）
#
#  方向说明（READY 是 [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]）：
#    q[1] 肩：0 = 上臂竖直；负 = 上臂往后仰；正 = 往前倾
#    q[3] 肘：接近 -0.07 = 几乎伸直；越负折得越紧（最多约 -3.07）
#    q[5] 腕：方向我也没把握 —— 这正是要你实验的：先试一个方向，看手腕往哪边翻
#  程序开头会打印每个关节的角度范围，填的值超出范围会被截断并提醒你。
#
#  思路：后摆 = 肩往后、肘折紧、腕往后翻（像扔球前把手拉到脑后）
#        随挥 = 肩往前、肘伸开、腕往前甩（要设得"远"一点：我们要的是途中最快，不是停在那里）
# ======================================================================

WINDUP_POSE = None  # TODO: 例如 [0.0, ?, 0.0, ?, 0.0, ?, 0.785, 0.04, 0.04]
FOLLOW_POSE = None  # TODO

WINDUP_TIME = 2.0  # 秒：摆到后摆姿态并停稳
SWING_TIME = 1.5  # 秒：挥臂过程记录多久
SWING_JOINTS = [1, 3, 5]  # 肩、肘、腕

if WINDUP_POSE is None or FOLLOW_POSE is None:
    raise SystemExit("先在脚本里填好 WINDUP_POSE 和 FOLLOW_POSE（见 TODO 注释）")

# ---------------------------------------------------------------- 场景

arm = make_scene(camera_eye=[0.6, 2.2, 0.9], camera_target=[0.4, 0.0, 0.5])  # 从侧面看，挥臂平面正对镜头
app_utils.play()
simulation_app.update()

# 打印关节的范围和速度上限，方便你选角度
lower, upper = (to_np(x)[0] for x in arm.robot.get_dof_limits())
max_vel = to_np(arm.robot.get_dof_max_velocities())[0]
print("\n关节 | 下限(rad) | 上限(rad) | 最大速度(rad/s)")
for j in range(7):
    print(f"q[{j}] | {lower[j]:9.3f} | {upper[j]:9.3f} | {max_vel[j]:8.3f}")


def clip_pose(name: str, pose) -> np.ndarray:
    """超出关节范围的值截断到范围内，并提醒。"""
    pose = np.array(pose, dtype=float)
    clipped = np.clip(pose, lower, upper)
    for j in np.nonzero(np.abs(clipped - pose) > 1e-6)[0]:
        print(f"[提醒] {name} 的 q[{j}] = {pose[j]:.3f} 超出范围，已截断为 {clipped[j]:.3f}")
    return clipped


windup = clip_pose("WINDUP_POSE", WINDUP_POSE)
follow = clip_pose("FOLLOW_POSE", FOLLOW_POSE)

# ---------------------------------------------------------------- 1) 后摆

arm.reset()
simulation_app.update()
arm.robot.set_dof_position_targets(windup.reshape(1, -1))
for _ in range(int(WINDUP_TIME / DT)):
    simulation_app.update()
q_now = to_np(arm.robot.get_dof_positions())[0]
print(f"\n[后摆] 目标 q[1,3,5] = {np.round(windup[SWING_JOINTS], 2)} | 实际 = {np.round(q_now[SWING_JOINTS], 2)}")
print(f"[后摆] 两指中心在 {np.round(arm.grasp_pos(), 3)}")

# ---------------------------------------------------------------- 2) 挥臂

arm.robot.set_dof_position_targets(follow[SWING_JOINTS].reshape(1, -1), dof_indices=SWING_JOINTS)

print(f"\n{'t(s)':>5} | {'两指中心 (m)':^24} | 速度(m/s) | 仰角(°) | q̇[1,3,5] (rad/s)")
prev = arm.grasp_pos()
best = (0.0, 0.0, None, None)  # (最大速度, 时刻, 位置, 速度向量)
for step in range(1, int(SWING_TIME / DT) + 1):
    simulation_app.update()
    pos = arm.grasp_pos()
    vel = (pos - prev) / DT
    prev = pos
    speed = np.linalg.norm(vel)
    # 仰角：速度方向和水平面的夹角（正 = 斜向上飞，负 = 往下）
    elevation = np.degrees(np.arctan2(vel[2], np.linalg.norm(vel[:2])))
    if speed > best[0]:
        best = (speed, step * DT, pos, vel)
    if step % 3 == 0:
        qd = to_np(arm.robot.get_dof_velocities())[0, SWING_JOINTS]
        print(f"{step * DT:5.2f} | {np.array2string(pos, precision=3):^24} | {speed:8.2f} | {elevation:7.1f} | "
              f"{np.round(qd, 2)}")

speed, t_best, pos_best, vel_best = best
if pos_best is not None:
    elevation = np.degrees(np.arctan2(vel_best[2], np.linalg.norm(vel_best[:2])))
    print(f"\n[总结] 最大速度 {speed:.2f} m/s，出现在 t={t_best:.2f}s，位置 {np.round(pos_best, 3)}，"
          f"速度方向 {np.round(vel_best / speed, 2)}（仰角 {elevation:.1f}°）")

if not args.headless:  # 有窗口时停在最后姿态，关窗口退出
    while simulation_app.is_running():
        simulation_app.update()

app_utils.stop()
simulation_app.close()
