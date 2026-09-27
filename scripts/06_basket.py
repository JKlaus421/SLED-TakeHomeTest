"""06_basket.py —— 投篮步骤 C：放一个篮子，按轨迹预测决定什么时候松手

流程（前半段和 05 一样）：
  1. 抓球 → 2. 平滑摆到后摆姿态 → 3. 挥臂
  4. 挥臂时每一帧：用球现在的位置 p、速度 v 算"如果现在松手，球下落经过篮筐口高度时在哪"（crossing_point）
     这个落点会随着挥臂越来越远；当它到达篮筐中心时 → 张开手指 + 刹车
  5. 等球停下，判断是否在篮子里

为什么用"最接近的那一帧"而不是"第一次超过的那一帧"：
  挥臂时预测落点每帧大约往前移 10 cm，只能在整帧上松手。
  如果本帧的落点已经过了"本帧和下一帧的中点"，本帧就比下一帧更接近目标 → 本帧松手。
  下一帧的落点用"本帧 + 上一帧到本帧的变化量"估算。

运行：
  run.bat scripts\\06_basket.py
  run.bat scripts\\06_basket.py --basket-x 1.0
"""

from __future__ import annotations

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--basket-x", type=float, default=0.9, help="篮子中心离机械臂底座多远（米，正前方）")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from franka_utils import (DT, FINGER_CLOSED, FINGER_OPEN, READY_POSE, add_marker, crossing_point, make_scene,
                          to_np)

BALL_RADIUS = 0.03
BALL_MASS = 0.06
PEDESTAL_HEIGHT = 0.10
BALL_START = [0.5, 0.0, PEDESTAL_HEIGHT + BALL_RADIUS]

# 篮子：内部 24×24 cm，墙高 15 cm，墙厚 3 cm（球每帧飞 ~7 cm，墙太薄会被一帧穿过去）
BASKET_INNER = 0.24
BASKET_HEIGHT = 0.15
WALL = 0.03
BASKET_CENTER = np.array([args.basket_x, 0.0])
RIM_Z = BASKET_HEIGHT  # 篮筐口的高度：瞄准"球心下落经过这个高度时，正好在篮子中心上方"

# 挥臂（04/05 的结论）：q[6]=0.785 手指左右排，松手快
Q6 = 0.785
WINDUP_POSE = [0.0, -0.785, 0.0, -0.07, 0.0, 3.14, Q6, FINGER_CLOSED, FINGER_CLOSED]
FOLLOW_POSE = [0.0, 1.0, 0.0, -1.8, 0.0, 1.4]
SWING_JOINTS = [1, 3, 5]
SWING_TIME = 0.4  # 秒：这段时间里没找到松手时机就放弃
SETTLE_TIME = 2.0  # 秒：松手后等球停下

# ---------------------------------------------------------------- 场景

arm = make_scene(camera_eye=[0.4, 2.6, 1.2], camera_target=[0.4, 0.0, 0.5])
Cube("/World/Pedestal", sizes=1.0, scales=[[0.12, 0.12, PEDESTAL_HEIGHT]],
     positions=[[0.5, 0.0, PEDESTAL_HEIGHT / 2]], colors=[[0.5, 0.5, 0.55]])
GeomPrim("/World/Pedestal", apply_collision_apis=True)


def make_basket(center_xy):
    """底 + 四面墙，都只有碰撞没有刚体 → 固定不动。"""
    cx, cy = center_xy
    half = BASKET_INNER / 2 + WALL / 2  # 墙中心离篮子中心的距离
    outer = BASKET_INNER + 2 * WALL
    parts = {  # 名字: (中心位置, 尺寸)
        "Bottom": ([cx, cy, WALL / 2], [outer, outer, WALL]),
        "WallFront": ([cx + half, cy, BASKET_HEIGHT / 2], [WALL, outer, BASKET_HEIGHT]),
        "WallBack": ([cx - half, cy, BASKET_HEIGHT / 2], [WALL, outer, BASKET_HEIGHT]),
        "WallLeft": ([cx, cy + half, BASKET_HEIGHT / 2], [BASKET_INNER, WALL, BASKET_HEIGHT]),
        "WallRight": ([cx, cy - half, BASKET_HEIGHT / 2], [BASKET_INNER, WALL, BASKET_HEIGHT]),
    }
    for name, (pos, size) in parts.items():
        path = f"/World/Basket/{name}"
        Cube(path, sizes=1.0, scales=[size], positions=[pos], colors=[[0.2, 0.4, 0.8]])
        GeomPrim(path, apply_collision_apis=True)


make_basket(BASKET_CENTER)
aim_marker = add_marker("/World/Aim", [*BASKET_CENTER, RIM_Z], radius=0.015)  # 瞄准点（篮筐口中心，纯视觉）

add_marker("/World/Ball", BALL_START, color=(1.0, 0.45, 0.0), radius=BALL_RADIUS)
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)
ball.set_masses([BALL_MASS])

app_utils.play()
simulation_app.update()


# ---------------------------------------------------------------- 工具（和 05 一样）


def ball_state():
    pos = to_np(ball.get_world_poses()[0])[0]
    vel = to_np(ball.get_velocities()[0])[0]
    return pos, vel


def run_ik_phase(target, finger, max_time, tol):
    arm.set_gripper(finger)
    for _ in range(int(max_time / DT)):
        err = arm.ik_step(target)
        simulation_app.update()
        if tol is not None and err < tol:
            break


def move_joints(target, duration):
    """7 个手臂关节平滑插值；手指直接设成 target[7]（不能插值，否则夹紧力归零）。"""
    target = np.asarray(target, dtype=float)
    start = to_np(arm.robot.get_dof_positions())[0, :7]
    arm.set_gripper(target[7])
    n = int(duration / DT)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        arm.robot.set_dof_position_targets((start + s * (target[:7] - start)).reshape(1, -1), dof_indices=list(range(7)))
        simulation_app.update()


def release_and_brake():
    arm.set_gripper(FINGER_OPEN)
    q_now = to_np(arm.robot.get_dof_positions())[0, :7]
    arm.robot.set_dof_position_targets(q_now.reshape(1, -1), dof_indices=list(range(7)))


# ---------------------------------------------------------------- 1) 抓球  2) 后摆

arm.reset()
arm.set_gripper(FINGER_OPEN)
for _ in range(30):
    simulation_app.update()
grasp_center = ball_state()[0]
run_ik_phase(grasp_center + [0, 0, 0.12], FINGER_OPEN, 3.0, 0.01)
run_ik_phase(grasp_center, FINGER_OPEN, 2.0, 0.005)
run_ik_phase(grasp_center, FINGER_CLOSED, 0.5, None)
run_ik_phase(grasp_center + [0, 0, 0.30], FINGER_CLOSED, 2.0, 0.01)
move_joints(WINDUP_POSE, 2.0)
for _ in range(30):
    simulation_app.update()

# ---------------------------------------------------------------- 3) 挥臂 + 闭环松手

target_x = BASKET_CENTER[0]
print(f"\n[瞄准] 篮筐口中心 {np.round([*BASKET_CENTER, RIM_Z], 3)}")
print(f"{'t(s)':>5} | 球速(m/s) | 仰角(°) | 现在松手 → 落到篮筐口高度时的 x (m)")
arm.robot.set_dof_position_targets(np.array(FOLLOW_POSE)[SWING_JOINTS].reshape(1, -1), dof_indices=SWING_JOINTS)

released, prev_x = None, None
for step in range(1, int(SWING_TIME / DT) + 1):
    simulation_app.update()
    pos, vel = ball_state()
    hit = crossing_point(pos, vel, RIM_Z)
    elevation = np.degrees(np.arctan2(vel[2], np.linalg.norm(vel[:2])))
    x_str = f"{hit[1][0]:6.3f}" if hit else "  够不到"
    print(f"{step * DT:5.2f} | {np.linalg.norm(vel):8.2f} | {elevation:7.1f} | {x_str}")
    if hit is None:
        continue
    x = hit[1][0]
    # 估算下一帧的落点 ≈ x + (x - prev_x)；本帧落点已过"本帧和下一帧的中点"→ 本帧更接近目标
    if prev_x is not None and x + (x - prev_x) / 2 >= target_x:
        release_and_brake()
        released = (step, pos, vel, hit)
        print(f"----- t={step * DT:.2f}s 松手（预测落点 x={x:.3f}，目标 {target_x:.3f}）-----")
        break
    prev_x = x

if released is None:
    print(f"\n[结果] {SWING_TIME}s 内没找到松手时机（篮子太远？），不松手")
else:
    # ---------------------------------------------------------------- 4) 看球落在哪
    _, p_rel, v_rel, (t_hit, p_hit) = released
    actual_cross = None
    prev_pos = ball_state()[0]
    for step in range(int(SETTLE_TIME / DT)):
        simulation_app.update()
        pos = ball_state()[0]
        if actual_cross is None and prev_pos[2] >= RIM_Z > pos[2]:  # 这一帧往下穿过了篮筐口高度
            actual_cross = pos
        prev_pos = pos

    final = ball_state()[0]
    inside = (np.all(np.abs(final[:2] - BASKET_CENTER) < BASKET_INNER / 2) and final[2] < BASKET_HEIGHT)
    print(f"\n[预测] 经过篮筐口高度时 x={p_hit[0]:.3f} y={p_hit[1]:.3f}（飞行 {t_hit:.2f}s）")
    if actual_cross is not None:
        miss = actual_cross[:2] - p_hit[:2]
        print(f"[实际] 经过篮筐口高度时 x={actual_cross[0]:.3f} y={actual_cross[1]:.3f} "
              f"| 和预测差 {np.linalg.norm(miss) * 100:.1f} cm | 离篮筐中心 "
              f"{np.linalg.norm(actual_cross[:2] - BASKET_CENTER) * 100:.1f} cm")
    print(f"[结果] 球最后停在 {np.round(final, 3)} → {'投进了！' if inside else '没进'}")

move_joints(READY_POSE, 2.0)
if not args.headless:
    while simulation_app.is_running():
        simulation_app.update()

app_utils.stop()
simulation_app.close()
