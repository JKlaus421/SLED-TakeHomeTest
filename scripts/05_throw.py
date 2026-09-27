"""05_throw.py —— 投篮步骤 A / B：拿着球挥臂；可选在固定时刻松手，对比预测和实际轨迹

流程：
  1. 抓球（和 03 一样：approach → descend → close → lift，用 IK）
  2. 平滑地摆到后摆姿态（关节空间插值，2 s）—— 不能直接跳过去，否则还没挥球就被甩掉了
  3. 挥臂（和 04 一样：肩、肘、腕目标一次设到随挥姿态）
  4. 每帧记录：球离两指中心多远（> 3 cm 就算"离手"）、球的速度
  5. （步骤 B）挥臂开始 --release-t 秒后张开手指，之后每帧对比：
       实际位置  vs  从"发指令那一帧"的状态预测  vs  从"真正离手那一帧"的状态预测
  6. 慢慢回到 READY

步骤 A 结论：--q6 2.356（手指前后排，后面那根手指兜住球）全程不甩脱；0.785 在 t=0.42s 甩脱。

运行：
  run.bat scripts\\05_throw.py                    （步骤 A：不松手）
  run.bat scripts\\05_throw.py --release-t 0.12   （步骤 B：0.12 s 时松手）
"""

from __future__ import annotations

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--q6", type=float, default=2.356, help="挥臂时夹爪绕自身轴线的转角（rad）")
parser.add_argument("--release-t", type=float, default=None, help="挥臂开始多少秒后张开手指（不填 = 不松手）")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from franka_utils import (DT, FINGER_CLOSED, FINGER_OPEN, READY_POSE, add_marker, make_scene, predict_position,
                          to_np)

BALL_RADIUS = 0.03
BALL_MASS = 0.06
PEDESTAL_HEIGHT = 0.10
BALL_START = [0.5, 0.0, PEDESTAL_HEIGHT + BALL_RADIUS]

# 挥臂姿态（04 推导的）；q[6] 用命令行参数，手指合拢
WINDUP_POSE = [0.0, -0.785, 0.0, -0.07, 0.0, 3.14, args.q6, FINGER_CLOSED, FINGER_CLOSED]
FOLLOW_POSE = [0.0, 1.0, 0.0, -1.8, 0.0, 1.4]  # 只用前 6 个里的 q[1]、q[3]、q[5]
SWING_JOINTS = [1, 3, 5]
SWING_TIME = 0.6  # 秒：不松手时只看这一段（04 里 0.6 s 后手已经很低了）
FLIGHT_TIME = 1.2  # 秒：松手时最多记录多久（球落地就提前结束）
LOST = 0.03  # 球心离两指中心超过 3 cm → 离手

# ---------------------------------------------------------------- 场景

arm = make_scene(camera_eye=[0.6, 2.4, 1.0], camera_target=[0.2, 0.0, 0.7])  # 侧面看挥臂
Cube("/World/Pedestal", sizes=1.0, scales=[[0.12, 0.12, PEDESTAL_HEIGHT]],
     positions=[[0.5, 0.0, PEDESTAL_HEIGHT / 2]], colors=[[0.5, 0.5, 0.55]])
GeomPrim("/World/Pedestal", apply_collision_apis=True)
add_marker("/World/Ball", BALL_START, color=(1.0, 0.45, 0.0), radius=BALL_RADIUS)
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)
ball.set_masses([BALL_MASS])

app_utils.play()
simulation_app.update()


def ball_state():
    """球的 (位置, 线速度)。"""
    pos = to_np(ball.get_world_poses()[0])[0]
    vel = to_np(ball.get_velocities()[0])[0]
    return pos, vel


def run_ik_phase(target, finger, max_time, tol):
    """03 的一个阶段：IK 让两指中心去 target，误差 < tol 提前结束（tol=None 则等满时间）。"""
    arm.set_gripper(finger)
    for _ in range(int(max_time / DT)):
        err = arm.ik_step(target)
        simulation_app.update()
        if tol is not None and err < tol:
            break


def move_joints(target, duration):
    """关节空间平滑插值：7 个手臂关节从当前角度慢慢过渡到 target[:7]。

    s 从 0 到 1 按余弦曲线变化 → 起步和结束都慢，中间快，不会猛地一下把球甩掉。
    手指不参与插值，直接设成 target[7:]：手指被球挡着，实际位置 ≠ 目标，
    这个差值就是夹紧力。如果从"实际位置"开始插值，一开始目标 = 实际，夹紧力归零，球就掉了。
    """
    target = np.asarray(target, dtype=float)
    start = to_np(arm.robot.get_dof_positions())[0, :7]
    arm.set_gripper(target[7])
    n = int(duration / DT)
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        arm.robot.set_dof_position_targets((start + s * (target[:7] - start)).reshape(1, -1), dof_indices=list(range(7)))
        simulation_app.update()


# ---------------------------------------------------------------- 1) 抓球

arm.reset()
arm.set_gripper(FINGER_OPEN)
for _ in range(30):
    simulation_app.update()
grasp_center = ball_state()[0]
run_ik_phase(grasp_center + [0, 0, 0.12], FINGER_OPEN, 3.0, 0.01)  # approach
run_ik_phase(grasp_center, FINGER_OPEN, 2.0, 0.005)  # descend
run_ik_phase(grasp_center, FINGER_CLOSED, 0.5, None)  # close
run_ik_phase(grasp_center + [0, 0, 0.30], FINGER_CLOSED, 2.0, 0.01)  # lift
print(f"\n[抓球] 球离两指中心 {np.linalg.norm(ball_state()[0] - arm.grasp_pos()) * 100:.1f} cm")

# ---------------------------------------------------------------- 2) 后摆

move_joints(WINDUP_POSE, 2.0)
for _ in range(30):  # 停稳 0.5 s
    simulation_app.update()
pos, _ = ball_state()
print(f"[后摆] q6={args.q6:.3f} | 球在 {np.round(pos, 3)} | 离两指中心 {np.linalg.norm(pos - arm.grasp_pos()) * 100:.1f} cm")

# ---------------------------------------------------------------- 3) 挥臂（可选松手）


def elevation_deg(v) -> float:
    """速度方向和水平面的夹角（正 = 斜向上）。"""
    return float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))))


release_step = round(args.release_t / DT) if args.release_t is not None else None
total_time = FLIGHT_TIME if release_step is not None else SWING_TIME
cmd = None  # 发出松手指令那一帧：(帧号, 位置, 速度)
sep = None  # 球真正离手那一帧：(帧号, 位置, 速度)
t_lost, max_speed = None, 0.0

arm.robot.set_dof_position_targets(np.array(FOLLOW_POSE)[SWING_JOINTS].reshape(1, -1), dof_indices=SWING_JOINTS)
print(f"\n{'t(s)':>5} | {'球位置 (m)':^22} | 球速(m/s) | 仰角(°) | 离两指中心(cm) | 预测误差: 按指令时 / 按离手时 (cm)")
for step in range(1, int(total_time / DT) + 1):
    if step == release_step:  # 用"上一帧结束时"的状态作为指令时刻的状态，然后张开手指
        cmd = (step - 1, *ball_state())
        arm.set_gripper(FINGER_OPEN)
        # 刹车：7 个手臂关节的目标 = 它们现在的角度 → 电机全力减速，手臂不会再追上球
        q_now = to_np(arm.robot.get_dof_positions())[0, :7]
        arm.robot.set_dof_position_targets(q_now.reshape(1, -1), dof_indices=list(range(7)))
        print(f"----- t={cmd[0] * DT:.2f}s 发出松手指令（同时刹车）-----")
    simulation_app.update()

    pos, vel = ball_state()
    gap = np.linalg.norm(pos - arm.grasp_pos())
    speed = np.linalg.norm(vel)
    if gap > LOST and t_lost is None:
        t_lost = step * DT
        if cmd is not None:
            sep = (step, pos, vel)
            print(f"----- t={step * DT:.2f}s 球离手（指令后 {step - cmd[0]} 帧）-----")
    if t_lost is None:
        max_speed = max(max_speed, speed)

    # 松手后：两种预测和实际位置的差
    err_str = ""
    if cmd is not None:
        err_cmd = np.linalg.norm(pos - predict_position(cmd[1], cmd[2], (step - cmd[0]) * DT))
        err_sep = np.linalg.norm(pos - predict_position(sep[1], sep[2], (step - sep[0]) * DT)) if sep else np.nan
        err_str = f" | {err_cmd * 100:6.1f} / {err_sep * 100:6.1f}"

    just_released = cmd is not None and step - cmd[0] <= 8  # 松手前后逐帧打印，其余每 3 帧打印
    if step % 3 == 0 or just_released:
        print(f"{step * DT:5.2f} | {np.array2string(pos, precision=3):^22} | {speed:8.2f} | "
              f"{elevation_deg(vel):7.1f} | {gap * 100:6.1f}{err_str}")

    if cmd is not None and pos[2] < BALL_RADIUS + 0.005:  # 落地
        print(f"----- t={step * DT:.2f}s 球落地 -----")
        break

if release_step is None:
    lost_str = f"在 t={t_lost:.2f}s 甩脱" if t_lost is not None else "全程没甩脱"
    print(f"\n[总结] q6={args.q6:.3f} | {lost_str} | 在手里时球的最大速度 {max_speed:.2f} m/s")
elif sep is None:
    print(f"\n[总结] 松手指令发出后，记录的 {total_time:.1f}s 内球一直没离手")
else:
    print(f"\n[总结] 指令 t={cmd[0] * DT:.2f}s → 离手 t={sep[0] * DT:.2f}s（延迟 {sep[0] - cmd[0]} 帧）")
    print(f"       指令时 速度 {np.linalg.norm(cmd[2]):.2f} m/s 仰角 {elevation_deg(cmd[2]):5.1f}°")
    print(f"       离手时 速度 {np.linalg.norm(sep[2]):.2f} m/s 仰角 {elevation_deg(sep[2]):5.1f}°")
    print(f"       最终位置 {np.round(pos, 3)} | 预测误差：按指令时 {err_cmd * 100:.1f} cm，按离手时 {err_sep * 100:.1f} cm")

# ---------------------------------------------------------------- 4) 收回

move_joints(READY_POSE, 2.0)
if not args.headless:  # 有窗口时停住，关窗口退出
    while simulation_app.is_running():
        simulation_app.update()

app_utils.stop()
simulation_app.close()
