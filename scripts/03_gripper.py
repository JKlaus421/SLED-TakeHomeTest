"""03_gripper.py —— 练习 3：夹爪抓球并抬起来

流程（一个简单的"状态机"，每个阶段做一件事，完成或超时就进入下一阶段）：
  approach  夹爪张开，移到球正上方 12 cm
  descend   垂直下降，让两指中心对准球心
  close     手臂不动，合拢手指，等 0.5 s 让手指夹紧
  lift      夹着球抬高 30 cm
  hold      保持 1 s，然后检查球还在不在手里

成功的判定看"球"的位置，而不是看夹爪：夹爪到位了，球也可能滑掉。
球放在一个小台子上（离地 10 cm），免得手指下降时碰到地面。

运行：
  run.bat scripts\\03_gripper.py
  run.bat scripts\\03_gripper.py --trials 10 --headless
"""

from __future__ import annotations

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from franka_utils import DT, FINGER_CLOSED, FINGER_OPEN, add_marker, make_scene, to_np

BALL_RADIUS = 0.03  # 直径 6 cm：Franka 两指最大张开 8 cm，留 1 cm 余量
BALL_MASS = 0.06  # kg，和网球差不多
PEDESTAL_HEIGHT = 0.10

# ---------------------------------------------------------------- 场景

arm = make_scene(camera_eye=[1.3, 1.0, 0.8], camera_target=[0.5, 0.0, 0.2])

# 台子：只有碰撞、没有刚体 → 固定不动的障碍物
Cube("/World/Pedestal", sizes=1.0, scales=[[0.12, 0.12, PEDESTAL_HEIGHT]],
     positions=[[0.5, 0.0, PEDESTAL_HEIGHT / 2]], colors=[[0.5, 0.5, 0.55]])
pedestal = GeomPrim("/World/Pedestal", apply_collision_apis=True)

# 球：有刚体（受重力）+ 碰撞
ball_marker = add_marker("/World/Ball", [0.5, 0.0, PEDESTAL_HEIGHT + BALL_RADIUS], color=(1.0, 0.45, 0.0),
                         radius=BALL_RADIUS)
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)
ball.set_masses([BALL_MASS])

app_utils.play()
simulation_app.update()


def ball_pos() -> np.ndarray:
    return to_np(ball.get_world_poses()[0])[0]


# ---------------------------------------------------------------- 状态机

# (阶段名, 两指中心相对球心的偏移, 手指开度, 最长时间 s, 误差小于多少就提前结束)
PHASES = [
    ("approach", [0.0, 0.0, 0.12], FINGER_OPEN, 3.0, 0.01),
    ("descend", [0.0, 0.0, 0.0], FINGER_OPEN, 2.0, 0.005),
    ("close", [0.0, 0.0, 0.0], FINGER_CLOSED, 0.5, None),  # None：不提前结束，固定等满时间
    ("lift", [0.0, 0.0, 0.30], FINGER_CLOSED, 2.0, 0.01),
    ("hold", [0.0, 0.0, 0.30], FINGER_CLOSED, 1.0, None),
]

rng = np.random.default_rng(args.seed)
successes = 0
for trial in range(args.trials):
    if not simulation_app.is_running():
        break

    # 球的位置在台面上随机偏一点点
    start = np.array([0.5, 0.0, PEDESTAL_HEIGHT + BALL_RADIUS]) + np.append(rng.uniform(-0.02, 0.02, 2), 0.0)
    ball.set_world_poses(positions=[start.tolist()], orientations=[[1.0, 0.0, 0.0, 0.0]])
    ball.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])
    arm.reset()
    arm.set_gripper(FINGER_OPEN)
    for _ in range(30):  # 等 0.5 s，让球在台子上放稳
        simulation_app.update()

    # 目标点在"开始抓时"算一次，之后不跟着球动（不然球被碰歪后，手会一直追着它）
    grasp_center = ball_pos()
    print(f"\n[试验 {trial + 1}] 球在 {np.round(grasp_center, 3)}")
    for name, offset, finger, max_time, tol in PHASES:
        target = grasp_center + np.array(offset)
        arm.set_gripper(finger)
        for step in range(int(max_time / DT)):
            err = arm.ik_step(target)
            simulation_app.update()
            if tol is not None and err < tol:
                break
        fingers = arm.finger_positions()
        print(f"  {name:8s} 用时 {(step + 1) * DT:4.2f}s | 夹爪误差 {err * 100:5.2f} cm | "
              f"手指 {np.round(fingers, 3)} | 球 {np.round(ball_pos(), 3)}")

    # 成功判定：球被抬高了 ≥ 20 cm，并且球心离两指中心 < 2 cm
    lifted = ball_pos()[2] - grasp_center[2]
    in_hand = np.linalg.norm(ball_pos() - arm.grasp_pos())
    ok = lifted > 0.20 and in_hand < 0.02
    successes += ok
    print(f"  ==> {'成功' if ok else '失败'}：球抬高 {lifted * 100:.1f} cm，离两指中心 {in_hand * 100:.1f} cm")

print(f"\n[总结] 成功 {successes}/{args.trials}")
app_utils.stop()
simulation_app.close()
