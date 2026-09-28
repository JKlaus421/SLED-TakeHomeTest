"""07_trials.py —— 投篮步骤 D：随机篮子位置，跑 N 次，统计成功率 + 失败分类

每次试验：
  1. 篮子随机放在 x ∈ [0.75, 1.1]、y ∈ [-0.25, 0.25]
  2. 抓球 → 后摆（q[0] 转到 yaw = atan2(by, bx)，整个挥臂平面对准篮子）→ 挥臂
  3. 闭环松手（和 06 一样，只是把"x"换成"沿瞄准方向的距离"）+ 刹车
  4. 判定：成功 / 没松手 / 没投准（经过篮筐口时在开口外）/ 弹出（经过时在开口内，最后却不在篮子里）
结果逐次打印，并写入 outputs/trials.csv。

运行：
  run.bat scripts\\07_trials.py --trials 10 --headless
  run.bat scripts\\07_trials.py --trials 10              （开窗口看）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--record", action="store_true", help="逐帧存图到 outputs/frames（之后用 tools/make_video.py 合成视频）")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import Cube, DomeLight
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from franka_utils import (DT, FINGER_CLOSED, FINGER_OPEN, add_marker, crossing_point, make_scene, to_np)
from pxr import Gf

BALL_RADIUS = 0.03
BALL_MASS = 0.06
PEDESTAL_HEIGHT = 0.10
BALL_START = np.array([0.5, 0.0, PEDESTAL_HEIGHT + BALL_RADIUS])

BASKET_INNER = 0.24
BASKET_HEIGHT = 0.15
WALL = 0.03
RIM_Z = BASKET_HEIGHT
BASKET_X_RANGE = (0.75, 1.1)
BASKET_Y_RANGE = (-0.25, 0.25)

Q6 = 0.785
FOLLOW_POSE = [0.0, 1.0, 0.0, -1.8, 0.0, 1.4]
SWING_JOINTS = [1, 3, 5]
SWING_TIME = 0.4
SETTLE_TIME = 2.0

OUT_CSV = Path(__file__).resolve().parent.parent / "outputs" / "trials.csv"

# ---------------------------------------------------------------- 场景

arm = make_scene(camera_eye=[0.25, 3.0, 1.3], camera_target=[0.25, 0.0, 0.6])
Cube("/World/Pedestal", sizes=1.0, scales=[[0.12, 0.12, PEDESTAL_HEIGHT]],
     positions=[[0.5, 0.0, PEDESTAL_HEIGHT / 2]], colors=[[0.5, 0.5, 0.55]])
GeomPrim("/World/Pedestal", apply_collision_apis=True)

# 篮子的 5 个部件：记下每个部件相对篮子中心的偏移，每次试验整体挪过去
_half = BASKET_INNER / 2 + WALL / 2
_outer = BASKET_INNER + 2 * WALL
BASKET_PARTS = {  # 名字: (相对篮子中心的偏移, 尺寸)
    "Bottom": ([0, 0, WALL / 2], [_outer, _outer, WALL]),
    "WallFront": ([_half, 0, BASKET_HEIGHT / 2], [WALL, _outer, BASKET_HEIGHT]),
    "WallBack": ([-_half, 0, BASKET_HEIGHT / 2], [WALL, _outer, BASKET_HEIGHT]),
    "WallLeft": ([0, _half, BASKET_HEIGHT / 2], [BASKET_INNER, WALL, BASKET_HEIGHT]),
    "WallRight": ([0, -_half, BASKET_HEIGHT / 2], [BASKET_INNER, WALL, BASKET_HEIGHT]),
}
basket_cubes = {}
for name, (offset, size) in BASKET_PARTS.items():
    path = f"/World/Basket/{name}"
    # 先放在 x=0.9 处（放在原点会和机械臂底座重叠，一开仿真就被碰撞弹开）
    basket_cubes[name] = Cube(path, sizes=1.0, scales=[size], positions=[[offset[0] + 0.9, offset[1], offset[2]]],
                              colors=[[0.2, 0.4, 0.8]])
    GeomPrim(path, apply_collision_apis=True)
aim_marker = add_marker("/World/Aim", [0, 0, RIM_Z], radius=0.015)

add_marker("/World/Ball", BALL_START, color=(1.0, 0.45, 0.0), radius=BALL_RADIUS)
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)
ball.set_masses([BALL_MASS])

app_utils.play()
simulation_app.update()

rec = None
if args.record:
    from recorder import FrameRecorder

    DomeLight("/World/DomeLight").set_intensities(600)  # headless 没有环境光，天空是黑的
    rec = FrameRecorder(OUT_CSV.parent / "frames", every=2)  # 30 fps


def tick():
    """仿真前进一步（录制时顺便截图）。"""
    simulation_app.update()
    if rec is not None:
        rec.step()


# ---------------------------------------------------------------- 工具


def move_basket(center_xy):
    """篮子（固定碰撞体）整体挪到新位置：篮子不是刚体，直接改位置即可。"""
    for name, (offset, _) in BASKET_PARTS.items():
        basket_cubes[name].set_world_poses(positions=[[center_xy[0] + offset[0], center_xy[1] + offset[1], offset[2]]])
    aim_marker.SetTranslate(Gf.Vec3d(center_xy[0], center_xy[1], RIM_Z))


def ball_state():
    pos = to_np(ball.get_world_poses()[0])[0]
    vel = to_np(ball.get_velocities()[0])[0]
    return pos, vel


def run_ik_phase(target, finger, max_time, tol):
    arm.set_gripper(finger)
    for _ in range(int(max_time / DT)):
        err = arm.ik_step(target)
        tick()
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
        tick()


def release_and_brake():
    arm.set_gripper(FINGER_OPEN)
    q_now = to_np(arm.robot.get_dof_positions())[0, :7]
    arm.robot.set_dof_position_targets(q_now.reshape(1, -1), dof_indices=list(range(7)))


def wait(seconds):
    for _ in range(int(seconds / DT)):
        tick()


# ---------------------------------------------------------------- 一次试验


def run_trial(basket_xy):
    """返回一行结果（dict）。"""
    basket_xy = np.asarray(basket_xy)
    move_basket(basket_xy)
    yaw = float(np.arctan2(basket_xy[1], basket_xy[0]))  # q[0] 转这么多，挥臂平面就对准篮子
    u = np.array([np.cos(yaw), np.sin(yaw)])  # 瞄准方向（水平单位向量）
    target_dist = float(np.linalg.norm(basket_xy))  # 沿瞄准方向要扔多远

    # 复位：机械臂回 READY，球放回台子
    arm.reset()
    arm.set_gripper(FINGER_OPEN)
    ball.set_world_poses(positions=[BALL_START.tolist()], orientations=[[1.0, 0.0, 0.0, 0.0]])
    ball.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])
    wait(0.5)

    # 抓球 → 后摆（录制时跳过抓球这段：每次都一样，又慢）
    if rec is not None:
        rec.paused = True
    grasp_center = ball_state()[0]
    run_ik_phase(grasp_center + [0, 0, 0.12], FINGER_OPEN, 3.0, 0.01)
    run_ik_phase(grasp_center, FINGER_OPEN, 2.0, 0.005)
    run_ik_phase(grasp_center, FINGER_CLOSED, 0.5, None)
    run_ik_phase(grasp_center + [0, 0, 0.30], FINGER_CLOSED, 2.0, 0.01)
    if rec is not None:
        rec.paused = False
    move_joints([yaw, -0.785, 0.0, -0.07, 0.0, 3.14, Q6, FINGER_CLOSED, FINGER_CLOSED], 2.0)
    wait(0.5)

    row = {"basket_x": basket_xy[0], "basket_y": basket_xy[1], "yaw_deg": np.degrees(yaw)}

    # 挥臂 + 闭环松手
    arm.robot.set_dof_position_targets(np.array(FOLLOW_POSE)[SWING_JOINTS].reshape(1, -1), dof_indices=SWING_JOINTS)
    released, prev_d = None, None
    for step in range(1, int(SWING_TIME / DT) + 1):
        tick()
        pos, vel = ball_state()
        hit = crossing_point(pos, vel, RIM_Z)
        if hit is None:
            continue
        d = float(hit[1][:2] @ u)  # 预测落点沿瞄准方向的距离
        if prev_d is not None and d + (d - prev_d) / 2 >= target_dist:
            release_and_brake()
            released = (step, vel, hit[1])
            break
        prev_d = d

    if released is None:
        row.update(result="没松手")
        wait(0.5)
        return row
    step, v_rel, p_pred = released
    row.update(release_t=step * DT, release_speed=np.linalg.norm(v_rel),
               release_elev_deg=np.degrees(np.arctan2(v_rel[2], np.linalg.norm(v_rel[:2]))),
               release_vy_perp=float(v_rel[:2] @ np.array([-u[1], u[0]])))  # 垂直于瞄准方向的侧向速度

    # 看球落在哪
    actual_cross, prev_pos = None, ball_state()[0]
    for _ in range(int(SETTLE_TIME / DT)):
        tick()
        pos = ball_state()[0]
        if actual_cross is None and prev_pos[2] >= RIM_Z > pos[2]:
            actual_cross = pos
        prev_pos = pos
    final = ball_state()[0]

    def along_across(p):  # 相对篮子中心：沿瞄准方向 / 垂直于瞄准方向 的偏差
        rel = p[:2] - basket_xy
        return float(rel @ u), float(rel @ np.array([-u[1], u[0]]))

    row["pred_along"], row["pred_across"] = along_across(p_pred)
    if actual_cross is not None:
        row["cross_along"], row["cross_across"] = along_across(actual_cross)
    inside_final = bool(np.all(np.abs(final[:2] - basket_xy) < BASKET_INNER / 2) and final[2] < BASKET_HEIGHT)
    # 经过篮筐口高度时，球心离开口边缘至少一个半径 → 能干净地落进去
    clear = BASKET_INNER / 2 - BALL_RADIUS
    inside_cross = actual_cross is not None and max(abs(row["cross_along"]), abs(row["cross_across"])) < clear
    if inside_final:
        row["result"] = "成功"
    elif inside_cross:
        row["result"] = "弹出"
    else:
        row["result"] = "没投准"
    return row


# ---------------------------------------------------------------- 主循环

rng = np.random.default_rng(args.seed)
rows = []
for i in range(args.trials):
    if not simulation_app.is_running():
        break
    bxy = [rng.uniform(*BASKET_X_RANGE), rng.uniform(*BASKET_Y_RANGE)]
    row = run_trial(bxy)
    row["trial"] = i + 1
    rows.append(row)
    miss = ""
    if "cross_along" in row:
        miss = f" | 经过篮筐口时偏差：沿瞄准方向 {row['cross_along'] * 100:+5.1f} cm，侧向 {row['cross_across'] * 100:+5.1f} cm"
    print(f"[试验 {i + 1:2d}] 篮子 ({bxy[0]:.3f}, {bxy[1]:+.3f}) yaw {row['yaw_deg']:+5.1f}° → {row['result']}{miss}")

# 汇总
n = len(rows)
if n:
    counts = {k: sum(r["result"] == k for r in rows) for k in ["成功", "没投准", "弹出", "没松手"]}
    print(f"\n[总结] 成功 {counts['成功']}/{n}（{counts['成功'] / n * 100:.0f}%）| "
          + " | ".join(f"{k} {v}" for k, v in counts.items() if k != "成功"))
    crossed = [r for r in rows if "cross_along" in r]
    if crossed:
        along = np.array([r["cross_along"] for r in crossed]) * 100
        across = np.array([r["cross_across"] for r in crossed]) * 100
        print(f"       偏差（cm）沿瞄准方向：平均 {along.mean():+.1f}，标准差 {along.std():.1f} | "
              f"侧向：平均 {across.mean():+.1f}，标准差 {across.std():.1f}")

    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["trial", "basket_x", "basket_y", "yaw_deg", "result", "release_t", "release_speed", "release_elev_deg",
              "release_vy_perp", "pred_along", "pred_across", "cross_along", "cross_across"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"       结果已写入 {OUT_CSV}")

if rec is not None:
    rec.close()
app_utils.stop()
simulation_app.close()
