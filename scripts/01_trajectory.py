"""01_trajectory.py —— 练习 1：抛体轨迹预测

目标：
  给定"起点 p0、目标点 p_target、飞行时间 T"，算出抛球的初速度 v0，
  让球在 T 秒后正好经过目标点（绿色小球标记，位于夹爪正前方）。
  然后对比"你预测的轨迹"和"仿真里真实的轨迹"，看误差有多大。

你要完成的只有下面标了 TODO 的两个函数（加起来 4~6 行）。其余的场景搭建、
计时、打印、统计都已经写好了。

运行：
  VS Code 按 F5，或
  run.bat scripts\\01_trajectory.py
  run.bat scripts\\01_trajectory.py --T 0.8 --target 0.45 0.2 0.6    （换飞行时间/目标点）
  run.bat scripts\\01_trajectory.py --headless                        （不开窗口，只看数字）
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--T", type=float, default=0.6, help="期望飞行时间（秒）")
parser.add_argument("--target", type=float, nargs=3, default=[0.45, 0.0, 0.55], help="目标点 x y z（米）")
parser.add_argument("--throws", type=int, default=5, help="headless 模式下抛几次后退出")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

G = 9.81  # 重力加速度，沿 -z 方向
DT = 1 / 60  # 物理步长（秒）：仿真每一步前进的时间，由练习 1 的误差数据反推得出

# ======================================================================
#  TODO：你来实现这两个函数
# ======================================================================


def launch_velocity(p0: np.ndarray, p_target: np.ndarray, T: float, g: float = G, dt: float = DT) -> np.ndarray:
    """给定起点 p0、目标点 p_target、飞行时间 T，返回初速度 v0 = [vx, vy, vz]。

    提示：
      抛体运动（忽略空气阻力）：
        x(t) = x0 + vx * t
        y(t) = y0 + vy * t
        z(t) = z0 + vz * t - 0.5 * g * t^2
      令 t = T 时 (x, y, z) = p_target，分别解出 vx, vy, vz。
      可以用 numpy 向量一次算完三个分量。
    """
    v0 = (p_target - p0) / T
    # 重力补偿 ½gT，再加离散积分多掉的 ½g·dt（注意这一项不乘时间）
    v0[2] += 0.5 * g * T + 0.5 * g * dt
    return v0


def predict_position(p0: np.ndarray, v0: np.ndarray, t: float, g: float = G, dt: float = DT) -> np.ndarray:
    """给定起点、初速度，返回 t 秒后球的理论位置 [x, y, z]。"""
    # 三个方向先都按匀速直线运动算：p = p0 + v0·t（numpy 会逐元素计算 x、y、z）
    p = p0 + v0 * t
    # 只有 z 方向受重力：连续公式往下掉 ½·g·t²，离散积分每秒再多掉 ½·g·dt
    p[2] -= 0.5 * g * t**2 + 0.5 * g * dt * t
    return p


# ======================================================================
#  以下不用改（看懂即可）
# ======================================================================


def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("找不到 Isaac Sim 资源路径")
    simulation_app.close()
    sys.exit(1)

stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/DistantLight").set_intensities(300)
set_camera_view(eye=[1.2, 2.4, 1.0], target=[0.9, 0.0, 0.6], camera_prim_path="/OmniverseKit_Persp")

franka_usd = assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
stage_utils.add_reference_to_stage(usd_path=franka_usd, path="/World/Franka")
franka = Articulation("/World/Franka")
franka.set_world_poses(positions=[0.0, 0.0, 0.0])

stage = omni.usd.get_context().get_stage()

# 被抛的球（有物理）
BALL_RADIUS = 0.035
P0 = np.array([1.6, 0.0, 0.9])
ball_geom = UsdGeom.Sphere.Define(stage, "/World/Ball")
ball_geom.CreateRadiusAttr(BALL_RADIUS)
ball_geom.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.45, 0.0)])
UsdGeom.XformCommonAPI(ball_geom.GetPrim()).SetTranslate(Gf.Vec3d(*P0))
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)

# 目标点标记（纯视觉，没有物理，不会挡住球）
P_TARGET = np.array(args.target)
marker = UsdGeom.Sphere.Define(stage, "/World/TargetMarker")
marker.CreateRadiusAttr(0.02)
marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
UsdGeom.XformCommonAPI(marker.GetPrim()).SetTranslate(Gf.Vec3d(*P_TARGET))

READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

timeline = omni.timeline.get_timeline_interface()
app_utils.play()
simulation_app.update()
franka.set_dof_position_targets(READY_POSE)

V0 = launch_velocity(P0, P_TARGET, args.T)
print(f"\n[练习1] 起点 {P0}，目标 {P_TARGET}，飞行时间 T={args.T}s")
print(f"[练习1] 你算出的初速度 v0 = {np.round(V0, 3)}  (|v0| = {np.linalg.norm(V0):.2f} m/s)\n")


def throw():
    ball.set_world_poses(positions=[P0.tolist()], orientations=[[1.0, 0.0, 0.0, 0.0]])
    ball.set_velocities(linear_velocities=[V0.tolist()], angular_velocities=[[0.0, 0.0, 0.0]])


WAIT_BEFORE_THROW = 1.0  # 秒：让机械臂先摆好姿态
t_throw = None
n_throws = 0
last_print = -1.0
closest = (np.inf, None)  # (到目标点的最小距离, 发生时刻)

while simulation_app.is_running():
    simulation_app.update()
    now = timeline.get_current_time()

    if t_throw is None:
        if now >= WAIT_BEFORE_THROW:
            throw()
            t_throw, last_print, closest = now, -1.0, (np.inf, None)
            n_throws += 1
            print(f"---------- 第 {n_throws} 次抛球 ----------")
            print(f"{'t(s)':>6} | {'仿真位置 (m)':^26} | {'预测位置 (m)':^26} | 误差(cm)")
        continue

    t = now - t_throw
    pos = to_np(ball.get_world_poses()[0]).reshape(-1)
    pred = predict_position(P0, V0, t)
    err_cm = np.linalg.norm(pos - pred) * 100
    d_target = np.linalg.norm(pos - P_TARGET)
    if d_target < closest[0]:
        closest = (d_target, t)

    if t - last_print >= 0.05:  # 每 0.05 秒打印一行
        last_print = t
        print(f"{t:6.2f} | {np.array2string(pos, precision=3):^26} | {np.array2string(pred, precision=3):^26} | {err_cm:6.2f}")

    if t > args.T + 0.4:  # 这次抛球结束，打印总结，准备下一次
        d, tc = closest
        print(f"==> 离目标点最近 {d*100:.1f} cm，发生在 t={tc:.2f}s（期望 T={args.T}s）\n")
        t_throw = None
        timeline.set_current_time(0.0) if False else None  # 保留仿真时间连续，不重置
        WAIT_BEFORE_THROW = now + 1.0
        if args.headless and n_throws >= args.throws:
            break

app_utils.stop()
simulation_app.close()
