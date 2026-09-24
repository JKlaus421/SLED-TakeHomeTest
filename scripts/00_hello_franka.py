"""00_hello_franka.py —— 第一个冒烟测试（Isaac Sim 6.1）

做的事情：
  1. 建一个场景：地面 + 灯光 + Franka Panda + 一个小球
  2. 让 Panda 移到一个常用的 "ready" 姿态
  3. 每隔几秒把小球从空中朝机械臂方向抛出去（为"接球"任务预热），打印球和关节状态

运行方式（任选其一）：
  - VS Code 里按 F5，选 "Isaac Sim: 运行/调试当前文件"
  - 终端：D:\\isaacsim\\python.bat scripts\\00_hello_franka.py
  - 无窗口：D:\\isaacsim\\python.bat scripts\\00_hello_franka.py --headless

API 参考（本机自带、与 6.1 版本一致）：
  D:\\isaacsim\\standalone_examples\\tutorials\\getting_started\\getting_started_robot.py
  D:\\isaacsim\\standalone_examples\\api\\isaacsim.core.experimental.api\\control_frankas.py
  D:\\isaacsim\\standalone_examples\\api\\isaacsim.core.experimental.api\\add_cubes.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="不开窗口运行")
parser.add_argument("--steps", type=int, default=1200, help="headless 模式下运行的步数")
args, _ = parser.parse_known_args()

# SimulationApp 必须在任何 omni / isaacsim 模块 import 之前创建
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom


def to_np(x):
    """实验版 API 常返回 warp 数组，统一转成 numpy 方便打印/计算。"""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ---------------------------------------------------------------- 场景
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("找不到 Isaac Sim 资源路径（需要联网访问 NVIDIA 资源服务器）")
    simulation_app.close()
    sys.exit(1)

stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/DistantLight").set_intensities(300)
set_camera_view(eye=[2.2, 1.6, 1.4], target=[0.3, 0.0, 0.4], camera_prim_path="/OmniverseKit_Persp")

# Franka Panda（官方资源）
franka_usd = assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
stage_utils.add_reference_to_stage(usd_path=franka_usd, path="/World/Franka")
franka = Articulation("/World/Franka")
franka.set_world_poses(positions=[0.0, 0.0, 0.0])

# 小球：用 USD 原生接口建一个球，再加刚体 + 碰撞
BALL_RADIUS = 0.035  # 米，大约网球大小
BALL_START = [1.6, 0.0, 0.9]
stage = omni.usd.get_context().get_stage()
sphere = UsdGeom.Sphere.Define(stage, "/World/Ball")
sphere.CreateRadiusAttr(BALL_RADIUS)
sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.45, 0.0)])
UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(Gf.Vec3d(*BALL_START))
ball = RigidPrim("/World/Ball")
GeomPrim("/World/Ball", apply_collision_apis=True)

# ---------------------------------------------------------------- 仿真
# 7 个手臂关节 + 2 个手指（单位：弧度 / 米）
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

app_utils.play()
simulation_app.update()
franka.set_dof_position_targets(READY_POSE)


def throw_ball():
    """把球放回起点，并给一个朝机械臂方向的初速度（抛物线）。"""
    ball.set_world_poses(positions=[BALL_START], orientations=[[1.0, 0.0, 0.0, 0.0]])
    ball.set_velocities(linear_velocities=[[-2.6, 0.0, 1.6]], angular_velocities=[[0.0, 0.0, 0.0]])


step = 0
print("[hello_franka] 开始运行。GUI 模式下关闭窗口即可退出。")
while simulation_app.is_running():
    simulation_app.update()
    step += 1

    if step % 240 == 60:  # 大约每 4 秒抛一次
        throw_ball()
        print(f"\n[step {step}] 抛球！")

    if step % 60 == 0:
        pos, _ = ball.get_world_poses()
        vel, _ = ball.get_velocities()
        q = to_np(franka.get_dof_positions()).reshape(-1)
        print(
            f"[step {step}] ball pos={np.round(to_np(pos).reshape(-1), 3)} "
            f"vel={np.round(to_np(vel).reshape(-1), 2)} | arm q={np.round(q[:7], 2)}"
        )

    if args.headless and step >= args.steps:
        break

app_utils.stop()
simulation_app.close()
