"""02_ik_reach.py —— 练习 2：微分 IK，让夹爪中心到达指定点，并测量要多久

做的事情：
  1. 场景：地面 + 灯光 + Franka + 绿色目标点（纯视觉，无物理）
  2. 每次试验：机械臂回到 READY 姿态 → 用微分 IK 让"两指中心"去够目标点（夹爪朝下）
  3. 记录：误差首次 < 1 cm 用了多久、最终误差是多少
  第 1 个目标是练习 1 的绿点，之后是工作空间里的随机点。

和官方示例 control_robot_numpy.py 的两个区别：
  - 官方示例控制的是手掌 panda_hand 的位置；我们控制的是两指中心（抓球的地方），
    它在 panda_hand 沿自身 z 轴往前 0.1034 m 处（数据来自 NVIDIA 的 Franka 配置）。
  - 我们同时指定朝向（夹爪竖直朝下），这样手掌位置和两指中心的换算很简单。

运行：
  run.bat scripts\\02_ik_reach.py
  run.bat scripts\\02_ik_reach.py --scale 0.3          （每步只走 30%，看看会慢多少）
  run.bat scripts\\02_ik_reach.py --trials 10 --headless
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--target", type=float, nargs=3, default=[0.45, 0.0, 0.55], help="第 1 个目标点 x y z（米）")
parser.add_argument("--trials", type=int, default=5, help="试验次数（第 1 次用 --target，其余随机）")
parser.add_argument("--scale", type=float, default=1.0, help="每步走误差的多少比例（0~1）")
parser.add_argument("--damping", type=float, default=0.05, help="阻尼最小二乘的 λ")
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

DT = 1 / 60  # 物理步长（练习 1 验证过）
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
HAND_TO_GRASP = np.array([0.0, 0.0, 0.1034])  # 两指中心在 panda_hand 坐标系里的位置
GRASP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)：绕 x 轴转 180°，夹爪朝下
MAX_TIME = 4.0  # 每次试验最多跑几秒
REACHED = 0.01  # 误差小于 1 cm 算到达


# ---------------------------------------------------------------- 数学小工具


def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """四元数乘法（w, x, y, z 顺序）：先转 b，再转 a。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """共轭 = 反向旋转（单位四元数）。"""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q 旋转向量 v：q · (0, v) · q*。"""
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]


def dls_step(J: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """阻尼最小二乘：Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ · error。

    J 是 6×7（3 行位置 + 3 行转动），error 是 6 维（位置误差 + 朝向误差）。
    λ 很小时约等于"J 的伪逆 × 误差"；靠近奇异姿态时 λ 防止关节角暴走。
    """
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), error)


# ---------------------------------------------------------------- 场景

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("找不到 Isaac Sim 资源路径")
    simulation_app.close()
    sys.exit(1)

stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/DistantLight").set_intensities(300)
set_camera_view(eye=[1.6, 1.4, 1.1], target=[0.4, 0.0, 0.4], camera_prim_path="/OmniverseKit_Persp")

franka_usd = assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
stage_utils.add_reference_to_stage(usd_path=franka_usd, path="/World/Franka")
franka = Articulation("/World/Franka")
franka.set_world_poses(positions=[0.0, 0.0, 0.0])
franka.set_default_state(dof_positions=READY_POSE)

hand = RigidPrim("/World/Franka/panda_hand")
hand_index = franka.get_link_indices("panda_hand").list()[0]

stage = omni.usd.get_context().get_stage()
marker = UsdGeom.Sphere.Define(stage, "/World/TargetMarker")
marker.CreateRadiusAttr(0.02)
marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
marker_xform = UsdGeom.XformCommonAPI(marker.GetPrim())

# 目标点：第 1 个是练习 1 的绿点，其余在机械臂前方一个盒子里随机取
rng = np.random.default_rng(args.seed)
targets = [np.array(args.target)] + [
    rng.uniform([0.3, -0.3, 0.25], [0.65, 0.3, 0.7]) for _ in range(args.trials - 1)
]

app_utils.play()
simulation_app.update()


def grasp_pose():
    """返回 (两指中心位置, 手掌位置, 手掌朝向)。"""
    pos, quat = hand.get_world_poses()
    hand_pos, hand_quat = to_np(pos)[0], to_np(quat)[0]
    return hand_pos + rotate(hand_quat, HAND_TO_GRASP), hand_pos, hand_quat


# ---------------------------------------------------------------- 试验

results = []
for i, target in enumerate(targets):
    if not simulation_app.is_running():
        break
    marker_xform.SetTranslate(Gf.Vec3d(*target))

    # 回到 READY：关节角直接复位，同时把关节目标也设成 READY，否则电机会把手臂拉回上一个目标
    franka.reset_to_default_state()
    franka.set_dof_position_targets(READY_POSE)
    simulation_app.update()

    # 夹爪朝下时，手掌目标 = 两指中心目标 + 往上 0.1034 m
    hand_target = target - rotate(GRASP_DOWN_QUAT, HAND_TO_GRASP)

    t_reach = None
    for step in range(int(MAX_TIME / DT)):
        grasp_pos, hand_pos, hand_quat = grasp_pose()
        err = np.linalg.norm(target - grasp_pos)
        if t_reach is None and err < REACHED:
            t_reach = step * DT

        # 1) 6 维误差：位置差 + 朝向差（朝向差取"还需要转多少"的四元数的向量部分）
        q_err = quat_mul(GRASP_DOWN_QUAT, quat_conj(hand_quat))
        error = np.concatenate([hand_target - hand_pos, q_err[1:] * np.sign(q_err[0])])

        # 2) 雅可比：固定底座的机器人，link 下标要减 1；只取 7 个手臂关节的列
        J = to_np(franka.get_jacobian_matrices())[0, hand_index - 1, :, :7]

        # 3) 解出关节增量，4) 目标 = 当前关节角 + 增量，交给关节电机
        dq = args.scale * dls_step(J, error, args.damping)
        q = to_np(franka.get_dof_positions())[0, :7]
        franka.set_dof_position_targets((q + dq).reshape(1, -1), dof_indices=list(range(7)))

        simulation_app.update()

    reach_str = f"{t_reach:.2f}s" if t_reach is not None else "未到达"
    print(f"[试验 {i + 1}] 目标 {np.round(target, 3)} | 到达 1cm 用时 {reach_str} | {MAX_TIME:.0f}s 后误差 {err * 100:.2f} cm")
    results.append((t_reach, err))

reached = [t for t, _ in results if t is not None]
if results:
    print(f"\n[总结] scale={args.scale} damping={args.damping} | 到达 {len(reached)}/{len(results)}", end="")
    print(f" | 平均用时 {np.mean(reached):.2f}s | 最长 {np.max(reached):.2f}s" if reached else "")

app_utils.stop()
simulation_app.close()
