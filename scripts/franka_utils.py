"""franka_utils.py —— 各脚本共用的 Franka 工具（场景、IK、夹爪）

注意：必须在创建 SimulationApp 之后再 import 本模块（里面会 import isaacsim.* / omni.*）。

内容：
  - 常量：物理步长、READY 姿态、两指中心偏移、夹爪朝下的朝向、手指开合值
  - 四元数小工具：quat_mul / quat_conj / rotate
  - make_scene()：地面 + 灯光 + Franka，返回 FrankaArm
  - FrankaArm：ik_step()（微分 IK 一步）、set_gripper()、grasp_pos()、reset()
  - add_marker()：纯视觉的小球标记（无物理）
"""

from __future__ import annotations

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

DT = 1 / 60  # 物理步长（练习 1 验证过）
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
ARM_DOFS = list(range(7))  # 7 个手臂关节
FINGER_DOFS = [7, 8]  # 2 根手指（平移关节，单位米）
FINGER_OPEN = 0.04  # 每根手指最多张开 4 cm → 两指间最大 8 cm
FINGER_CLOSED = 0.0
HAND_TO_GRASP = np.array([0.0, 0.0, 0.1034])  # 两指中心在 panda_hand 坐标系里的位置（NVIDIA 的 Franka 配置）
GRASP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)：绕 x 轴转 180°，夹爪朝下


# ---------------------------------------------------------------- 数学小工具


def to_np(x):
    """实验版 API 返回 warp 数组，统一转成 numpy。"""
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


def make_scene(camera_eye, camera_target) -> "FrankaArm":
    """新建场景：地面 + 灯光 + Franka（READY 姿态为默认状态）。需要在 play() 之前调用。"""
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("找不到 Isaac Sim 资源路径（需要联网访问 NVIDIA 资源服务器）")

    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(300)
    set_camera_view(eye=camera_eye, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    franka_usd = assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
    stage_utils.add_reference_to_stage(usd_path=franka_usd, path="/World/Franka")
    return FrankaArm("/World/Franka")


def add_marker(path: str, position, color=(0.1, 0.9, 0.2), radius: float = 0.02) -> UsdGeom.XformCommonAPI:
    """纯视觉小球（没有物理、不挡东西）。返回的对象可以用 .SetTranslate(Gf.Vec3d(...)) 移动它。"""
    stage = omni.usd.get_context().get_stage()
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.XformCommonAPI(sphere.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    return xform


# ---------------------------------------------------------------- 机械臂


class FrankaArm:
    """把 Franka 常用操作包一层：微分 IK、夹爪、读两指中心位置。"""

    def __init__(self, path: str, damping: float = 0.05, scale: float = 1.0):
        self.robot = Articulation(path)
        self.robot.set_world_poses(positions=[0.0, 0.0, 0.0])
        self.robot.set_default_state(dof_positions=READY_POSE)
        self.hand = RigidPrim(f"{path}/panda_hand")
        self.hand_index = self.robot.get_link_indices("panda_hand").list()[0]
        self.damping = damping
        self.scale = scale

    def reset(self):
        """关节角复位到 READY，关节目标也设成 READY（否则电机会把手臂拉回上一个目标）。"""
        self.robot.reset_to_default_state()
        self.robot.set_dof_position_targets(READY_POSE)

    def hand_pose(self):
        """手掌 panda_hand 的 (位置, 朝向四元数)。"""
        pos, quat = self.hand.get_world_poses()
        return to_np(pos)[0], to_np(quat)[0]

    def grasp_pos(self) -> np.ndarray:
        """两指中心的世界坐标。"""
        hand_pos, hand_quat = self.hand_pose()
        return hand_pos + rotate(hand_quat, HAND_TO_GRASP)

    def ik_step(self, grasp_target, grasp_quat=GRASP_DOWN_QUAT) -> float:
        """微分 IK 走一步：让两指中心朝 grasp_target 移动、夹爪朝向朝 grasp_quat 转动。

        返回当前两指中心到目标的距离（米）。每帧调用一次，然后 simulation_app.update()。
        """
        grasp_target = np.asarray(grasp_target, dtype=float)
        hand_pos, hand_quat = self.hand_pose()
        # 手掌目标 = 两指中心目标 − （目标朝向下的）偏移
        hand_target = grasp_target - rotate(grasp_quat, HAND_TO_GRASP)

        # 1) 6 维误差：位置差 + 朝向差（"还需要转多少"的四元数的向量部分）
        q_err = quat_mul(grasp_quat, quat_conj(hand_quat))
        error = np.concatenate([hand_target - hand_pos, q_err[1:] * np.sign(q_err[0])])
        # 2) 雅可比：固定底座的机器人，link 下标要减 1；只取 7 个手臂关节的列
        J = to_np(self.robot.get_jacobian_matrices())[0, self.hand_index - 1, :, :7]
        # 3) 解出关节增量，4) 目标 = 当前关节角 + 增量，交给关节电机
        dq = self.scale * dls_step(J, error, self.damping)
        q = to_np(self.robot.get_dof_positions())[0, :7]
        self.robot.set_dof_position_targets((q + dq).reshape(1, -1), dof_indices=ARM_DOFS)

        return float(np.linalg.norm(grasp_target - (hand_pos + rotate(hand_quat, HAND_TO_GRASP))))

    def set_gripper(self, finger: float):
        """设置两根手指的目标开度（每根 0 ~ 0.04 m）。"""
        self.robot.set_dof_position_targets([[finger, finger]], dof_indices=FINGER_DOFS)

    def finger_positions(self) -> np.ndarray:
        return to_np(self.robot.get_dof_positions())[0, 7:9]
