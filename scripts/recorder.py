"""recorder.py —— 把仿真画面逐帧存成 PNG（之后用 tools/make_video.py 合成 MP4 / GIF）

为什么不直接录屏：GUI 渲染比仿真慢，录屏得到的是忽快忽慢的慢动作。
这里每 N 个物理步截一张图，按固定帧率合成，视频就是"真实速度"，而且 --headless 也能录。

用法（必须在创建 SimulationApp 之后 import）：
    rec = FrameRecorder("outputs/frames", every=2)   # 每 2 个物理步（1/30 s）存一帧
    ...每次 simulation_app.update() 之后调用 rec.step()
    rec.close()
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import omni.replicator.core as rep


def write_png(path: Path, rgb: np.ndarray):
    """不依赖 PIL 的最小 PNG 编码器（Isaac Sim 自带的 Python 没有 PIL）。rgb: (H, W, 3) uint8。"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))  # 每行前加一个"不过滤"标记

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8 位、RGB
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


class FrameRecorder:
    """从视口相机截图：Replicator 的 rgb 标注器挂在相机的渲染产品上，每帧都能取到画面。"""

    def __init__(self, out_dir, every: int = 2, resolution=(1280, 720), camera="/OmniverseKit_Persp"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob("frame_*.png"):  # 清掉上一次的帧，保证编号连续
            old.unlink()
        self.every = every
        self.render_product = rep.create.render_product(camera, resolution)
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot.attach([self.render_product])
        self.count = 0  # 已保存的帧数
        self.ticks = 0  # 调用 step() 的次数（= 物理步数）
        self.paused = False

    def step(self):
        """每个物理步之后调用一次；每 every 步存一帧。"""
        if self.paused:
            return
        self.ticks += 1
        if self.ticks % self.every:
            return
        data = self.annot.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return
        write_png(self.out_dir / f"frame_{self.count:05d}.png", np.asarray(data)[:, :, :3])
        self.count += 1

    def close(self):
        self.annot.detach()
        print(f"[录制] 共 {self.count} 帧 → {self.out_dir}（{60 // self.every} fps）")
