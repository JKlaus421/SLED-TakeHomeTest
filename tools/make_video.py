"""make_video.py —— 把 outputs/frames/frame_*.png 合成 MP4（和 GIF）

注意：用 Anaconda 的 Python 运行（它有 PIL / imageio），不是 Isaac Sim 的 python.bat：
  D:\\anaconda\\python.exe tools\\make_video.py
  D:\\anaconda\\python.exe tools\\make_video.py --fps 30 --slowmo 0.25 --name throw_slowmo

MP4 需要 imageio-ffmpeg（D:\\anaconda\\python.exe -m pip install imageio-ffmpeg）；
没装的话只生成 GIF。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--frames", default=str(ROOT / "outputs" / "frames"))
parser.add_argument("--fps", type=float, default=30, help="录制帧率（recorder every=2 → 30）")
parser.add_argument("--slowmo", type=float, default=1.0, help="播放速度倍率，0.25 = 四分之一速慢动作")
parser.add_argument("--name", default="throw")
parser.add_argument("--gif-width", type=int, default=640, help="GIF 缩小到这个宽度（控制文件大小）")
parser.add_argument("--no-gif", action="store_true")
args = parser.parse_args()

files = sorted(Path(args.frames).glob("frame_*.png"))
if not files:
    raise SystemExit(f"{args.frames} 里没有 frame_*.png，先用 --record 跑一次仿真")
play_fps = args.fps * args.slowmo
out_dir = ROOT / "outputs"
print(f"{len(files)} 帧，播放 {play_fps:g} fps（时长 {len(files) / play_fps:.1f} s）")

# MP4
try:
    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa: F401  （检查是否装了 ffmpeg）

    mp4 = out_dir / f"{args.name}.mp4"
    with imageio.get_writer(mp4, fps=play_fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for f in files:
            w.append_data(np.asarray(Image.open(f).convert("RGB")))
    print(f"MP4 → {mp4}")
except ImportError:
    print("没有 imageio-ffmpeg，跳过 MP4（安装：D:\\anaconda\\python.exe -m pip install imageio-ffmpeg）")

# GIF（GitHub README 里能直接播放）
if not args.no_gif:
    gif = out_dir / f"{args.name}.gif"
    frames = []
    for f in files[::2] if play_fps > 20 else files:  # 帧率高时隔帧取，GIF 小一些
        im = Image.open(f).convert("RGB")
        h = round(im.height * args.gif_width / im.width)
        frames.append(im.resize((args.gif_width, h), Image.LANCZOS))
    step = 2 if play_fps > 20 else 1
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 * step / play_fps), loop=0,
                   optimize=True)
    print(f"GIF → {gif}（{gif.stat().st_size / 1e6:.1f} MB）")
