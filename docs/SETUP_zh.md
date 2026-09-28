# 环境设置笔记（Isaac Sim 6.1 + Franka Panda）

> 项目说明见根目录 README.md（英文）；下面的"目录"是项目刚建立时的样子，完整文件列表见 README 第 6 节。

Isaac Sim 装在 `D:\isaacsim`，本项目的代码都放在这里，不要写进 Isaac Sim 目录。

## 目录
```
sled-takehome/
├─ sled-takehome.code-workspace   VS Code 工作区（双击打开）：解释器、代码补全、F5 运行/调试
├─ scripts/
│  └─ 00_hello_franka.py   冒烟测试：Panda + 抛球
├─ tools/
│  └─ first_launch.bat     第一次启动 Isaac Sim（双击运行）
├─ run.bat             用 Isaac Sim 自带的 Python 运行脚本
└─ README.md
```

## 怎么运行脚本
1. **VS Code**：双击 `sled-takehome.code-workspace` 打开工作区 → 打开某个 `.py` 文件 → 按 `F5` → 选 "Isaac Sim: 运行/调试当前文件"（可以打断点）
2. **终端**：`run.bat scripts\00_hello_franka.py`，加 `--headless` 就不开窗口

> 不要用系统 Python 或 conda 里的 Python 运行，会提示找不到 `isaacsim` 模块。

## 6.1 版本 API 注意事项
- 新代码请用 `isaacsim.core.experimental.*`（例如 `Articulation`、`RigidPrim`、`GeomPrim`）。
  旧的 `isaacsim.core.api`、`isaacsim.robot.manipulators` 在 6.1 中已被标为 deprecated。
  网上教程和 AI 生成的代码很多还是旧写法，遇到时要留意。
- 本机自带的参考示例（版本和你装的一致）：
  - `D:\isaacsim\standalone_examples\tutorials\getting_started\`
  - `D:\isaacsim\standalone_examples\tutorials\manipulation\`（IK 跟随目标、夹爪、pick&place）
  - `D:\isaacsim\standalone_examples\api\isaacsim.core.experimental.api\`
- `D:\isaacsim\skills\` 下有 NVIDIA 官方写给 AI 编程助手的说明（manipulation-ik、physics-simulation 等），
  人读也很有用。
- 6.x 的 full app 默认物理引擎可能是 **Newton**，不是 PhysX。调物理参数前先打印一下
  `SimulationManager.get_active_physics_engine()`。
