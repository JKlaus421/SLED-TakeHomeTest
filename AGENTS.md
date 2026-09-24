# Project guide for AI coding assistants (Claude Code / Codex)

## Context
- SLED Lab (U-Michigan) take-home test: Franka Panda in **Isaac Sim 6.1** (Windows 11).
  Candidate tasks: (1) catch a flying ball, (2) roll a ball to knock down bottles, (3) throw a ball into a basket.
- The owner is new to Isaac Sim and is **learning while building**: explain the *why* behind changes,
  keep code readable, prefer small incremental steps over large rewrites.

## Environment
- Isaac Sim install: `D:\isaacsim` (bundled Python 3.12). Always run scripts with
  `D:\isaacsim\python.bat <script>` (or `run.bat <script>` in this repo) — never system/conda Python.
- `--headless` flag in our scripts disables the GUI (faster iteration, less RAM).
- Hardware is tight: RTX 4060 Laptop **8 GB VRAM**, **16 GB RAM**. Only one Isaac Sim process at a time;
  avoid extra cameras/sensors unless needed.
- Physics engine shown in the GUI is PhysX; verify in scripts with
  `SimulationManager.get_active_physics_engine()` before tuning physics.

## API rules (Isaac Sim 6.1)
- Use `isaacsim.core.experimental.*` (`Articulation`, `RigidPrim`, `GeomPrim`, `objects`, `utils.stage`, `utils.app`).
- `isaacsim.core.api` and `isaacsim.robot.manipulators` are **deprecated** in 6.1 — do not use them,
  even though many online tutorials still do.
- Create `SimulationApp` before importing any `omni.*` / `isaacsim.*` module.
- Experimental getters return warp arrays: convert with `.numpy()` before numpy math.
- Before guessing an API, read the version-matched examples/skills on disk (requires access to `D:\isaacsim`):
  - `D:\isaacsim\standalone_examples\tutorials\` (getting_started, manipulation: IK follow-target, gripper, pick&place)
  - `D:\isaacsim\standalone_examples\api\isaacsim.core.experimental.api\`
  - `D:\isaacsim\skills\SKILLS.md` → route to `manipulation-ik`, `motion-generation`, `physics-simulation` as needed.

## Repo layout
- `scripts/NN_name.py` — runnable experiments, numbered in the order they were built.
- Deliverables at the end: code + demo video + README with method, success rate over randomized trials, failure analysis.
