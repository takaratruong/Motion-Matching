# MotionBricks Hill Root Conditioning Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a controllable G1 MotionBricks spike that walks kinematically over one smooth 8--10 degree hill by adding terrain-relative height to MotionBricks' future root targets before inference.

**Architecture:** A shared analytic `GentleHillProfile` owns terrain height and render mesh generation. A runtime adapter wraps the pinned MotionBricks agent's private target-transform method, reconstructs spring targets in world XY, and adds sampled relative elevation only to target root Y in MotionBricks coordinates. A dedicated MuJoCo viewer reuses the official WASD controller and keeps output qpos raw except for the existing current-support world offset.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, MuJoCo, official pinned MotionBricks, pytest.

---

### Task 1: Analytic hill contract

**Files:**
- Create: `sonic/python/mm_sonic/motionbricks_hill.py`
- Test: `sonic/tests/test_motionbricks_hill.py`

**Step 1: Write the failing tests**

Cover the flat approach/exit, cosine crest, domain rejection, maximum slope angle, and exact agreement between sampled profile vertices and `mesh()` output.

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m pytest -q sonic/tests/test_motionbricks_hill.py
```

Expected: FAIL because `mm_sonic.motionbricks_hill` does not exist.

**Step 3: Implement the minimal profile**

Implement an immutable profile with:

```python
@dataclass(frozen=True)
class GentleHillProfile:
    domain_x: tuple[float, float] = (-3.0, 12.0)
    half_width: float = 3.0
    hill_start_x: float = 1.5
    hill_length: float = 7.0
    height_m: float = 0.35

    def height(self, xy: object) -> float: ...
    def profile_vertices(self, sample_count: int = 151) -> np.ndarray: ...
    def mesh(self, sample_count: int = 151) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def max_slope_degrees(self) -> float: ...
```

Use `0.5 * height_m * (1 - cos(2*pi*s))` within the hill interval and zero on both aprons.

**Step 4: Run the focused tests**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill.py sonic/tests/test_motionbricks_hill.py
git commit -m "feat: add deterministic gentle hill profile"
```

### Task 2: Pre-inference MotionBricks conditioner

**Files:**
- Create: `sonic/python/mm_sonic/motionbricks_hill_conditioning.py`
- Test: `sonic/tests/test_motionbricks_hill_conditioning.py`

**Step 1: Write the failing tests**

Test canonical target XY reconstruction at zero and 90-degree headings, flat/uphill/crest/downhill height deltas, preservation of clip vertical oscillation, hook installation/removal, and hard failures for unexpected batch or target shapes.

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m pytest -q sonic/tests/test_motionbricks_hill_conditioning.py
```

Expected: FAIL because the conditioning module does not exist.

**Step 3: Implement pure coordinate and delta helpers**

Implement:

```python
def canonical_targets_to_world_xy(
    target_root_positions: np.ndarray,
    heading_radians: float,
    first_frame_world_xy: object,
) -> np.ndarray: ...

def terrain_height_deltas(
    current_world_xy: object,
    target_world_xy: object,
    height_query: Callable[[object], float],
) -> np.ndarray: ...
```

Map MotionBricks planar `[internal_x, internal_z]` to MuJoCo `[world_y, world_x]`, then rotate by the canonicalization heading and translate by the first context position.

**Step 4: Implement the guarded runtime hook**

`MotionBricksHillConditioner.install(agent)` wraps `_generate_target_joint_transforms`. After the original method returns, it validates batch size one and required canonicalization fields, computes deltas, and adds them to `target_global_root_positions[:, :, 1]` on the existing tensor device and dtype. `remove()` restores the exact original callable. Store the latest immutable diagnostic trace for the viewer.

**Step 5: Run the focused tests**

Run the command from Step 2. Expected: PASS.

**Step 6: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_conditioning.py sonic/tests/test_motionbricks_hill_conditioning.py
git commit -m "feat: condition MotionBricks root targets on terrain"
```

### Task 3: Controllable MuJoCo hill viewer

**Files:**
- Create: `sonic/python/mm_sonic/motionbricks_hill_viewer.py`
- Modify: `sonic/README.md`

**Step 1: Add a bounded CLI contract test**

Extend `sonic/tests/test_motionbricks_hill.py` to call `motionbricks_hill_viewer.main(["--help"])` and assert the documented runtime options without importing MotionBricks at module import time.

**Step 2: Verify the test fails**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m pytest -q sonic/tests/test_motionbricks_hill.py
```

Expected: FAIL because the viewer module does not exist.

**Step 3: Implement the dedicated launcher**

Load `navigation_demo` lazily, replace its flat scene model with an `MjSpec` containing the profile mesh, install the conditioner, and run the official `WASD_controller` loop. Before each generation subtract the current sampled support height from context root Z; after a genuinely new generated batch add that one support height back once. Do not change generated orientation or joint values.

Print controls and a compact live line containing root Z, terrain Z, and the four latest conditioned target heights. Stop safely if the root leaves the certified hill domain.

**Step 4: Document exact setup and launch**

Add the external MotionBricks checkout, dependency install, Git-LFS, and viewer command to `sonic/README.md`. Mark this as an experimental kinematic probe without IK or collision correction.

**Step 5: Run CPU verification**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m pytest -q sonic/tests/test_motionbricks_hill.py sonic/tests/test_motionbricks_hill_conditioning.py
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m compileall -q sonic/python/mm_sonic
```

Expected: all focused tests pass and compilation exits zero.

**Step 6: Commit**

```bash
git add sonic/python/mm_sonic/motionbricks_hill_viewer.py sonic/README.md sonic/tests/test_motionbricks_hill.py
git commit -m "feat: add controllable MotionBricks hill viewer"
```

### Task 4: GPU runtime canary

**Files:**
- Modify only if the smoke test exposes a scoped integration defect.

**Step 1: Prepare isolated runtime dependencies**

Install only the missing MotionBricks runtime packages into a worktree-local environment or the selected disposable environment. Fetch the pinned released Git-LFS weights and verify they are real checkpoint files.

**Step 2: Load the released agent**

Run the viewer with `--smoke-steps 1 --no-viewer` so it loads checkpoints, installs the hook, performs a bounded generation, and exits.

**Step 3: Inspect the conditioning evidence**

Require finite generated qpos, four finite conditioned targets, nonzero target elevation when the bounded smoke command reaches the hill, and no post-generation terrain projection.

**Step 4: Re-run focused tests after any integration fix**

Use the Task 3 CPU verification commands, then record the exact launch command and any remaining environment limitation in the handoff.
