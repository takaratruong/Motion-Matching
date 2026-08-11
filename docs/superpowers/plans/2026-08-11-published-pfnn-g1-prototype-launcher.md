# Published PFNN to G1 Prototype Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the working published terrain-PFNN → pinned GMR → G1 MuJoCo prototype behind one synchronized six-scene launcher.

**Architecture:** Repository-owned bridge, viewer, and terrain modules preserve the validated `/tmp` data path byte-for-byte where possible. A scene manifest binds each PFNN world to its exact heightmap, while a small launcher prepares the patched exporter, builds a cached display mesh, starts all three processes, and records authenticated PIDs. The viewer rejects a world ID that does not match its selected mesh.

**Tech Stack:** Python 3.10+, NumPy, MuJoCo passive viewer, pinned General Motion Retargeting, Daniel Holden's released C++ PFNN demo, `subprocess`, JSON receipts, `unittest`.

## Global Constraints

- Scene 6 is the default; scenes 1–6 use the exact mapping in the approved design.
- Do not change PFNN inference, released IK, GMR retargeting, G1 qpos, root motion, contacts, or controls.
- Do not add smoothing, foot locking, root correction, a daemon, a web UI, or training changes.
- Preserve the current manually launched prototype until all non-live tests and dry-run checks pass.
- Preserve unstaged `.superpowers/sdd/task-1-report.md` and `.superpowers/sdd/task-2-report.md` changes.
- Use `/home/ubuntu/datasets/pfnn/pfnn/demo` as the default released PFNN source and verify upstream `pfnn.cpp` SHA-256 `6deb74a9af58874e2b9c88cca760f9aa1c9e6d4db10fa10ae02b39609e3d3154` before patching.
- Keep the validated GMR commit `bb1bbe40774794fceb2a7c579a3464a28e68c844` and morphology terrain scale `0.875`.

---

## File Structure

- `sonic/python/mm_sonic/published_pfnn_g1_bridge.py`: parse `PFNNXFM/v1`, convert released Y-up centimetres to GMR Z-up metres, and emit unchanged pinned-GMR qpos JSONL.
- `sonic/python/mm_sonic/published_pfnn_g1_viewer.py`: consume GMR qpos JSONL, enforce expected world ID, keep newest pre-retargeted display pose, and render the exact qpos/root on a terrain mesh.
- `sonic/python/mm_sonic/published_pfnn_heightmap.py`: reproduce the released heightmap loader and generate the morphology-scaled display mesh.
- `sonic/python/mm_sonic/published_pfnn_g1_scenes.json`: bind scene numbers, world IDs, keys, heightmaps, and display stride.
- `sonic/python/mm_sonic/published_pfnn_g1_scenes.py`: load and validate the manifest.
- `sonic/python/mm_sonic/published_pfnn_g1_prepare.py`: authenticate, patch, and compile the released exporter and cache the selected terrain mesh.
- `sonic/python/mm_sonic/published_pfnn_g1_launcher.py`: implement `prepare`, `start`, `switch`, `status`, and `stop`.
- `sonic/resources/published_pfnn_g1/pfnn_export.patch`: minimal patch against the released `pfnn.cpp`, including deterministic `--world 0..5` startup.
- `sonic/resources/published_pfnn_g1/EXPORT_SCHEMA.md`: checked-in `PFNNXFM/v1` wire contract.
- `sonic/resources/published_pfnn_g1/README.md`: one-command operator runbook.
- `tests/python/test_published_pfnn_g1_bridge.py`: bridge contract tests.
- `tests/python/test_published_pfnn_g1_viewer.py`: latest-frame and expected-world tests.
- `tests/python/test_published_pfnn_heightmap.py`: released mesh-parity tests.
- `tests/python/test_published_pfnn_g1_scenes.py`: exact six-scene manifest tests.
- `tests/python/test_published_pfnn_g1_prepare.py`: source authentication and patch/mesh preparation tests.
- `tests/python/test_published_pfnn_g1_launcher.py`: command construction and exact-PID lifecycle tests.

---

### Task 1: Preserve the Working Data Plane

**Files:**
- Create: `sonic/python/mm_sonic/published_pfnn_g1_bridge.py`
- Create: `sonic/python/mm_sonic/published_pfnn_g1_viewer.py`
- Create: `sonic/python/mm_sonic/published_pfnn_heightmap.py`
- Create: `tests/python/test_published_pfnn_g1_bridge.py`
- Create: `tests/python/test_published_pfnn_g1_viewer.py`
- Create: `tests/python/test_published_pfnn_heightmap.py`

**Interfaces:**
- Consumes: `/tmp/pfnn_g1_bridge.py` SHA `60a065743bfef6cecde003fd15244d32cc613911390eabe8e3ec5d994a4f6429`, `/tmp/g1_live_retarget_viewer.py` SHA `a358fbb631631a1098dfeee786a16bf3911c6bdcbd9c8f1ddf957428dd090868`, and `/tmp/pfnn_heightmap_mesh.py` SHA `cd20053711a45b746692a7f80df51b6da78e2f7fc8d19fe1812f1cd9399e3dff`.
- Produces: importable modules with existing `read_header`, `read_record`, `G1RetargetBridge`, `run_stream`, `Frame`, `_put_frame`, `_open_frames`, `build_mesh`, and `main` interfaces.

- [ ] **Step 1: Write import and behavior tests before adding modules**

Adapt the existing `/tmp/test_pfnn_g1_bridge.py`, `/tmp/test_g1_live_retarget_viewer.py`, and `/tmp/test_pfnn_heightmap_mesh.py` tests to import from `mm_sonic`. Retain the real binary record parsing, pinned GMR passthrough, latest-qpos replacement, raw-frame sequencing, and mesh-value parity assertions.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python -m unittest -v \
  tests.python.test_published_pfnn_g1_bridge \
  tests.python.test_published_pfnn_g1_viewer \
  tests.python.test_published_pfnn_heightmap
```

Expected: import errors for the three missing `mm_sonic.published_pfnn_*` modules.

- [ ] **Step 3: Relocate the validated implementations with no motion changes**

Add the three modules using the exact validated `/tmp` implementations. Change only module docstrings/import paths required by repository placement. Preserve these critical lines:

```python
SOURCE_POSITION_SCALE_M = 1.0 / 100.0
GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"

qpos = np.asarray(retargeter.retarget(human), dtype=np.float64)

if frame.qpos_wxyz is None:
    output.put(frame)
    return
```

The viewer must continue publishing `qpos_wxyz` and `source_root_pos_zup_m` unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all relocated bridge/viewer/mesh tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/published_pfnn_g1_bridge.py \
  sonic/python/mm_sonic/published_pfnn_g1_viewer.py \
  sonic/python/mm_sonic/published_pfnn_heightmap.py \
  tests/python/test_published_pfnn_g1_bridge.py \
  tests/python/test_published_pfnn_g1_viewer.py \
  tests/python/test_published_pfnn_heightmap.py
git commit -m "feat: preserve published PFNN G1 data path"
```

---

### Task 2: Bind Scenes and Reject Mismatches

**Files:**
- Create: `sonic/python/mm_sonic/published_pfnn_g1_scenes.json`
- Create: `sonic/python/mm_sonic/published_pfnn_g1_scenes.py`
- Modify: `sonic/python/mm_sonic/published_pfnn_g1_viewer.py`
- Create: `tests/python/test_published_pfnn_g1_scenes.py`
- Modify: `tests/python/test_published_pfnn_g1_viewer.py`

**Interfaces:**
- Produces: `SceneSpec`, `load_scenes(path: Path | None = None) -> dict[int, SceneSpec]`, and viewer CLI `--expected-world WORLD_ID`.
- Consumes: `Frame.metadata["world"]` from Task 1.

- [ ] **Step 1: Write failing scene and mismatch tests**

The scene test must assert the complete mapping:

```python
expected = {
    1: (0, 1, "hmap_000_smooth.txt"),
    2: (1, 2, "hmap_000_smooth.txt"),
    3: (2, 3, "hmap_004_smooth.txt"),
    4: (3, 4, "hmap_007_smooth.txt"),
    5: (4, 5, "hmap_013_smooth.txt"),
    6: (5, 6, "hmap_urban_001_smooth.txt"),
}
```

The viewer test must construct a `Frame(metadata={"world": 4})`, request world 5, and assert a clear `ValueError` rather than rendering it.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python -m unittest -v \
  tests.python.test_published_pfnn_g1_scenes \
  tests.python.test_published_pfnn_g1_viewer
```

Expected: missing manifest loader and missing expected-world validator.

- [ ] **Step 3: Add the manifest, strict loader, and viewer gate**

Implement the public data type and validation:

```python
@dataclass(frozen=True)
class SceneSpec:
    scene: int
    world_id: int
    key: int
    heightmap: str
    display_stride: int


def require_expected_world(frame: Frame, expected_world: int | None) -> None:
    if expected_world is None:
        return
    actual = frame.metadata.get("world")
    if actual is None or int(actual) != expected_world:
        raise ValueError(
            f"PFNN world mismatch: expected {expected_world}, received {actual}"
        )
```

Call `require_expected_world` before the viewer accepts each frame. Add `--expected-world` with choices `0..5`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command. Expected: all scene and viewer tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add sonic/python/mm_sonic/published_pfnn_g1_scenes.json \
  sonic/python/mm_sonic/published_pfnn_g1_scenes.py \
  sonic/python/mm_sonic/published_pfnn_g1_viewer.py \
  tests/python/test_published_pfnn_g1_scenes.py \
  tests/python/test_published_pfnn_g1_viewer.py
git commit -m "feat: bind published PFNN scenes exactly"
```

---

### Task 3: Preserve and Prepare the Exporter

**Files:**
- Create: `sonic/resources/published_pfnn_g1/pfnn_export.patch`
- Create: `sonic/resources/published_pfnn_g1/EXPORT_SCHEMA.md`
- Create: `sonic/python/mm_sonic/published_pfnn_g1_prepare.py`
- Create: `tests/python/test_published_pfnn_g1_prepare.py`

**Interfaces:**
- Produces: `prepare_exporter(source_demo: Path, cache_demo: Path, resources: Path) -> Path` and `prepare_terrain(scene: SceneSpec, source_demo: Path, cache_root: Path) -> Path`.
- Consumes: `SceneSpec` and `build_mesh` from Tasks 1–2.

- [ ] **Step 1: Write failing source-authentication and preparation tests**

Tests must verify:

```python
with self.assertRaisesRegex(ValueError, "released pfnn.cpp SHA"):
    prepare_exporter(tampered_demo, cache_demo, resources)

mesh = prepare_terrain(scenes[6], released_demo, cache_root)
with np.load(mesh) as archive:
    self.assertEqual(archive["faces"].shape[1], 3)
```

Also apply the patch to a temporary copy of the real released source and assert the result contains `PFNNXFM`, `--export`, and `--world`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_published_pfnn_g1_prepare
```

Expected: missing preparation module.

- [ ] **Step 3: Check in the exact exporter patch and schema**

Generate the patch against the authenticated source, then add deterministic startup:

```cpp
static int initial_world = 0;

static void load_world(int world) {
  switch (world) {
    case 0: load_world0(); break;
    case 1: load_world1(); break;
    case 2: load_world2(); break;
    case 3: load_world3(); break;
    case 4: load_world4(); break;
    case 5: load_world5(); break;
  }
}
```

Parse `--world 0..5`, reject other values, and replace the unconditional `load_world0()` call with `load_world(initial_world)`. Preserve the existing exporter stream implementation and copy `/tmp/pfnn-export-demo/EXPORT_SCHEMA.md` unchanged.

- [ ] **Step 4: Implement authenticated preparation**

`prepare_exporter` must verify the upstream SHA, copy the released demo into a stable cache, apply the checked-in patch, and compile directly without a shell:

```python
subprocess.run(
    [
        "g++", "-std=gnu++11", "-Wall", "-O3", "-ffast-math",
        "pfnn.cpp", "-lGL", "-lGLEW", "-lSDL2", "-g", "-o", "pfnn_export",
    ],
    cwd=cache_demo,
    check=True,
)
```

`prepare_terrain` must call `build_mesh(..., uniform_scale=0.875, stride=scene.display_stride)` and cache by input heightmap SHA plus parameters.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all prepare tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add sonic/resources/published_pfnn_g1 \
  sonic/python/mm_sonic/published_pfnn_g1_prepare.py \
  tests/python/test_published_pfnn_g1_prepare.py
git commit -m "feat: prepare authenticated published PFNN exporter"
```

---

### Task 4: Add the Synchronized Launcher

**Files:**
- Create: `sonic/python/mm_sonic/published_pfnn_g1_launcher.py`
- Create: `tests/python/test_published_pfnn_g1_launcher.py`

**Interfaces:**
- Produces CLI: `python -m mm_sonic.published_pfnn_g1_launcher {prepare,start,switch,status,stop}`.
- Consumes: Tasks 1–3 modules and cached artifacts.

- [ ] **Step 1: Write failing command and lifecycle tests**

Test `build_launch_spec(scene=6, ...)` without starting a viewer. Assert:

```python
self.assertIn("--world", spec.exporter_command)
self.assertIn("5", spec.exporter_command)
self.assertIn("--expected-world", spec.viewer_command)
self.assertIn("5", spec.viewer_command)
self.assertEqual(spec.terrain_path.name, expected_scene6_mesh_name)
```

Spawn a real `sleep 60` subprocess for the lifecycle test, record its PID, `/proc/<pid>/stat` start time, and command digest, and prove `stop_recorded_process` terminates it. A deliberately mismatched start time or command digest must be rejected without signaling the process.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python \
  -m unittest -v tests.python.test_published_pfnn_g1_launcher
```

Expected: missing launcher module.

- [ ] **Step 3: Implement pure launch construction and authenticated process records**

Use immutable records:

```python
@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    command_sha256: str


@dataclass(frozen=True)
class LaunchSpec:
    scene: SceneSpec
    exporter_command: tuple[str, ...]
    bridge_command: tuple[str, ...]
    viewer_command: tuple[str, ...]
    terrain_path: Path
```

Read start ticks from `/proc/<pid>/stat` field 22 and hash NUL-separated `/proc/<pid>/cmdline`. Never identify a process by name alone.

- [ ] **Step 4: Implement CLI lifecycle**

Defaults:

```python
DEFAULT_SCENE = 6
DEFAULT_RUNTIME_ROOT = Path.home() / ".cache/native-g1-pfnn/published-g1-live"
DEFAULT_DISPLAY = ":1"
```

`start` must refuse an authenticated live receipt, call preparation, make a per-run FIFO/log directory, start bridge and viewer with a direct pipe, start the exporter with `--world`, and write the receipt atomically. `switch` calls authenticated stop then start. `status` emits JSON with scene/world/PIDs and live identity results. `stop` sends `SIGTERM` only after every process identity matches, waits, and reports any survivor without escalating to `SIGKILL`.

Add `--dry-run` to `prepare/start/switch`; it prints the resolved manifest, paths, and commands without launching or signaling.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all launcher tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add sonic/python/mm_sonic/published_pfnn_g1_launcher.py \
  tests/python/test_published_pfnn_g1_launcher.py
git commit -m "feat: launch synchronized published PFNN G1 scenes"
```

---

### Task 5: Runbook, Full Verification, and Live Handoff

**Files:**
- Create: `sonic/resources/published_pfnn_g1/README.md`
- Modify only if required by verification: files from Tasks 1–4 and their direct tests.

**Interfaces:**
- Produces the operator commands below and a verified packaged scene-6 process.

- [ ] **Step 1: Write the concise runbook**

Document exactly:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher prepare
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher start --scene 6
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher switch --scene 5
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher status
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher stop
```

State that controls remain in the published PFNN window: WASD movement, Shift run, Ctrl strafe, arrows camera, Q/E zoom.

- [ ] **Step 2: Run the complete non-live verification**

Run:

```bash
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python -m unittest -v \
  tests.python.test_published_pfnn_g1_bridge \
  tests.python.test_published_pfnn_g1_viewer \
  tests.python.test_published_pfnn_heightmap \
  tests.python.test_published_pfnn_g1_scenes \
  tests.python.test_published_pfnn_g1_prepare \
  tests.python.test_published_pfnn_g1_launcher
PYTHONPATH=sonic/python /home/ubuntu/.cache/native-g1-pfnn/venv/bin/python -m py_compile \
  sonic/python/mm_sonic/published_pfnn_*.py
git diff --check
```

Expected: every focused test passes, compilation exits 0, and diff check is clean.

- [ ] **Step 3: Verify dry-run scene pairing for all six scenes**

Run `start --dry-run --scene N` for `N=1..6`. Parse each JSON result and assert exporter `--world` and viewer `--expected-world` agree with the manifest.

- [ ] **Step 4: Replace the manual run only after all prior gates pass**

Resolve the current manual exporter, bridge, and viewer PIDs by their exact commands; stop only those verified processes. Start the packaged launcher with scene 6. Do not use a broad `pkill` pattern.

- [ ] **Step 5: Verify the live packaged scene**

Run `status` and inspect logs. Required evidence:

- exporter and viewer both report world ID 5;
- viewer queue remains 0 or 1;
- both PFNN and MuJoCo windows are visible;
- a bounded W press changes the newest G1 root without a multi-second backlog;
- `status` authenticates every recorded PID.

- [ ] **Step 6: Commit the runbook and final bounded fixes**

```bash
git add sonic/resources/published_pfnn_g1/README.md
git commit -m "docs: add published PFNN G1 runbook"
```

Record the final test count, commit hashes, runtime receipt, and active scene in the handoff.
