# Stable G1 Playable Mesh Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch the stable classic-Raygui G1 terrain motion-matching controller with its certified sideways/diagonal and independent-heading controls plus the corrected 35-part G1 mesh.

**Architecture:** Create an isolated playable branch from stable controller commit `8f7ead2`, leaving `g1-footprint-task6` unchanged. Import the already-certified mesh asset and renderer, then make a narrow controller integration that loads once, transfers the final stable global pose once per frame, draws inside the existing 3D pass, exposes mesh/bone toggles, and unloads through the existing normal-cleanup path.

**Tech Stack:** C++17, GCC, Raylib, Raygui, Python 3 `unittest`, glTF/GLB validation, X11/`xdotool`, Git.

## Global Constraints

- Preserve the controller behavior from commit `8f7ead2`.
- Preserve the classic Raygui layout, keyboard/gamepad input, orbit camera, terrain scenes, and exact 25 Hz simulation cadence.
- Preserve independent desired travel and desired heading commands, including forward, backward, lateral, and diagonal prediction.
- Do not change motion matching, cost weights, search cadence, thresholds, terrain sampling, contact logic, foot processing, or IK math.
- Do not import the transactional runtime, bounded-candidate experiments, or unfinished best-effort IK work.
- Feed the mesh only the final `state.global_bone_positions` and `state.global_bone_rotations` produced by the stable runtime.
- Apply no empirical scale, root, axis, ankle, foot, or sole correction.
- Keep terrain data and motion data separate.
- Commit and push every coherent verified checkpoint to `checkpoint/g1-playable-mesh`.
- Launch exactly one new interactive visualizer only after all finite and live-input gates pass.

## Execution Setup

Use `superpowers:using-git-worktrees` before Task 1. Create branch
`g1-playable-mesh` at exact commit `8f7ead2` in:

```text
/home/ubuntu/projects/motion-matching/.worktrees/g1-playable-mesh
```

All task commands run from that worktree unless a command gives an absolute
path. The ignored terrain artifact pack remains external and is selected with:

```bash
export G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain
```

---

### Task 1: Import the Corrected Mesh Asset into the Stable Branch

**Files:**

- Create: `resources/g1_mesh/__init__.py`
- Create: `resources/g1_mesh/export_g1_raylib_glb.py`
- Create: `resources/g1_mesh/g1_raylib.glb`
- Create: `resources/g1_mesh/manifest.json`
- Create: `resources/g1_mesh/validate_g1_raylib_glb.py`
- Create: `tests/python/test_g1_mesh_asset.py`
- Create: `docs/superpowers/specs/2026-07-18-g1-stable-playable-mesh-design.md`
- Create: `docs/superpowers/plans/2026-07-18-g1-stable-playable-mesh.md`

**Interfaces:**

- Consumes: corrected asset tree exactly as recorded at `dbb5f41` and approved design/plan documents from branch `g1-footprint-task6`.
- Produces: validated `resources/g1_mesh/g1_raylib.glb`, its authenticated manifest, and reproducible validator/exporter code for Task 2.

- [x] **Step 1: Confirm the isolated stable base before changing files**

```bash
test "$(git rev-parse HEAD)" = "$(git rev-parse 8f7ead2)"
test "$(git branch --show-current)" = g1-playable-mesh
git diff --exit-code
git grep -F 'terrain scene / runtime' HEAD -- controller.cpp
git grep -F 'G1CommandIntent command_intent;' HEAD -- controller.cpp
git grep -F 'predicted_desired_headings' HEAD -- controller.cpp
git grep -F 'run sideways speed' HEAD -- controller.cpp
git grep -F 'walk sideways speed' HEAD -- controller.cpp
! git grep -F 'Transactional terrain IK' HEAD -- controller.cpp
```

Expected: every command exits `0`; the stable controller contains the classic
UI and independent travel/heading/lateral paths and does not contain the
transactional UI.

- [x] **Step 2: Install the asset test first**

```bash
git restore --source=dbb5f41 -- tests/python/test_g1_mesh_asset.py
```

This restores only the accepted test. Do not restore the asset implementation
yet.

- [x] **Step 3: Run the asset test and verify the intended RED failure**

```bash
python3 -m unittest tests.python.test_g1_mesh_asset -v
```

Expected: FAIL during import because `resources.g1_mesh` does not exist on the
stable branch. A different failure must be diagnosed before continuing.

- [x] **Step 4: Import the exact corrected asset implementation**

```bash
git restore --source=dbb5f41 -- resources/g1_mesh
git restore --source=g1-footprint-task6 -- \
  docs/superpowers/specs/2026-07-18-g1-stable-playable-mesh-design.md \
  docs/superpowers/plans/2026-07-18-g1-stable-playable-mesh.md
```

Expected: `git status --short` lists only the asset package, its test, and the
two approved documents.

- [x] **Step 5: Run the corrected asset GREEN gate**

```bash
python3 -m unittest tests.python.test_g1_mesh_asset -v
python3 - <<'PY'
import json
from pathlib import Path
from resources.g1_mesh.validate_g1_raylib_glb import validate_g1_glb

report = validate_g1_glb(
    Path("resources/g1_mesh/g1_raylib.glb"),
    Path("resources/g1_mesh/manifest.json"),
)
print(json.dumps(report, sort_keys=True))
assert report["primitive_count"] == 35
assert report["bone_count"] == 39
PY
```

Expected: all seven unit tests pass; the validator exits `0`, reports 35 meshes
and 39 bones, and reports no bind-frame, hash, rigid-weight, or index-limit
error.

- [x] **Step 6: Commit and push the asset checkpoint**

```bash
git add resources/g1_mesh tests/python/test_g1_mesh_asset.py \
  docs/superpowers/specs/2026-07-18-g1-stable-playable-mesh-design.md \
  docs/superpowers/plans/2026-07-18-g1-stable-playable-mesh.md
git commit -m "feat: import corrected G1 playable mesh asset"
git push -u checkpoint g1-playable-mesh
```

Expected: commit succeeds, the remote branch is created, and
`git status --short` is empty.

---

### Task 2: Port the Renderer into the Stable Classic UI

**Files:**

- Create: `g1_mesh_renderer.h`
- Create: `tests/cpp/test_g1_mesh_renderer.cpp`
- Create: `tests/python/test_g1_playable_mesh_integration.py`
- Modify: `controller.cpp:35-55,1969-2050,2160-2200,3488-3580,3768-3795,3960-4000`

**Interfaces:**

- Consumes: Task 1 GLB path `resources/g1_mesh/g1_raylib.glb`; stable final pose slices `state.global_bone_positions` and `state.global_bone_rotations`; existing `normal_cleanup` lifecycle.
- Produces: `G1MeshRenderer` load/update/draw/unload integration; local booleans `show_g1_mesh` and `show_g1_bones`; `M` and `B` edge-triggered display toggles.

- [x] **Step 1: Add a source-contract test before production code**

Create `tests/python/test_g1_playable_mesh_integration.py` with:

```python
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "controller.cpp").read_text(encoding="utf-8")


class G1PlayableMeshIntegrationTests(unittest.TestCase):
    def test_classic_ui_and_control_contract_remains(self):
        for text in (
            '"terrain scene / runtime"',
            '"controls"',
            '"WASD / left stick - move"',
            '"Arrows / right stick - camera"',
            '"Left trigger - strafe"',
            '"run sideways speed"',
            '"walk sideways speed"',
            "G1CommandIntent command_intent;",
            "predicted_desired_headings",
        ):
            self.assertIn(text, SOURCE)
        self.assertNotIn("Transactional terrain IK", SOURCE)
        self.assertNotIn("frame_runtime", SOURCE)

    def test_mesh_uses_only_the_final_stable_global_pose(self):
        self.assertIn('#include "g1_mesh_renderer.h"', SOURCE)
        self.assertEqual(
            SOURCE.count(
                "if (::IsKeyPressed(KEY_M)) show_g1_mesh = !show_g1_mesh;"
            ),
            1,
        )
        self.assertEqual(
            SOURCE.count(
                "if (::IsKeyPressed(KEY_B)) show_g1_bones = !show_g1_bones;"
            ),
            1,
        )
        self.assertIn("bool show_g1_mesh = true;", SOURCE)
        self.assertIn("bool show_g1_bones = true;", SOURCE)

        update = SOURCE.index("::g1_mesh_renderer_update(")
        update_end = SOURCE.index(");", update)
        update_call = SOURCE[update:update_end]
        self.assertIn("state.global_bone_positions", update_call)
        self.assertIn("state.global_bone_rotations", update_call)
        self.assertNotIn("adjusted_bone_positions", update_call)
        self.assertNotIn("adjusted_bone_rotations", update_call)

        begin = SOURCE.index("BeginMode3D(camera);", update)
        draw = SOURCE.index(
            "::g1_mesh_renderer_draw(g1_mesh_renderer);", begin
        )
        end = SOURCE.index("EndMode3D();", draw)
        self.assertLess(update, begin)
        self.assertLess(begin, draw)
        self.assertLess(draw, end)

    def test_mesh_lifecycle_uses_existing_normal_cleanup(self):
        load = SOURCE.index("::g1_mesh_renderer_load(")
        loop = SOURCE.index("auto update_func = [&]()")
        cleanup = SOURCE.index("auto normal_cleanup = [&]()")
        unload = SOURCE.index(
            "::g1_mesh_renderer_unload(g1_mesh_renderer);", cleanup
        )
        terrain_unload = SOURCE.index("model_unloader(terrain_model);", unload)
        close = SOURCE.index("CloseWindow();", terrain_unload)
        self.assertLess(load, loop)
        self.assertLess(cleanup, unload)
        self.assertLess(unload, terrain_unload)
        self.assertLess(terrain_unload, close)
        self.assertEqual(
            SOURCE.count("::g1_mesh_renderer_unload(g1_mesh_renderer);"), 1
        )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the integration test and verify RED**

```bash
python3 -m unittest tests.python.test_g1_playable_mesh_integration -v
```

Expected: the classic UI/control test passes; the mesh pose and lifecycle tests
fail because the stable controller has no mesh renderer yet.

- [x] **Step 3: Install the renderer component test and verify RED before importing the header**

```bash
git restore --source=2080266 -- tests/cpp/test_g1_mesh_renderer.cpp
```

In `tests/cpp/test_g1_mesh_renderer.cpp`, remove the two controller-source
tests and their file-reading helpers beginning at `static std::string
read_text`. Retain all component tests through
`test_unload_is_idempotent_for_canonical_renderer`. The final `main` is exactly:

```cpp
int main()
{
    test_exact_articulated_mapping();
    test_fixed_attachment_follows_parent_bind_local();
    test_rest_pose_reproduces_global_bind_pose();
    test_binding_rejects_malformed_layouts();
    test_pose_rejects_invalid_inputs_transactionally();
    test_unload_is_idempotent_for_canonical_renderer();
    return 0;
}
```

Also remove now-unused `<fstream>`, `<iterator>`, and `<string>` includes. The
Python test owns controller integration; the C++ test owns renderer math,
mapping, validation, and idempotent unload.

Run the component build before restoring the production header:

```bash
mkdir -p /tmp/g1-playable-mesh-build
if g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_mesh_renderer.cpp \
  -o /tmp/g1-playable-mesh-build/test-g1-mesh-renderer \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11; then
  exit 1
fi
```

Expected: compilation fails only because `g1_mesh_renderer.h` is missing.

- [x] **Step 4: Import the certified renderer component and verify its GREEN gate**

```bash
git restore --source=2080266 -- g1_mesh_renderer.h
g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_mesh_renderer.cpp \
  -o /tmp/g1-playable-mesh-build/test-g1-mesh-renderer \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/tmp/g1-playable-mesh-build/test-g1-mesh-renderer
```

Expected: compilation succeeds and the component test exits `0` with no
output.

- [x] **Step 5: Add the renderer include and one owned runtime instance**

Add the include beside the other G1 runtime owners:

```cpp
#include "cleanup_runtime.h"
#include "g1_mesh_renderer.h"
```

Immediately after the terrain model readiness gate, add:

```cpp
    G1MeshRenderer g1_mesh_renderer = {};
    bool show_g1_mesh = true;
    bool show_g1_bones = true;
    if (!controller_exit_requested && !::g1_mesh_renderer_load(
            g1_mesh_renderer,
            "resources/g1_mesh/g1_raylib.glb",
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        std::fprintf(stderr, "G1 mesh load error: %s\n", artifact_error);
        controller_exit_code = 2;
        controller_exit_requested = true;
    }
    if (!controller_exit_requested)
    {
        std::fprintf(
            stdout,
            "G1 mesh loaded: %d parts\n",
            g1_mesh_renderer.model.meshCount);
    }
```

Do not change terrain `model_load_count` or `model_unload_count`; those counters
remain terrain-only.

- [x] **Step 6: Transfer and draw the final stable pose**

At the beginning of `update_func`, add the edge-triggered display controls:

```cpp
        if (::IsKeyPressed(KEY_M)) show_g1_mesh = !show_g1_mesh;
        if (::IsKeyPressed(KEY_B)) show_g1_bones = !show_g1_bones;
```

After final foot processing and camera update, but before `BeginDrawing`, add:

```cpp
        if (!::g1_mesh_renderer_update(
                g1_mesh_renderer,
                state.global_bone_positions,
                state.global_bone_rotations,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }
```

Inside the existing `BeginMode3D(camera)` block, after terrain drawing and
before the skeleton, add:

```cpp
        if (show_g1_mesh)
        {
            ::g1_mesh_renderer_draw(g1_mesh_renderer);
        }
```

Wrap the unchanged skeleton loop in:

```cpp
        if (show_g1_bones)
        {
            for (int bi = 1; bi < db.nbones(); bi++)
            {
                vec3 bp = state.global_bone_positions(bi);
                DrawSphereWires(to_Vector3(bp), 0.028f, 4, 8, DARKBLUE);
                int par = db.bone_parents(bi);
                if (par > 0)
                {
                    DrawCylinderEx(
                        to_Vector3(state.global_bone_positions(par)),
                        to_Vector3(bp),
                        0.018f,
                        0.018f,
                        6,
                        SKYBLUE);
                }
            }
        }
```

- [x] **Step 7: Preserve the classic UI and expose only two display toggles**

Keep every existing Raygui group, slider, and control label. Add one line at
the bottom of the existing `controls` group:

```cpp
        GuiLabel(
            Rectangle{ 990, ui_ctrl_hei + 135, 250, 20 },
            TextFormat(
                "M mesh %s | B bones %s",
                show_g1_mesh ? "ON" : "OFF",
                show_g1_bones ? "ON" : "OFF"));
```

No other UI dimensions or labels change.

- [x] **Step 8: Route mesh ownership through normal cleanup**

In `normal_cleanup`, immediately before `model_unloader(terrain_model);`, add:

```cpp
        ::g1_mesh_renderer_unload(g1_mesh_renderer);
        model_unloader(terrain_model);
```

There must be exactly one controller call to `g1_mesh_renderer_unload`.

- [x] **Step 9: Run focused GREEN tests and build the stable controller**

```bash
python3 -m unittest \
  tests.python.test_g1_playable_mesh_integration \
  tests.python.test_g1_mesh_asset -v

mkdir -p /tmp/g1-playable-mesh-build
g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_mesh_renderer.cpp \
  -o /tmp/g1-playable-mesh-build/test-g1-mesh-renderer \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/tmp/g1-playable-mesh-build/test-g1-mesh-renderer

g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/g1-playable-mesh-build/controller-g1-playable \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: Python tests pass; the C++ renderer test exits `0`; the controller
build exits `0`. `strings /tmp/g1-playable-mesh-build/controller-g1-playable`
contains `terrain scene / runtime`, `run sideways speed`, `M mesh`, and
`G1 mesh loaded`, and does not contain `Transactional terrain IK`.

- [x] **Step 10: Commit and push the playable renderer checkpoint**

```bash
git add controller.cpp g1_mesh_renderer.h \
  tests/cpp/test_g1_mesh_renderer.cpp \
  tests/python/test_g1_playable_mesh_integration.py
git commit -m "feat: render G1 mesh in stable playable controller"
git push checkpoint g1-playable-mesh
```

Expected: commit and push succeed and `git status --short` is empty.

---

### Task 3: Certify Real Controls and Launch the User Visualizer

**Files:**

- Modify: `docs/superpowers/plans/2026-07-18-g1-stable-playable-mesh.md` (mark completed checkpoints)
- Build only: `/tmp/g1-playable-mesh-final/controller-g1-playable`
- Runtime evidence only: `/tmp/g1-playable-mesh-final/*.csv`, `*.out`, `*.err`, `*.pid`

**Interfaces:**

- Consumes: Task 2 controller binary contract; external authenticated terrain pack; X11 display `:1`.
- Produces: deterministic 32-frame evidence, live forward/back/lateral/diagonal/strafe/camera input evidence, a pushed source checkpoint, and one running classic-UI mesh visualizer.

- [x] **Execution amendment: atomically refresh the stale active terrain pack before certification**

The exact Step 2 command against the prescribed active path failed closed with
`scene route count changed`: that path held the authenticated July-14 pack,
while the reviewed stable controller requires the authenticated July-15
directional-route metadata already present at
`/tmp/g1-terrain-footprint-runtime-v1`. The trees contain the same 62 files and
differ only in `manifest.json`, `scenes/index.json`, and the route-bearing
`scene.json` files for `stairs-shallow`, `stairs-standard`, and
`mixed-multilevel`. The protected `database.bin`, `terrain_features.bin`,
`terrain_support.bin`, and `validation.json` payloads, plus every terrain,
mesh, and walkability payload, are byte-identical. Before exchange, validate
the candidate, prove the exact five-file difference and zero exact final-binary
processes, and retain `/tmp/g1-terrain-active-v1.SHA256SUMS` as the stale-pack
rollback oracle. Use the repository `_rename_exchange` while holding both
parent-directory locks, then fsync both parents. Validate the new active pack
and recheck the four protected hashes; on any failure, immediately exchange
back, fsync both parents, and prove the rollback oracle. After success, the old
active pack remains intact at the former candidate path and must not be used by
the remaining gates.

- [x] **Step 1: Rebuild all focused tests from the reviewed commit**

```bash
rm -rf /tmp/g1-playable-mesh-final
mkdir -p /tmp/g1-playable-mesh-final
python3 -m unittest \
  tests.python.test_g1_playable_mesh_integration \
  tests.python.test_g1_mesh_asset \
  tests.python.test_runtime_log -v

g++ -std=c++17 -O2 -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_mesh_renderer.cpp \
  -o /tmp/g1-playable-mesh-final/test-g1-mesh-renderer \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/tmp/g1-playable-mesh-final/test-g1-mesh-renderer

g++ -std=c++17 -O2 -I. tests/cpp/test_g1_command_runtime.cpp \
  -o /tmp/g1-playable-mesh-final/test-g1-command-runtime
/tmp/g1-playable-mesh-final/test-g1-command-runtime

g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/g1-playable-mesh-final/controller-g1-playable \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: all tests and builds exit `0` from a clean output directory.

- [x] **Step 2: Run the deterministic mixed-multilevel 32-frame gate**

```bash
DISPLAY=:1 \
G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
MM_TERRAIN_SCENE=mixed-multilevel \
MM_TEST_MODE=route \
MM_TEST_ROUTE=full-course \
MM_TEST_FRAMES=32 \
MM_LOG=/tmp/g1-playable-mesh-final/mixed-32.csv \
/tmp/g1-playable-mesh-final/controller-g1-playable \
  >/tmp/g1-playable-mesh-final/mixed-32.out \
  2>/tmp/g1-playable-mesh-final/mixed-32.err

python3 resources/check_g1_runtime_log.py \
  /tmp/g1-playable-mesh-final/mixed-32.csv
test "$(wc -l </tmp/g1-playable-mesh-final/mixed-32.csv)" -eq 33
rg -F 'G1 mesh loaded: 35 parts' \
  /tmp/g1-playable-mesh-final/mixed-32.out
test ! -s /tmp/g1-playable-mesh-final/mixed-32.err
```

Expected: runtime exits `0`; checker prints `VALID runtime-log frames=32`;
the CSV has one header plus 32 frames; the corrected asset reports 35 parts;
stderr is empty.

- [x] **Step 3: Exercise real keyboard input against the live event path**

```bash
DISPLAY=:1 \
G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
MM_TERRAIN_SCENE=mixed-multilevel \
/tmp/g1-playable-mesh-final/controller-g1-playable \
  >/tmp/g1-playable-mesh-final/live-smoke.out \
  2>/tmp/g1-playable-mesh-final/live-smoke.err &
live_pid=$!

printf '%s\n' "$live_pid" >/tmp/g1-playable-mesh-final/live-smoke.pid
for attempt in $(seq 1 100); do
  test -e "/proc/$live_pid/exe" && break
  sleep 0.1
done
test "$(readlink -f "/proc/$live_pid/exe")" = \
  /tmp/g1-playable-mesh-final/controller-g1-playable

window_id=$(DISPLAY=:1 xdotool search --sync --pid "$live_pid" \
  --name 'G1 terrain motion matching - Holden runtime' | head -1)
DISPLAY=:1 xdotool windowactivate --sync "$window_id"
DISPLAY=:1 xdotool keydown w
DISPLAY=:1 xdotool sleep 0.8
DISPLAY=:1 xdotool keyup w
DISPLAY=:1 xdotool keydown s
DISPLAY=:1 xdotool sleep 0.5
DISPLAY=:1 xdotool keyup s
DISPLAY=:1 xdotool keydown a
DISPLAY=:1 xdotool sleep 0.6
DISPLAY=:1 xdotool keyup a
DISPLAY=:1 xdotool keydown w keydown d
DISPLAY=:1 xdotool sleep 0.8
DISPLAY=:1 xdotool keyup d keyup w
DISPLAY=:1 xdotool keydown Control_L keydown a
DISPLAY=:1 xdotool sleep 0.8
DISPLAY=:1 xdotool keyup a keyup Control_L
DISPLAY=:1 xdotool keydown Right
DISPLAY=:1 xdotool sleep 0.5
DISPLAY=:1 xdotool keyup Right
DISPLAY=:1 xdotool key m b b
DISPLAY=:1 xdotool sleep 0.5

kill -0 "$live_pid"
rg -F 'G1 mesh loaded: 35 parts' \
  /tmp/g1-playable-mesh-final/live-smoke.out
! rg -n 'controlled runtime error|mesh load error|Segmentation|Assertion' \
  /tmp/g1-playable-mesh-final/live-smoke.err
test "$(readlink -f "/proc/$live_pid/exe")" = \
  /tmp/g1-playable-mesh-final/controller-g1-playable
kill -TERM "$live_pid"
if wait "$live_pid"; then
  true
else
  test "$?" -eq 143
fi
```

Expected: one live window accepts forward, backward, lateral, diagonal,
Ctrl-strafe, camera, mesh-toggle, and bone-toggle events without exiting or
printing an error. The identity-checked smoke process then terminates cleanly.

- [x] **Step 4: Record and push the certified checkpoint**

Mark every completed plan checkbox `[x]`, then run:

```bash
git add docs/superpowers/plans/2026-07-18-g1-stable-playable-mesh.md
git commit -m "test: certify stable G1 playable mesh visualizer"
git push checkpoint g1-playable-mesh
git status --short --branch
```

Expected: the plan checkpoint is pushed and the worktree is clean.

- [x] **Step 5: Launch exactly one user visualizer on mixed terrain**

```bash
nohup env \
  DISPLAY=:1 \
  G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel \
  /tmp/g1-playable-mesh-final/controller-g1-playable \
  >/tmp/g1-playable-mesh-final/user.out \
  2>/tmp/g1-playable-mesh-final/user.err &
user_pid=$!

printf '%s\n' "$user_pid" >/tmp/g1-playable-mesh-final/user.pid
for attempt in $(seq 1 100); do
  test -e "/proc/$user_pid/exe" && break
  sleep 0.1
done
test "$(readlink -f "/proc/$user_pid/exe")" = \
  /tmp/g1-playable-mesh-final/controller-g1-playable
kill -0 "$user_pid"
rg -F 'G1 mesh loaded: 35 parts' \
  /tmp/g1-playable-mesh-final/user.out
test ! -s /tmp/g1-playable-mesh-final/user.err
```

Expected: exactly one new window remains alive with the classic UI and
corrected G1 mesh. The user can use WASD, arrows, Left Ctrl strafe, Left Shift
walk, `M` mesh toggle, and `B` skeleton toggle on `mixed-multilevel` terrain.
