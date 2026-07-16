# G1 SONIC Scene-Aware Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the existing privileged terrain-aware G1 motion matcher can drive an unmodified GEAR-SONIC policy through flat ground, a curb, a ramp, and stairs in MuJoCo, with physics paused while each complete 0.4-second reference chunk is generated and validated.

**Architecture:** Extract the existing 25 Hz matcher step into a shared headless kernel, wrap it in a transactional JSONL chunk server, project the inertialized G1 pose onto the source MJCF's 29 hinge coordinates, and let a Python bridge create one canonical 50 Hz buffer for both official SONIC CSV references and protocol-v1 ZMQ messages. A coordinator owns process readiness, candidate commit/abort, a branch-owned gated MuJoCo runner that imports the pinned external simulator without modifying it, immutable run artifacts, and matched aware/blind evaluation.

**Tech Stack:** C++17, the existing Holden motion-matching runtime, Python 3.10+, NumPy, MuJoCo Python bindings, pyzmq, standard-library `unittest`, JSONL, CSV, SHA-256, GEAR-SONIC/GR00T-WholeBodyControl pinned at `60de0df7ffedeef415fe58d435e92cc5b01ba3d9`, and Linux process groups/PTYS for the sim-only orchestration boundary.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline` on branch `g1-sonic-scene-aware-baseline`, based on terrain commit `c1f97c201228376b22a7a114dfd4c757090383b4`.
- Do not modify, clean, copy from, or commit the user's dirty database/features binaries, executables, logs, videos, or untracked terrain artifacts in the primary terrain workspace. Explicit read-only artifact paths are allowed.
- Keep the root wildcard controller build free of a second `main`; all server translation units live under `sonic/cpp/`.
- This branch tests H1 only. Do not add depth, lidar, root-state feedback, obstacle planning, learned replacement models, DAgger, action chunking, delay injection, hardware deployment, or SONIC training.
- The matcher remains open loop with respect to MuJoCo. Physical pelvis/root state is logged and evaluated but never sent back to MM.
- Joint references come from the inertialized pose before display foot-lock IK. Keep `ik_enabled == false`; never project an off-manifold IK pose into a hinge silently.
- Source time is exact binary32 `0.04f` (25 Hz). The registered experiment chunk is exactly 10 source intervals, 11 source boundaries, 20 new 50 Hz frames, and 0.4 seconds.
- Emit target frame zero once at session start. Each accepted chunk emits 20 new frames after validating and removing its repeated first boundary.
- Joint interpolation is cubic Hermite using source position and velocity; target velocity is the analytic derivative. Pelvis orientation uses normalized shortest-path SLERP. Never clip interpolation overshoot.
- The source basis is `holden-y-up-right-handed-forward-plus-z`; the target vector map is exactly `(x_m, y_m, z_m) = (x_h, -z_h, y_h)`. Global orientations use the corresponding proper quaternion conjugation and are written `wxyz`.
- SONIC receives 29 joints in the pinned IsaacLab order, a root-only pelvis quaternion, zero root position, and metadata `[0]`. It never receives MM root position, root height, or terrain observations.
- ZMQ v1 uses topic bytes `pose`, a 1,280-byte NUL-padded JSON header, little-endian row-major payloads, and fields `joint_pos`, `joint_vel`, `body_quat_w`, `frame_index`, and `catch_up`. Baseline chunks set `catch_up=0` because generation delay is intentionally unbounded while physics is paused.
- No candidate becomes active before complete source validation, target conversion, local enqueue, artifact write, and a successful ZMQ `send` return. Pre-commit failure aborts the MM candidate and leaves both MM and simulation paused.
- GEAR's PUB/SUB transport has no receipt acknowledgement. Readiness is a separate initial-frame preflight; post-run target-row parity and transmitted frame-index continuity are mandatory integration evidence.
- GPU/MuJoCo policy experiments are explicit commands, not ordinary unit tests. A missing external checkout, LFS asset, policy, encoder, observation config, or GPU produces `not_run` evidence and cannot be reported as a hypothesis result.
- Generated scenes, references, subprocess logs, checkpoints, videos, and runs are ignored. Only code, schemas, fixed experiment definitions, and compact machine-readable result summaries may be committed.

## Preconditions and fixed external facts

- The source G1 MJCF is supplied explicitly; the currently registered source is `/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml` and has a free pelvis plus 29 hinge joints.
- The external checkout is supplied explicitly and must resolve to commit `60de0df7ffedeef415fe58d435e92cc5b01ba3d9`. Never write through that path.
- The official source-of-truth files at that commit are:
  - `gear_sonic/envs/manager_env/robots/g1.py` for IsaacLab joint names;
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp` for order permutations;
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_zmq_manager.py` for packed ZMQ v1 framing;
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp` and `streamed_motion_merger.hpp` for decoding/continuity behavior;
  - `gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/` as the registered known-good reference directory.
- The pinned target order, excluding `pelvis`, is:

~~~text
left_hip_pitch_joint,right_hip_pitch_joint,waist_yaw_joint,
left_hip_roll_joint,right_hip_roll_joint,waist_roll_joint,
left_hip_yaw_joint,right_hip_yaw_joint,waist_pitch_joint,
left_knee_joint,right_knee_joint,left_shoulder_pitch_joint,
right_shoulder_pitch_joint,left_ankle_pitch_joint,right_ankle_pitch_joint,
left_shoulder_roll_joint,right_shoulder_roll_joint,left_ankle_roll_joint,
right_ankle_roll_joint,left_shoulder_yaw_joint,right_shoulder_yaw_joint,
left_elbow_joint,right_elbow_joint,left_wrist_roll_joint,
right_wrist_roll_joint,left_wrist_pitch_joint,right_wrist_pitch_joint,
left_wrist_yaw_joint,right_wrist_yaw_joint
~~~

- The target-index-to-source-MJCF-index permutation is exactly:

~~~text
0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28
~~~

## File and ownership map

- `sonic/cpp/g1_runtime.h`: shared, renderer-free 25 Hz matcher step and boundary observation.
- `g1_controller_state.h`: complete mutable matcher state plus checked deep clone/swap.
- `sonic/cpp/g1_joint_projection.h`: named source-hinge projection and residual/range checks.
- `sonic/cpp/mm_chunk_protocol.h`: request/state-machine contracts independent of real artifacts.
- `sonic/cpp/mm_chunk_server.cpp`: persistent JSONL process and real artifact/runtime adapter.
- `sonic/python/mm_sonic/schema.py`: duplicate-key-safe source/target/run contracts.
- `sonic/python/mm_sonic/joints.py`: checked name mapping and joint-contract generation/verification.
- `sonic/python/mm_sonic/transform.py`: Holden/MuJoCo basis and quaternion operations.
- `sonic/python/mm_sonic/resample.py`: 25-to-50 Hz interpolation and seam ownership.
- `sonic/python/mm_sonic/reference.py`: official CSV sink and decoded parity.
- `sonic/python/mm_sonic/zmq_v1.py`: packed-message codec, publisher, and loopback decoder.
- `sonic/python/mm_sonic/scene.py`: source OBJ conversion, run-local GEAR MJCF overlay, geom registry, and kinematic replay.
- `sonic/python/mm_sonic/process.py`: MM client, PTY/process-group GEAR wrapper, and gated simulator client.
- `sonic/python/mm_sonic/gated_sim.py`: branch-owned step-counted MuJoCo runner importing the pinned external code.
- `sonic/python/mm_sonic/coordinator.py`: generate/validate/send/commit/advance state machine.
- `sonic/python/mm_sonic/metrics.py`: tracking, contact, delivery, timing, and hypothesis verdicts.
- `sonic/python/mm_sonic/commands.py`: flat script and frozen route-to-chunk compiler.
- `sonic/python/mm_sonic/cli.py`: `preflight`, `reference`, `stream`, `evaluate`, and `matrix` entry points.
- `sonic/configs/`: dependency lock, committed G1 joint contract, and fixed experiment definitions.
- `sonic/schemas/`: source chunk, target chunk, run manifest, and verdict JSON schemas.
- `sonic/runs/`: ignored, exclusive run directories only.
- `tests/cpp/` and `tests/python/`: focused CPU tests; real-artifact and GPU tests carry explicit environment guards.

---

### Task 1: Establish the isolated package, dependency lock, and fail-closed preflight

**Files:**
- Modify: `.gitignore`
- Create: `sonic/README.md`
- Create: `sonic/pyproject.toml`
- Create: `sonic/cpp/Makefile`
- Create: `sonic/python/mm_sonic/__init__.py`
- Create: `sonic/python/mm_sonic/external.py`
- Create: `sonic/configs/gear_sonic.lock.json`
- Create: `sonic/schemas/run_manifest_v1.schema.json`
- Create: `tests/python/test_sonic_external.py`

**Interfaces:**
- `ExternalInputs.from_cli(args: argparse.Namespace)` resolves every path, checks that inputs are files/directories of the expected type, hashes files without following writes, and rejects any output path nested under an input path.
- `verify_gear_checkout(path, lock)` checks `git rev-parse HEAD`, records dirty status and rejects it for scored runs, verifies required files, checks the pinned permutation text, and verifies known-good reference contents.
- The lock contains no machine-local model path. Policy/checkpoint, observation config, encoder, terrain directory, and source MJCF remain mandatory explicit run inputs and are hashed into the manifest.

- [ ] **Step 1: Write RED tests for external identity and output isolation**

Add `unittest` cases that create a tiny fake Git repository and require rejection for a wrong commit, missing official file, modified permutation, duplicate input/output location, missing checkpoint, and a non-file observation config. Require a valid fixture to return normalized absolute paths and deterministic SHA-256 strings.

The public shape is:

~~~python
@dataclass(frozen=True)
class ExternalInputs:
    gear_checkout: Path
    policy: Path
    observation_config: Path
    encoder: Path | None
    terrain_dir: Path
    source_mjcf: Path

@dataclass(frozen=True)
class VerifiedExternal:
    inputs: ExternalInputs
    gear_commit: str
    hashes: Mapping[str, str]
    known_good_reference: Path
~~~

- [ ] **Step 2: Run the RED test**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_external -v
~~~

Expected: import failure because `mm_sonic.external` does not exist.

- [ ] **Step 3: Add the package, exact external lock, and preflight implementation**

Use this lock content as the fixed contract:

~~~json
{
  "schema": "gear-sonic-lock/v1",
  "repository": "https://github.com/NVlabs/GR00T-WholeBodyControl.git",
  "commit": "60de0df7ffedeef415fe58d435e92cc5b01ba3d9",
  "known_good_reference": "gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428",
  "joint_names_source": "gear_sonic/envs/manager_env/robots/g1.py",
  "policy_parameters_source": "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp",
  "zmq_example_source": "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_zmq_manager.py",
  "zmq_decoder_source": "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp",
  "stream_merger_source": "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp"
}
~~~

`sonic/pyproject.toml` declares Python `>=3.10`, NumPy, pyzmq, and an `integration` extra containing MuJoCo. Do not add Torch, SciPy, OpenCV, a training framework, or a depth dependency. Add `/sonic/.venv/`, `/sonic/build/`, and `/sonic/runs/` to `.gitignore`.

- [ ] **Step 4: Run GREEN and verify the real pinned checkout read-only**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_external -v
test -d "$SONIC_GEAR_CHECKOUT"
SONIC_GEAR_CHECKOUT="$SONIC_GEAR_CHECKOUT" PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import os
from pathlib import Path
from mm_sonic.external import verify_gear_checkout
verified = verify_gear_checkout(
    Path(os.environ['SONIC_GEAR_CHECKOUT']),
    Path('sonic/configs/gear_sonic.lock.json'))
assert verified.gear_commit == '60de0df7ffedeef415fe58d435e92cc5b01ba3d9'
print(verified.known_good_reference)
PY
~~~

Expected: tests pass and the printed path ends in `walking_quip_360_R_002__A428`.

- [ ] **Step 5: Commit**

~~~bash
git add .gitignore sonic tests/python/test_sonic_external.py
git commit -m "build: scaffold isolated SONIC baseline"
~~~

---

### Task 2: Generate and certify the named 29-joint projection contract

**Files:**
- Create: `sonic/python/mm_sonic/joints.py`
- Create: `sonic/configs/g1_joint_contract.json`
- Create: `sonic/cpp/g1_joint_projection.h`
- Create: `sonic/cpp/g1_project_pose_cli.cpp`
- Create: `tests/python/test_sonic_joints.py`
- Create: `tests/cpp/test_g1_joint_projection.cpp`

**Interfaces:**
- The committed source order is the source MJCF hinge order: left leg six, right leg six, waist three, left arm seven, right arm seven.
- Each contract row stores `source_index`, source bone/parent, MJCF joint name, qpos address, normalized Holden-local hinge axis, static Holden-local parent-to-child quaternion, sign, zero offset, inclusive range, target name, and target index.
- For a local rotation `q_local`, extraction uses `delta = inverse(q_static) * q_local`, signed twist about the registered Holden-local axis, and reconstruction `q_static * q_twist`.
- Projection rejects a geodesic off-axis residual above `0.001` rad, a reconstructed local-rotation error above `0.001` rad, non-finite data, or values outside the registered range. Certification tightens round-trip angle and local-rotation errors to `0.0001` rad.

The C++ public API is:

~~~cpp
static constexpr int SonicG1JointCount = 29;

struct sonic_joint_contract_entry {
    int source_index = -1;
    int source_bone = -1;
    int source_parent = -1;
    int target_index = -1;
    vec3 axis_holden;
    quat static_local_holden;
    float sign = 0.0f;
    float zero_offset = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
    std::string source_joint;
    std::string target_joint;
};

struct sonic_projected_pose {
    float source_joint_position[SonicG1JointCount] = {};
    float source_joint_velocity[SonicG1JointCount] = {};
    float off_axis_residual[SonicG1JointCount] = {};
    vec3 physical_pelvis_position_holden;
    quat physical_pelvis_orientation_holden;
};

bool sonic_project_pose(
    sonic_projected_pose& out,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    slice1d<quat> local_rotations,
    slice1d<vec3> local_angular_velocities,
    slice1d<vec3> global_positions,
    slice1d<quat> global_rotations,
    char* error,
    int error_capacity);
~~~

The corresponding Python contract types are:

~~~python
@dataclass(frozen=True)
class JointRow:
    source_index: int
    source_bone: str
    source_parent: str
    source_joint: str
    qpos_address: int
    axis_holden: tuple[float, float, float]
    static_local_holden_wxyz: tuple[float, float, float, float]
    sign: float
    zero_offset: float
    lower: float
    upper: float
    target_name: str
    target_index: int

@dataclass(frozen=True)
class JointContract:
    source_mjcf_sha256: str
    target_order_source_sha256: str
    rows: tuple[JointRow, ...]
~~~

- [ ] **Step 1: Write RED generator, mapping, twist, and rejection tests**

Python tests load the source MJCF with MuJoCo, assert all 29 names exactly once, assert the pinned target permutation, generate the contract twice byte-for-byte, and sample zero, both limits inset by `1e-5`, midpoints, and 256 seeded random qpos vectors. For every sample, run MuJoCo FK, apply the existing Z-up-to-Holden conversion, recover local rotations, invoke the C++ projection CLI, and compare source qpos, reconstructed local quaternions, and target ordering. Load the pinned GEAR scene too and require the same 29 body-joint names, hinge types, axes, ranges, and zero semantics by name; extra hand joints are permitted but never enter the 29-column reference.

C++ tests independently require signed angles around all three axes, antipodal quaternion equivalence, angular-velocity projection, range boundaries, transactional unchanged output on failure, and rejection of a `0.00101` rad swing residual while accepting `0.00099` rad.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_joints -v
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp \
  -o /tmp/test_g1_joint_projection
~~~

Expected: Python import failure and C++ missing-header failure.

- [ ] **Step 3: Implement deterministic contract generation and projection**

The generator infers the Holden-local axis by perturbing each source qpos by a small positive angle and verifying the result against the MJCF axis; it does not infer mapping by positional coincidence. Normalize quaternion signs with the first nonzero component positive before JSON formatting, use sorted keys plus compact separators, and write a final newline.

The target reorder is always name driven:

~~~python
def reorder_source_to_target(
    values: np.ndarray,
    contract: JointContract,
) -> np.ndarray:
    source = np.asarray(values)
    if source.shape[-1] != 29:
        raise ContractError(f"expected 29 source joints, got {source.shape}")
    output = np.empty_like(source)
    for row in contract.rows:
        output[..., row.target_index] = source[..., row.source_index]
    return output
~~~

Generate and commit `sonic/configs/g1_joint_contract.json` from the registered source MJCF. The file records the source MJCF SHA-256 and external joint-order source SHA-256 so another G1 file cannot reuse it silently.

- [ ] **Step 4: Run focused GREEN and certification**

~~~bash
mkdir -p sonic/build
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  sonic/cpp/g1_project_pose_cli.cpp \
  -o sonic/build/g1_project_pose_cli
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_joint_projection.cpp \
  -o /tmp/test_g1_joint_projection
/tmp/test_g1_joint_projection
PYTHONPATH=sonic/python SONIC_PROJECT_CLI=sonic/build/g1_project_pose_cli \
  SONIC_SOURCE_MJCF=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_joints -v
~~~

Expected: maximum joint-angle and reconstructed-local-rotation errors are each at most `0.0001` rad, with exact name coverage.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/joints.py sonic/configs/g1_joint_contract.json \
  sonic/cpp/g1_joint_projection.h sonic/cpp/g1_project_pose_cli.cpp \
  tests/python/test_sonic_joints.py tests/cpp/test_g1_joint_projection.cpp
git commit -m "feat: certify G1 MM to SONIC joint projection"
~~~

---

### Task 3: Extract one renderer-free matcher step with strict visual-runtime behavioral parity

**Files:**
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Create: `sonic/cpp/g1_runtime.h`
- Create: `sonic/python/mm_sonic/runtime_parity.py`
- Create: `tests/cpp/test_g1_runtime.cpp`
- Create: `tests/python/test_sonic_runtime_parity.py`
- Create: `tests/fixtures/sonic/g1_runtime_flat_64.csv`

**Interfaces:**
- `g1_controller_state_clone` performs a deep, transactional copy; mutating or swapping the clone cannot alias any source array.
- `g1_runtime_step` owns ordinary MM only. The dormant LAFAN learned path remains in the visual controller but is unavailable to the server.
- Input is an already resolved world-space requested velocity and unit desired heading. For `direct` mode, all four command horizons use the latched request; for `route` mode, the existing deterministic route callbacks remain unchanged; for `visual` mode, the controller supplies its existing gamepad-derived callbacks.
- The kernel includes command publication, traversability, 31D query creation, search/transition, inertialization, simulation-root update, support observation/update, horizontal adjustment/clamping, and final support-retargeted FK. It stops before logging, foot-lock IK, camera, GUI, drawing, and window handling.
- Oracle parity is byte-exact for the header and every field except `incumbent_cost`, `selected_cost`, `selected_terrain_error`, and `continuation_cost`. Those four diagnostic binary32 costs may differ by at most 8 ULP because translation-unit extraction changes GCC `-O3 -ffast-math` scheduling. Row count/order, all frame/range/selection decisions, the complete 31D query bits, and every physical/root/support state remain exact.

Use these public structures:

~~~cpp
enum g1_runtime_command_mode {
    G1RuntimeVisual,
    G1RuntimeRoute,
    G1RuntimeDirect
};

struct g1_runtime_step_request {
    g1_runtime_command_mode mode = G1RuntimeDirect;
    vec3 requested_velocity_holden;
    quat desired_heading_holden;
    bool matching_enabled = true;
};

struct g1_runtime_step_result {
    int query_database_frame = -1;
    int selected_database_frame = -1;
    int query_range = -1;
    float query[31] = {};
    terrain_centerline_snapshot terrain;
    traversability_diagnostics traversal;
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic projected;
};

struct g1_runtime_config {
    float dt = 0.04f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float inertialize_blending_halflife = 0.10f;
    float desired_velocity_change_threshold = 50.0f;
    float desired_rotation_change_threshold = 50.0f;
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    float adjustment_position_halflife = 0.10f;
    float adjustment_rotation_halflife = 0.20f;
    float adjustment_position_max_ratio = 0.50f;
    float adjustment_rotation_max_ratio = 0.50f;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    bool synchronization_enabled = false;
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    bool clamping_enabled = true;
};
~~~

- [ ] **Step 1: Capture the pre-refactor 64-frame oracle**

Before editing `controller.cpp`, compile the exact branch and run the deterministic flat path against the user's terrain pack read-only:

~~~bash
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src \
  -I/home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_sonic_oracle -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
DISPLAY=:1 G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  MM_TEST_MODE=flat MM_TEST_FRAMES=64 \
  MM_LOG=/tmp/g1_runtime_flat_64.csv \
  /tmp/controller_sonic_oracle
mkdir -p tests/fixtures/sonic
cp /tmp/g1_runtime_flat_64.csv tests/fixtures/sonic/g1_runtime_flat_64.csv
~~~

Expected: 64 deterministic data rows plus the existing header. Confirm the input terrain directory remains unchanged with `git status` in the primary workspace.

- [ ] **Step 2: Add RED clone, direct-command, boundary, and source-parity tests**

Extend the controller-state test to fill every scalar and array, deep clone, mutate the clone, and prove value and pointer independence. Add runtime tests for invalid unit heading, invalid `dt`, 31-value finite queries, direct-horizon latching, one-step frame advance, support-retargeted global pelvis observation, and unchanged state/result on failure.

Add a source-ownership test requiring `controller.cpp` to call `g1_runtime_step` exactly once per update and forbidding `database_search`, `inertialize_pose_update`, `support_frame_update`, and `terrain_centerline_snapshot_compute_v2` in the render adapter after extraction.

Add Python parity-checker tests requiring exact headers and 64-row ownership, exact equality for all non-cost fields, acceptance at exactly 8 binary32 ULP for only the four registered diagnostic costs, rejection at 9 ULP, and rejection of malformed, non-finite, missing, extra, or reordered data.

- [ ] **Step 3: Run RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state
/tmp/test_g1_controller_state
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/test_g1_runtime
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_runtime_parity -v
~~~

Expected: clone/runtime API failures and source-ownership failure.

- [ ] **Step 4: Implement the deep clone and mechanically extract the existing ordinary-MM block**

The clone publishes only after a full implicit deep copy of all `array1d` members succeeds:

~~~cpp
static inline bool g1_controller_state_clone(
    g1_controller_state& out,
    const g1_controller_state& source,
    char* error,
    int capacity)
{
    if (&out == &source) {
        return true;
    }
    g1_controller_state candidate(source);
    if (!g1_controller_state_is_valid_shape(candidate)) {
        return scene_error(
            error, capacity, "controller clone: invalid state shape");
    }
    g1_controller_state_swap(out, candidate);
    return true;
}
~~~

Move, without algebraic cleanup or constant changes, the existing ordinary matcher statements from per-frame diagnostic reset through final support-retargeted FK into `sonic/cpp/g1_runtime.h`. Move any renderer-free helper currently private to `controller.cpp` beside that kernel. Keep gamepad reads, route selection UI, deterministic log construction, IK/rendering, and error-to-window shutdown in `controller.cpp`; the adapter builds a request and maps the checked error back to its existing controlled shutdown.

For direct server commands, use the existing transactional prediction builder with `route_mode=true`, a constant four-horizon velocity callback, and an active heading override. This preserves the same rotation/position smoothing without calling gamepad code.

- [ ] **Step 5: Run unit GREEN, rebuild the visual controller, and enforce strict oracle parity**

~~~bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state
/tmp/test_g1_controller_state
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/test_g1_runtime
/tmp/test_g1_runtime
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_runtime_parity -v
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src \
  -I/home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_sonic_refactored -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
DISPLAY=:1 G1_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  MM_TEST_MODE=flat MM_TEST_FRAMES=64 \
  MM_LOG=/tmp/g1_runtime_flat_64_after.csv \
  /tmp/controller_sonic_refactored
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m mm_sonic.runtime_parity \
  tests/fixtures/sonic/g1_runtime_flat_64.csv \
  /tmp/g1_runtime_flat_64_after.csv
~~~

Expected: all tests pass; every non-cost field is exact; and the checker reports no more than 8 ULP in only the four registered diagnostic cost columns. Any decision, query-bit, physical/root/support-state drift, any unregistered column drift, or a registered cost drift above 8 ULP blocks server work.

- [ ] **Step 6: Commit**

~~~bash
git add g1_controller_state.h controller.cpp sonic/cpp/g1_runtime.h \
  sonic/python/mm_sonic/runtime_parity.py tests/cpp/test_g1_controller_state.cpp \
  tests/cpp/test_g1_runtime.cpp tests/python/test_sonic_runtime_parity.py \
  tests/fixtures/sonic/g1_runtime_flat_64.csv \
  docs/superpowers/plans/2026-07-15-g1-sonic-scene-aware-baseline.md \
  docs/superpowers/specs/2026-07-15-g1-sonic-scene-aware-baseline-design.md
git commit -m "refactor: expose renderer-free G1 matcher step"
~~~

---

### Task 4: Implement the transactional JSONL chunk protocol and real MM server

**Files:**
- Create: `sonic/cpp/mm_chunk_protocol.h`
- Create: `sonic/cpp/mm_chunk_json.h`
- Create: `sonic/cpp/mm_chunk_server.cpp`
- Create: `sonic/schemas/mm_chunk_v1.schema.json`
- Create: `tests/cpp/test_mm_chunk_protocol.cpp`
- Create: `tests/python/test_mm_chunk_server.py`
- Modify: `sonic/cpp/Makefile`

**Interfaces:**
- stdin accepts exactly one JSON object per line; stdout emits exactly one JSON response per line; stderr owns all human diagnostics.
- Every request uses exact keys, beginning with `{"v":1,"op":"hello","request_id":"r0"}`. Unknown/duplicate keys, unknown versions, invalid UTF-8, JSON constants, non-integral integers, non-finite or non-binary32 numeric inputs, and trailing data are errors.
- Legal sequence is `hello`, `reset`, repeated `generate` then `commit|abort`, and `close`. A reset is illegal with an outstanding candidate. A second generate is illegal. Commit/abort IDs must match.
- `generate` takes one latched direct command and exactly 10 source intervals. It observes active boundary zero, advances a deep clone 10 times, observes after every step, projects all 11 states, and publishes a candidate without mutating active state.

Requests are exact:

~~~json
{"v":1,"op":"hello","request_id":"r0"}
{"v":1,"op":"reset","request_id":"r1","session_id":"s1","scene_id":"grail-curb-low","route_id":"curb-forward","terrain_weight":4.0}
{"v":1,"op":"generate","request_id":"r2","session_id":"s1","candidate_id":"c000000","predecessor_id":null,"source_intervals":10,"requested_velocity_holden":[0.0,0.0,0.5],"desired_heading_holden_wxyz":[1.0,0.0,0.0,0.0]}
{"v":1,"op":"commit","request_id":"r3","session_id":"s1","candidate_id":"c000000"}
{"v":1,"op":"abort","request_id":"r4","session_id":"s1","candidate_id":"c000001"}
{"v":1,"op":"close","request_id":"r5"}
~~~

Every response has exact envelope keys:

~~~json
{"v":1,"ok":true,"op":"commit","request_id":"r3","data":{"session_id":"s1","candidate_id":"c000000","active_candidate_id":"c000000"}}
~~~

or:

~~~json
{"v":1,"ok":false,"op":"generate","request_id":"r2","error":{"code":"candidate_outstanding","message":"candidate c000000 must be committed or aborted"}}
~~~

- [ ] **Step 1: Write RED protocol-state and subprocess tests**

C++ tests use a fake state adapter to prove every legal/illegal transition, active-state hashes before and after generate/abort/failure, commit swap, close behavior, deterministic regenerate-after-abort, and exactly 11 boundaries for 10 steps.

Python subprocess tests require stdout JSON purity, stderr isolation, schema identity, one candidate only, abort/regenerate equality, commit successor continuity, and clean EOF/close. Mark the real-artifact test with `SONIC_TERRAIN_DIR`; unit protocol tests must not require the user's terrain pack.

- [ ] **Step 2: Run RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp \
  -o /tmp/test_mm_chunk_protocol
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_mm_chunk_server -v
~~~

Expected: missing protocol/header and server-binary failures.

- [ ] **Step 3: Implement strict parse/serialize, the protocol engine, and server adapter**

Reuse `json_runtime.h` for parsing, but add explicit duplicate-member and exact-key checks. Serialize finite C++ floats with enough decimal digits to round-trip binary32. Set the C locale, disable stdout buffering, and never print artifact-loader diagnostics to stdout.

The candidate owner is explicit:

~~~cpp
struct mm_chunk_session {
    bool reset = false;
    std::string session_id;
    std::string active_candidate_id;
    g1_controller_state active;
    bool candidate_ready = false;
    std::string candidate_id;
    g1_controller_state candidate;
};

static inline bool mm_chunk_commit(
    mm_chunk_session& session,
    const std::string& candidate_id,
    char* error,
    int capacity)
{
    if (!session.candidate_ready || session.candidate_id != candidate_id) {
        return scene_error(
            error, capacity, "commit candidate does not match");
    }
    g1_controller_state_swap(session.active, session.candidate);
    session.active_candidate_id = session.candidate_id;
    session.candidate_id.clear();
    session.candidate_ready = false;
    return true;
}
~~~

`hello` reports protocol version, source/target joint names, skeleton signature, source rate, supported interval count, build commit, joint-contract hash, motion-manifest hash, database hash, terrain-feature/support hashes, scene-index hash, and coordinate signature. `reset` resolves the scene/route and rebuilds matching features at the requested weight before resetting state. A failed rebuild/reset leaves the previous active session unchanged.

- [ ] **Step 4: Build and run protocol GREEN**

~~~bash
make -C sonic/cpp clean all
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp \
  -o /tmp/test_mm_chunk_protocol
/tmp/test_mm_chunk_protocol
PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_mm_chunk_server -v
SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_mm_chunk_server.RealArtifactChunkServerTest -v
~~~

Expected: all protocol tests pass; the real test emits 11 projected source boundaries and proves abort/regenerate equality.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/cpp/mm_chunk_protocol.h sonic/cpp/mm_chunk_json.h \
  sonic/cpp/mm_chunk_server.cpp sonic/cpp/Makefile \
  sonic/schemas/mm_chunk_v1.schema.json \
  tests/cpp/test_mm_chunk_protocol.cpp tests/python/test_mm_chunk_server.py
git commit -m "feat: add transactional terrain MM chunk server"
~~~

---

### Task 5: Validate source chunks, map named joints, and convert the coordinate basis

**Files:**
- Create: `sonic/python/mm_sonic/schema.py`
- Create: `sonic/python/mm_sonic/transform.py`
- Create: `sonic/schemas/target_chunk_v1.schema.json`
- Create: `tests/python/test_sonic_schema.py`
- Create: `tests/python/test_sonic_transform.py`
- Modify: `tests/python/test_mm_chunk_server.py`

**Interfaces:**
- JSON loading uses `object_pairs_hook` to reject duplicate keys and `parse_constant` to reject NaN/Infinity. Runtime validation enforces exact key sets instead of relying on a permissive schema library.
- All C++ source floats must round-trip to finite binary32. Arrays are copied into owned C-contiguous NumPy arrays and made read-only after validation.
- Source boundaries have length 11; step diagnostics have length 10; source joints have length 29.
- The reset response includes one `InitialBoundary` with the same projected pose fields as source boundary zero. It is frame zero for the target timeline and is not an accepted chunk.
- Joint mapping is read only from the checked joint contract. Positional fallback is forbidden.
- Vector conversion is `[x, y, z] -> [x, -z, y]`. Quaternion conversion is `q_m = b * q_h * inverse(b)` with `b = [sqrt(0.5), sqrt(0.5), 0, 0]` in `wxyz`.

The immutable data model is:

~~~python
@dataclass(frozen=True)
class InitialBoundary:
    session_id: str
    source_joint_names: tuple[str, ...]
    joint_position_source: np.ndarray
    joint_velocity_source: np.ndarray
    physical_pelvis_position_holden: np.ndarray
    physical_pelvis_orientation_holden: np.ndarray
    virtual_root_position_holden: np.ndarray
    virtual_root_orientation_holden: np.ndarray

@dataclass(frozen=True)
class SourceChunk:
    session_id: str
    candidate_id: str
    predecessor_id: str | None
    source_rate_hz: int
    source_intervals: int
    timestamps_s: np.ndarray
    source_joint_names: tuple[str, ...]
    target_joint_names: tuple[str, ...]
    joint_position_source: np.ndarray
    joint_velocity_source: np.ndarray
    physical_pelvis_position_holden: np.ndarray
    physical_pelvis_orientation_holden: np.ndarray
    virtual_root_position_holden: np.ndarray
    virtual_root_orientation_holden: np.ndarray
    selected_database_frame: np.ndarray
    searched: np.ndarray
    transitioned: np.ndarray
    terrain_cost: np.ndarray
    terrain_values: np.ndarray
    terrain_points_holden: np.ndarray
    support_height: np.ndarray
    support_target: np.ndarray
    scene: Mapping[str, object]
    command: Mapping[str, object]
    artifacts: Mapping[str, str]
~~~

- [ ] **Step 1: Write RED exact-schema and basis tests**

Require rejection of duplicate keys at every nesting level, extra/missing keys, wrong schema/version, wrong lengths/shapes, duplicate/missing joints, non-finite values, non-unit quaternions, non-monotonic timestamps, timestamps not equal to `i / 25`, and input arrays that mutate after construction.

Basis tests cover all three unit axes, identity, ±90-degree rotations around every axis, an arbitrary known pelvis pose, quaternion antipodes, zero norm, and round-trip `Holden -> MuJoCo -> Holden`. Transform the registered terrain OBJ bounds independently and compare them later with the scene adapter.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_transform -v
~~~

Expected: missing module failures.

- [ ] **Step 3: Implement fail-closed parsing and proper basis conversion**

Use this exact duplicate-key loader:

~~~python
def _object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output

def loads_exact(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_object_no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ContractError(f"invalid JSON constant: {value}")),
    )
~~~

Normalize quaternions in float64, reject norm below `1e-12`, force adjacent dot products nonnegative, convert basis, renormalize, then cast once to float32. Joint position/velocity mapping casts the source values to float32 before reordering so CSV and ZMQ share exact bits.

- [ ] **Step 4: Run GREEN plus a real server decode**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_transform -v
SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_mm_chunk_server.RealArtifactChunkServerTest -v
~~~

Expected: exact schemas and all basis fixtures pass; the real chunk decodes without name or quaternion repair.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/schema.py sonic/python/mm_sonic/transform.py \
  sonic/schemas/target_chunk_v1.schema.json \
  tests/python/test_sonic_schema.py tests/python/test_sonic_transform.py \
  tests/python/test_mm_chunk_server.py
git commit -m "feat: validate and transform MM source chunks"
~~~

---

### Task 6: Resample 11 source boundaries into one transactional 50 Hz timeline

**Files:**
- Create: `sonic/python/mm_sonic/resample.py`
- Create: `sonic/python/mm_sonic/timeline.py`
- Create: `tests/python/test_sonic_resample.py`
- Create: `tests/python/test_sonic_timeline.py`

**Interfaces:**
- `resample_source_chunk` first maps source joints by name and converts physical/virtual global transforms to MuJoCo basis.
- Every 25 Hz interval produces its midpoint and right endpoint. The candidate's left endpoint is checked but excluded, so exactly 20 target rows result.
- Joint source endpoints at target even samples are bit-equal float32 after mapping. Hermite calculations use float64 and cast once to float32 after range validation.
- Diagnostic pelvis/root positions interpolate linearly. Diagnostic virtual-root and physical-pelvis orientations use shortest-path SLERP.
- `TargetTimeline.prepare` is non-mutating. It verifies the first source boundary against the last accepted boundary to `1e-6` for scalar/vector values and `1e-6` rad for orientations, assigns contiguous indices, and returns a prepared target candidate.
- `commit` advances the boundary/index only for the matching candidate; `abort` discards it. No neutral or repeated chunk exists.

Use the exact Hermite equations:

~~~python
def hermite_pair(
    q0: np.ndarray,
    v0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    dt: float,
    u: float,
) -> tuple[np.ndarray, np.ndarray]:
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    q = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u
    v = (dh00 * q0 + dh01 * q1) / dt + dh10 * v0 + dh11 * v1
    return q, v
~~~

- [ ] **Step 1: Write RED analytic interpolation and timeline tests**

Test constant, linear, and cubic trajectories; endpoint positions and derivatives; finite-difference agreement only as a secondary check; joint-limit overshoot rejection; antipodal/near-identical SLERP; exact source endpoint recovery; 11-to-20 shape; initial frame zero; first chunk indices 1–20; second chunk 21–40; seam mismatch at and below tolerance; prepare/abort/regenerate; wrong predecessor; wrong candidate commit; and no timeline mutation on any failure.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_resample \
  tests.python.test_sonic_timeline -v
~~~

Expected: missing resampler/timeline modules.

- [ ] **Step 3: Implement target chunks and boundary ownership**

The prepared target has exact immutable fields:

~~~python
@dataclass(frozen=True)
class TargetChunk:
    schema: str
    session_id: str
    accepted_chunk_id: str
    source_candidate_id: str
    frame_index: np.ndarray
    timestamps_s: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    physical_pelvis_position: np.ndarray
    virtual_root_position: np.ndarray
    virtual_root_quat_w: np.ndarray
    scene: Mapping[str, object]
    command: Mapping[str, object]
    hashes: Mapping[str, str]

@dataclass(frozen=True)
class CanonicalTargetBuffer:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    frame_index: np.ndarray

    @property
    def count(self) -> int:
        return int(self.frame_index.shape[0])
~~~

Target timestamps are exactly `frame_index / 50.0`. Hash the canonical target as the ordered concatenation of little-endian C-contiguous `joint_position<f4`, `joint_velocity<f4`, `body_quat_w<f4`, and `frame_index<i8` bytes. Hash diagnostic arrays separately so they cannot change the policy buffer hash.

- [ ] **Step 4: Run GREEN and sanitizer-like numeric fuzzing**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_resample \
  tests.python.test_sonic_timeline -v
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import numpy as np
from mm_sonic.resample import hermite_pair
rng = np.random.default_rng(20260715)
for _ in range(10000):
    q0, q1, v0, v1 = rng.normal(size=(4, 29))
    for u in (0.0, 0.5, 1.0):
        q, v = hermite_pair(q0, v0, q1, v1, 0.04, u)
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(v))
print("10000 finite Hermite fixtures")
PY
~~~

Expected: unit tests pass and the fuzz command prints the fixture count.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/resample.py \
  sonic/python/mm_sonic/timeline.py \
  tests/python/test_sonic_resample.py tests/python/test_sonic_timeline.py
git commit -m "feat: add exact 25 to 50 Hz SONIC timeline"
~~~

---

### Task 7: Write immutable run bundles and official SONIC reference directories

**Files:**
- Create: `sonic/python/mm_sonic/artifacts.py`
- Create: `sonic/python/mm_sonic/reference.py`
- Create: `tests/python/test_sonic_artifacts.py`
- Create: `tests/python/test_sonic_reference.py`
- Modify: `sonic/schemas/run_manifest_v1.schema.json`

**Interfaces:**
- `RunBundle.create(root, experiment_id, run_id)` uses exclusive directory creation; an existing ID is a hard failure. Inputs are opened read-only and outputs are confined beneath that run.
- A run starts with `status=running`, records every candidate/abort/accept and timing sample incrementally, and finalizes once as `complete`, `failed`, or `not_run`. Finalization writes hashes, fsyncs, and makes evidence files read-only.
- One full canonical session buffer owns both sinks. Official reference output includes frame zero and all accepted target rows.
- CSVs have exact headers, decimal strings that round-trip float32, and a final newline. Decoding every CSV back to float32 must be bit-equal to the canonical arrays.
- `body_pos.csv` is all zeros with shape `[N,3]`. `body_quat.csv` is root-only `[N,4]`. `metadata.txt` contains body indexes `[0]`. Actual physical pelvis and virtual-root positions go only to `mm_root_diagnostic.csv`.

The required reference files are:

~~~text
reference/mm_sonic/
  joint_pos.csv
  joint_vel.csv
  body_quat.csv
  body_pos.csv
  metadata.txt
  info.txt
mm_root_diagnostic.csv
canonical_target.npz
~~~

- [ ] **Step 1: Write RED immutability and CSV parity tests**

Require exclusive creation, path traversal/symlink escape rejection, no writes into an input tree, lifecycle transitions, idempotence rejection after finalization, SHA-256 inventory, partial-failure evidence, exact headers/shapes/frame counts, metadata parsing, zero body positions, float32 bit parity, and rejection if any CSV is edited after writing.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_artifacts \
  tests.python.test_sonic_reference -v
~~~

Expected: missing modules.

- [ ] **Step 3: Implement the artifact writer and official reference sink**

Use a stable formatter:

~~~python
def format_f32(value: np.float32) -> str:
    scalar = np.float32(value)
    if not np.isfinite(scalar):
        raise ContractError("cannot serialize non-finite float32")
    return format(float(scalar), ".9g")
~~~

`info.txt` records shapes, rates, source/target hashes, scene/route, and states explicitly that body position is zero and untracked. It is diagnostic only; no external parser depends on it. The manifest includes repository commits, dirty flags, policy/encoder/config/model/MJCF hashes, motion/terrain/joint-map/scene hashes, command script, perturbation, coordinate transform, process command lines, and outcome.

- [ ] **Step 4: Run GREEN and visualize/load with the pinned reader when available**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_artifacts \
  tests.python.test_sonic_reference -v
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.cli \
  reference --self-test-output /tmp/mm-sonic-reference-self-test
~~~

Expected: unit tests pass and the CLI writes, decodes, hashes, then removes only its own temporary self-test directory.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/artifacts.py \
  sonic/python/mm_sonic/reference.py \
  sonic/schemas/run_manifest_v1.schema.json \
  tests/python/test_sonic_artifacts.py tests/python/test_sonic_reference.py
git commit -m "feat: export immutable SONIC references"
~~~

---

### Task 8: Encode and publish exact GEAR ZMQ protocol v1 messages

**Files:**
- Create: `sonic/python/mm_sonic/zmq_v1.py`
- Create: `tests/python/test_sonic_zmq_v1.py`
- Modify: `sonic/pyproject.toml`
- Modify: `sonic/python/mm_sonic/artifacts.py`

**Interfaces:**
- Codec construction and decoding are pure and testable without pyzmq. Socket import occurs only when `PosePublisher` is instantiated.
- Header JSON is compact UTF-8 and shorter than 1,280 bytes, followed by NUL bytes to exactly 1,280. No delimiter appears between topic, header, and payload.
- Payload order is exactly target joint positions, target joint velocities, pelvis quaternion, int64 frame indices, and one uint8 `catch_up`.
- Arrays are little-endian, C-contiguous, and bit-identical to the canonical target buffer. `PosePublisher.send` archives the exact message bytes and SHA-256 before returning.
- Production uses PUB bind and GEAR SUB connect. `ZMQ_CONFLATE` remains off.

Use this exact header builder:

~~~python
def pose_header(count: int) -> dict[str, object]:
    return {
        "v": 1,
        "endian": "le",
        "count": count,
        "fields": [
            {"name": "joint_pos", "dtype": "f32", "shape": [count, 29]},
            {"name": "joint_vel", "dtype": "f32", "shape": [count, 29]},
            {"name": "body_quat_w", "dtype": "f32", "shape": [count, 4]},
            {"name": "frame_index", "dtype": "i64", "shape": [count]},
            {"name": "catch_up", "dtype": "u8", "shape": [1]},
        ],
    }
~~~

- [ ] **Step 1: Write RED golden-byte and live-loopback tests**

Construct a two-frame fixture with distinctive endian-sensitive values and assert the exact header object, topic offset, 1,280-byte padding, payload offsets/length, dtype, shape, and full decoded bit parity. Reject wrong shape, dtype, index gaps, non-finite values, oversized header, wrong topic, extra payload bytes, and `catch_up != 0` in baseline mode.

A live test binds a random localhost port, establishes SUB/PUB readiness with a separate test-only synchronization socket, sends one initial frame and one 20-frame chunk, and verifies both messages. Do not use sleep as the correctness condition.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_zmq_v1 -v
~~~

Expected: missing codec module.

- [ ] **Step 3: Implement pure codec, publisher lifecycle, and archived parity**

The encoder is exactly:

~~~python
def encode_pose_v1(buffer: CanonicalTargetBuffer) -> bytes:
    header_json = json.dumps(
        pose_header(buffer.count),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(header_json) > 1280:
        raise ContractError("ZMQ v1 header exceeds 1280 bytes")
    header = header_json + bytes(1280 - len(header_json))
    payload = b"".join((
        buffer.joint_position.astype("<f4", copy=False).tobytes(order="C"),
        buffer.joint_velocity.astype("<f4", copy=False).tobytes(order="C"),
        buffer.body_quat_w.astype("<f4", copy=False).tobytes(order="C"),
        buffer.frame_index.astype("<i8", copy=False).tobytes(order="C"),
        bytes((0,)),
    ))
    return b"pose" + header + payload
~~~

The decoder independently parses offsets from the header rather than slicing with encoder constants. Compare decoded arrays to the canonical buffer before every production send and store files such as `transmitted/000001-000020.bin` plus their digests in the run bundle.

- [ ] **Step 4: Install only pyzmq into an isolated environment and run GREEN**

~~~bash
python -m venv sonic/.venv
sonic/.venv/bin/pip install -e 'sonic[integration]'
PYTHONPATH=sonic/python sonic/.venv/bin/python \
  -m unittest tests.python.test_sonic_zmq_v1 -v
~~~

Expected: golden bytes and live loopback pass.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/pyproject.toml sonic/python/mm_sonic/zmq_v1.py \
  sonic/python/mm_sonic/artifacts.py tests/python/test_sonic_zmq_v1.py
git commit -m "feat: publish canonical SONIC ZMQ v1 chunks"
~~~

---

### Task 9: Register one shared flat scene, convert terrain scenes, and kinematically replay references

**Files:**
- Create: `sonic/configs/scene_registry.json`
- Create: `sonic/cpp/sonic_flat_scene.h`
- Create: `sonic/python/mm_sonic/scene.py`
- Create: `tests/cpp/test_sonic_flat_scene.cpp`
- Create: `tests/python/test_sonic_scene.py`
- Modify: `sonic/cpp/mm_chunk_server.cpp`
- Modify: `sonic/schemas/run_manifest_v1.schema.json`

**Interfaces:**
- `sonic-flat-baseline` is the only new scene. It is one committed analytic definition consumed by both MM and MuJoCo: zero height, certified walkability, bounds `[-10,10]` on both horizontal axes, spawn at the origin, yaw zero, and no obstacle. It is not added to or written into the user's terrain artifact directory.
- Existing terrain scenes are loaded from their authenticated `scene.json`, `terrain.bin`, and `terrain.obj`. The adapter never estimates a surface from depth or from another mesh.
- OBJ vertices and normals use the exact Holden-to-MuJoCo basis. Face indices/material assignments are retained, and source/transformed hashes plus bounds are recorded.
- The run-local GEAR scene rewrites the official scene's relative robot include to an absolute read-only include, removes only the official infinite floor for non-flat terrain, and adds one named `mm_terrain` mesh geom. The flat baseline keeps one named plane geom.
- Kinematic replay sets free-root position/quaternion from the diagnostic physical pelvis, maps the 29 target joints into the pinned GEAR MJCF qpos addresses by name, retains that model's default hand qpos, calls `mj_forward` and `mj_collision`, and records forbidden penetration. It does not use zero-valued SONIC `body_pos.csv` or assume that source/target qpos addresses match.
- Allowed foot geoms are descendants of `left_ankle_roll_link` and `right_ankle_roll_link`. Forbidden groups are pelvis; left/right knee; waist yaw/roll and torso; and every wrist/hand/finger body. Each group must resolve nonempty and be disjoint from allowed feet.

The flat registry entry is exact:

~~~json
{
  "schema": "mm-sonic-scene-registry/v1",
  "scenes": {
    "sonic-flat-baseline": {
      "kind": "analytic-flat",
      "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
      "bounds_xz": [-10.0, -10.0, 10.0, 10.0],
      "spawn_position_holden": [0.0, 0.0, 0.0],
      "spawn_yaw_holden": 0.0,
      "height_m": 0.0,
      "walkability_class": 1
    }
  }
}
~~~

- [ ] **Step 1: Write RED flat-scene, OBJ, overlay, geom, and replay tests**

C++ tests require zero height everywhere in bounds, certified walkability, exact spawn, finite centerline queries through the full 12-second flat path, and rejection outside bounds.

Python tests use a small asymmetric OBJ with vertices, normals, texture indices, and triangular faces; verify transformed vertices/bounds/normals, deterministic output bytes, source-hash rejection, exact generated MJCF loading, one terrain geom, official robot include identity, no non-flat floor, and complete contact-group resolution. Kinematic fixtures cover no contact, allowed foot contact, exactly 0.005 m forbidden penetration, and 0.00501 m failure.

- [ ] **Step 2: Run RED**

~~~bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_sonic_flat_scene.cpp \
  -o /tmp/test_sonic_flat_scene
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_scene -v
~~~

Expected: missing flat-scene and Python scene modules.

- [ ] **Step 3: Implement one authenticated scene path for MM and MuJoCo**

The scene record returned to the coordinator is:

~~~python
@dataclass(frozen=True)
class RegisteredScene:
    scene_id: str
    route_id: str | None
    source_kind: str
    source_mesh: Path | None
    source_heightfield: Path | None
    source_hashes: Mapping[str, str]
    coordinate_source: str
    coordinate_target: str
    transform_matrix: np.ndarray
    transformed_obj: Path | None
    gear_scene_xml: Path
    output_hashes: Mapping[str, str]
    allowed_foot_geoms: tuple[int, ...]
    forbidden_geom_groups: Mapping[str, tuple[int, ...]]
~~~

For the flat scene, the C++ server resolves the committed registry entry directly and constructs `scene_pack` arrays in memory. For terrain scenes, it continues to use `scene_pack_load`. Python uses the same registry ID and authenticated scene metadata; a mismatch between MM hello/reset identity and the generated overlay is a hard failure.

- [ ] **Step 4: Run GREEN on flat and all three registered terrain classes**

~~~bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_sonic_flat_scene.cpp \
  -o /tmp/test_sonic_flat_scene
/tmp/test_sonic_flat_scene
PYTHONPATH=sonic/python \
  SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  SONIC_GEAR_CHECKOUT="$SONIC_GEAR_CHECKOUT" \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_scene -v
~~~

Expected: the flat definition and curb/ramp/stairs meshes load, bounds convert exactly, and every contact group resolves.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/configs/scene_registry.json sonic/cpp/sonic_flat_scene.h \
  sonic/python/mm_sonic/scene.py sonic/cpp/mm_chunk_server.cpp \
  sonic/schemas/run_manifest_v1.schema.json \
  tests/cpp/test_sonic_flat_scene.cpp tests/python/test_sonic_scene.py
git commit -m "feat: share registered scenes with SONIC MuJoCo"
~~~

---

### Task 10: Add a step-counted MuJoCo runner and unmodified GEAR process gate

**Files:**
- Create: `sonic/python/mm_sonic/process.py`
- Create: `sonic/python/mm_sonic/gated_sim.py`
- Create: `tests/python/test_sonic_process.py`
- Create: `tests/python/test_sonic_gated_sim.py`

**Interfaces:**
- `GatedSimulatorClient` launches `gated_sim.py` as a JSONL child. The child imports `BaseSimulator` and configuration loaders from the pinned external checkout, overrides only `ROBOT_SCENE` with the run-local absolute XML, and never modifies external source.
- The child is idle after `reset`. `advance` accepts an integer step count, calls the external `sim_env.sim_step()` exactly that many times with its normal real-time cadence, logs contacts every physics step and state at 50 Hz, then replies with exact starting/ending MuJoCo time.
- Reset initializes the free root and 29 joints from target frame zero. Scored perturbations alter only physical horizontal root translation/yaw; the MM initial boundary is unchanged.
- `GearProcess` launches the official deployment under a PTY and a new process group, passes `--input-type zmq`, `--target-motion-logfile`, `--logs-dir`, `--enable-csv-logs`, `--zmq-conflate` absent, and `--zmq-verbose`. It writes `]` and newline to start control and enter stream mode, waits for official readiness markers, and archives stdout/stderr.
- `SimulationPolicyGate.pause()` SIGSTOPs the whole GEAR process group and verifies stopped state. `advance(0.4)` first requires `0.4 / SIMULATE_DT` to be an exact positive integer within `1e-12`, SIGCONTs GEAR, asks the simulator for exactly that many steps, then SIGSTOPs GEAR immediately after the simulator reply. The reply must advance MuJoCo time by 0.4 seconds within `1e-12`; MuJoCo time, not wall time, is authoritative.
- Watchdogs detect child death and allow operator cancellation. Chunk generation duration has no deadline and never fails the scientific test for being slow.

The gated simulator protocol is exact:

~~~json
{"v":1,"op":"hello","request_id":"g0"}
{"v":1,"op":"reset","request_id":"g1","scene_xml":"/tmp/run/scene/gear_scene.xml","initial_qpos":[0.0],"lateral_offset_m":0.0,"yaw_offset_rad":0.0,"log_dir":"/tmp/run/sim"}
{"v":1,"op":"advance","request_id":"g2","steps":200}
{"v":1,"op":"snapshot","request_id":"g3"}
{"v":1,"op":"close","request_id":"g4"}
~~~

`initial_qpos` in production has exactly `nq` values; the one-value example is intentionally invalid and is used by a RED test.

- [ ] **Step 1: Write RED process-group, PTY, step-count, reset, and death tests**

Use fake Python children, never the real policy, to prove PTY key delivery, separate process groups, STOP/CONT ordering, stderr capture, child-death detection, cleanup escalation, and no orphan. Use a fake simulator backend to prove no step while paused, exact step count, exact sim-time delta, 50 Hz sample count, contact logging, physical-only perturbation, and invalid reset rejection.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_process \
  tests.python.test_sonic_gated_sim -v
~~~

Expected: missing process and gated simulator modules.

- [ ] **Step 3: Implement explicit process lifecycles and injection-friendly simulator backend**

Use interfaces so CPU tests do not import Unitree or start DDS:

~~~python
class SimulatorBackend(Protocol):
    @property
    def model(self) -> object:
        raise NotImplementedError
    @property
    def data(self) -> object:
        raise NotImplementedError
    @property
    def sim_dt(self) -> float:
        raise NotImplementedError
    def reset_from_qpos(
        self,
        qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
    ) -> None:
        raise NotImplementedError
    def step(self) -> None:
        raise NotImplementedError
    def sample(self) -> Mapping[str, object]:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError

@dataclass(frozen=True)
class AdvanceResult:
    steps: int
    sim_time_start_s: float
    sim_time_end_s: float
    state_rows: int
    contact_rows: int

class SimulationPolicyGate:
    def pause(self) -> None:
        raise NotImplementedError
    def release_steps(self, steps: int) -> AdvanceResult:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
~~~

Production `ExternalGearBackend` resolves the external checkout from verified inputs, prepends it to `sys.path` only in the child, loads the official YAML, assigns the absolute scene path, and delegates to the official simulator. Process cleanup sends the official stop key first, then SIGTERM, then SIGKILL only after a recorded grace period.

- [ ] **Step 4: Run GREEN and an external import-only preflight**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_process \
  tests.python.test_sonic_gated_sim -v
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.gated_sim \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --import-preflight
~~~

Expected: unit tests pass; import preflight reports the pinned `BaseSimulator` and exits without starting DDS or stepping physics.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/process.py \
  sonic/python/mm_sonic/gated_sim.py \
  tests/python/test_sonic_process.py tests/python/test_sonic_gated_sim.py
git commit -m "feat: gate unmodified SONIC and MuJoCo stepping"
~~~

---

### Task 11: Enforce generate–validate–send–commit–advance coordination

**Files:**
- Create: `sonic/python/mm_sonic/coordinator.py`
- Create: `tests/python/test_sonic_coordinator.py`
- Modify: `sonic/python/mm_sonic/artifacts.py`
- Modify: `sonic/python/mm_sonic/timeline.py`
- Modify: `sonic/python/mm_sonic/process.py`

**Interfaces:**
- The coordinator is the only owner of session/chunk IDs and command sampling.
- Operator/script input is latched at chunk boundaries. New input arriving during generation is queued for the next chunk and cannot mutate the outstanding candidate.
- Readiness happens with physics paused: reset MM, validate frame zero, enter official ZMQ stream mode, publish frame zero until the official target log shows the same target row, record the scoring log offset, then stop repeating frame zero.
- Per chunk order is exactly: `pause -> generate -> source validate -> target prepare -> archive/enqueue -> ZMQ send -> MM commit -> timeline commit -> release exact sim steps -> pause`.
- Pre-commit failure writes a structured rejection, sends MM abort if the server is alive, and terminates with simulation paused. Post-commit/release failure terminates without rollback or successor generation.
- A successful local `send` is not called an acknowledgement. Final delivery audit remains mandatory.

The core transaction is shaped as:

~~~python
@dataclass(frozen=True)
class AcceptedChunk:
    target: TargetChunk
    advance: AdvanceResult
    timings_ns: Mapping[str, int]

def run_one_chunk(
    self,
    command: CommandSample,
) -> AcceptedChunk:
    self.gate.require_paused()
    self.timing.start_chunk()
    source = self.mm.generate(command)
    try:
        checked_source = self.validator.validate_source(source)
        prepared = self.timeline.prepare(checked_source)
        self.run.write_prepared(checked_source, prepared.target)
        self.publisher.send(prepared.target)
    except Exception as error:
        self.run.write_rejection(source.candidate_id, error)
        self.mm.abort(source.candidate_id)
        raise IntegrationFailure("pre_commit", error) from error
    self.mm.commit(source.candidate_id)
    accepted_target = self.timeline.commit(prepared)
    try:
        advance = self.gate.release_steps(self.steps_per_chunk)
    except Exception as error:
        raise IntegrationFailure("post_commit", error) from error
    accepted = AcceptedChunk(
        target=accepted_target,
        advance=advance,
        timings_ns=self.timing.finish_chunk(),
    )
    self.run.write_accepted(accepted)
    return accepted
~~~

- [ ] **Step 1: Write RED event-order and every-failure-boundary tests**

Fakes append every call to a shared event list. Assert exact success order and separate failures at generation, source validation, resampling, artifact enqueue, ZMQ encoding, ZMQ send, MM commit, timeline commit, process resume, and simulator advance. Check whether abort is sent, whether active MM/timeline state changes, whether simulation remains paused, and whether a successor is forbidden.

Also test queued operator input, one frame-zero readiness offset, slow generation with no timeout, dead-server detection, cleanup, and immutable terminal verdict.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_coordinator -v
~~~

Expected: missing coordinator module.

- [ ] **Step 3: Implement the coordinator as an explicit state machine**

Use states `created`, `preflight`, `ready_paused`, `candidate`, `committed`, `advancing`, `terminal`. Every public method checks state. Keep timing samples in integer monotonic nanoseconds for MM generation, projection/validation, resampling, enqueue, publication, and simulation advance. Record wall time separately and never use it as target timestamps.

The readiness parser compares the official target-motion log's 36 values per row: zero body position, four root quaternion values, and 29 joint values after applying the pinned logger permutation. It stores the byte offset after the confirmed frame-zero row; repeated handshake rows before that offset are not scored.

- [ ] **Step 4: Run GREEN and a real MM/file-only session without SONIC**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_coordinator -v
PYTHONPATH=sonic/python SONIC_MM_SERVER=sonic/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.cli \
  reference \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --scene sonic-flat-baseline \
  --terrain-weight 0 \
  --script flat-12s \
  --output-root sonic/runs
~~~

Expected: unit tests pass; file-only run completes 30 chunks, 601 total target frames, contiguous indices 0–600, and leaves no outstanding MM candidate.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/coordinator.py \
  sonic/python/mm_sonic/artifacts.py sonic/python/mm_sonic/timeline.py \
  sonic/python/mm_sonic/process.py tests/python/test_sonic_coordinator.py
git commit -m "feat: coordinate transactional MM to SONIC chunks"
~~~

---

### Task 12: Freeze command scripts and implement tracking/contact/verdict metrics

**Files:**
- Create: `sonic/python/mm_sonic/commands.py`
- Create: `sonic/python/mm_sonic/metrics.py`
- Create: `sonic/configs/experiments/flat.json`
- Create: `sonic/configs/experiments/terrain_baseline.json`
- Create: `sonic/schemas/trial_verdict_v1.schema.json`
- Create: `sonic/schemas/hypothesis_verdict_v1.schema.json`
- Create: `sonic/cpp/route_schedule_cli.cpp`
- Create: `tests/python/test_sonic_commands.py`
- Create: `tests/python/test_sonic_metrics.py`
- Modify: `sonic/cpp/Makefile`

**Interfaces:**
- The flat script has exactly 30 chunks:
  - chunks 0–4: stand at initial heading;
  - chunks 5–14: 0.5 m/s local-forward at initial heading;
  - chunks 15–24: heading advances by exactly 4.5 degrees per boundary to +45 degrees, with 0.5 m/s local-forward recomputed from that heading;
  - chunks 25–29: stand at the final heading.
- Route compilation ports `deterministic_route_command` exactly at 25 Hz, groups each ten-frame block, and uses the block's mean world velocity so integrated displacement is retained under command latching. Desired heading aligns with a nonzero block velocity; zero blocks retain the preceding heading. Pad only the final incomplete block with zero commands. Write and hash the complete command JSON before any aware/blind trial.
- The frozen command list, duration, MM initial state, and MM reference are identical across physical perturbations. Only terrain weight and physical initial perturbation vary as registered.
- Joint tracking is global RMSE over aligned 50 Hz target/actual 29-joint values. Pelvis orientation error is RMS geodesic angle `2*acos(abs(dot(q_target,q_actual)))`.
- Secondary outputs always include swing-foot scuff count, minimum foot clearance, horizontal path drift, joint and pelvis tracking traces, contact impulses, and policy execution timing when the pinned logger exposes it. They never override the registered success booleans.
- Timing summaries use deterministic nearest-rank p50/p95/p99.
- Dynamic trial failure thresholds and hypothesis aggregation match the approved design exactly.

Commands use one target-basis public type; conversion to Holden occurs only at the MM client boundary:

~~~python
@dataclass(frozen=True)
class CommandSample:
    chunk_index: int
    requested_velocity_mujoco: tuple[float, float, float]
    desired_heading_mujoco_wxyz: tuple[float, float, float, float]
~~~

The scored perturbations are:

~~~python
PERTURBATIONS = (
    ("lateral_p003", 0.03, 0.0),
    ("lateral_m003", -0.03, 0.0),
    ("lateral_p006", 0.06, 0.0),
    ("lateral_m006", -0.06, 0.0),
    ("yaw_p002", 0.0, math.radians(2.0)),
    ("yaw_m002", 0.0, math.radians(-2.0)),
    ("yaw_p004", 0.0, math.radians(4.0)),
    ("yaw_m004", 0.0, math.radians(-4.0)),
    ("combined_p", 0.03, math.radians(2.0)),
    ("combined_m", -0.03, math.radians(-2.0)),
)
~~~

- [ ] **Step 1: Write RED command-parity and threshold-boundary tests**

Compare Python route output against `route_schedule_cli` for every 25 Hz frame of `curb-forward`, `up-landing-down`, and `ascent-landing-descent`, then compare chunk-integrated displacement. Assert the exact flat chunk values and hash stability.

Metric tests cover aligned/missing/duplicate target rows; joint and quaternion formulas; local terrain clearance; up-dot; allowed versus forbidden contacts; target radius at 0.25 and just above; time at 1.25 nominal and just above; penetration at 0.005 and just above; known-good ratio at 1.5 and just above; nearest-rank quantiles; 8/10 and 7/10; aware-minus-blind 3 and 2; both-pass inconclusive; kinematic/dynamic failure attribution; and integration failure precedence.

- [ ] **Step 2: Run RED**

~~~bash
make -C sonic/cpp route_schedule_cli
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_commands \
  tests.python.test_sonic_metrics -v
~~~

Expected: route target or Python modules are missing.

- [ ] **Step 3: Implement fixed experiment registries and pure evaluators**

The three Stage C entries are exact:

~~~json
{
  "schema": "mm-sonic-terrain-experiment/v1",
  "chunk_intervals": 10,
  "source_rate_hz": 25,
  "target_rate_hz": 50,
  "conditions": {
    "aware": 4.0,
    "blind": 0.0
  },
  "scenes": [
    {"scene_id": "grail-curb-low", "route_id": "curb-forward"},
    {"scene_id": "ramp-10-up-down", "route_id": "up-landing-down"},
    {"scene_id": "stairs-shallow", "route_id": "ascent-landing-descent"}
  ],
  "success": {
    "target_radius_m": 0.25,
    "duration_multiplier": 1.25,
    "minimum_pelvis_local_height_m": 0.45,
    "minimum_pelvis_up_dot": 0.5,
    "forbidden_contact_groups": ["pelvis", "knees", "torso", "hands"],
    "require_exact_frame_coverage": true
  },
  "hypothesis": {
    "aware_minimum_successes": 8,
    "aware_minus_blind_minimum": 3
  }
}
~~~

Delivery audit first checks transmitted decoded indices are exactly `0..N-1`, then checks post-readiness official target rows against the canonical row at the sequential cursor. Because the pinned target logger omits frame index, this row-by-row equality plus the pinned `CurrentFrameAdvancement` source hash is the consumption evidence; any extra, missing, reordered, or unequal row is an integration failure.

- [ ] **Step 4: Run GREEN**

~~~bash
make -C sonic/cpp route_schedule_cli
PYTHONPATH=sonic/python \
  SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_commands \
  tests.python.test_sonic_metrics -v
~~~

Expected: all exact schedule, threshold, attribution, and aggregation tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/python/mm_sonic/commands.py \
  sonic/python/mm_sonic/metrics.py sonic/configs/experiments \
  sonic/schemas/trial_verdict_v1.schema.json \
  sonic/schemas/hypothesis_verdict_v1.schema.json \
  sonic/cpp/route_schedule_cli.cpp sonic/cpp/Makefile \
  tests/python/test_sonic_commands.py tests/python/test_sonic_metrics.py
git commit -m "feat: register SONIC baseline experiments and verdicts"
~~~

---

### Task 13: Implement the CLI and clear every Stage A integration gate

**Files:**
- Create: `sonic/python/mm_sonic/cli.py`
- Create: `tests/python/test_sonic_cli.py`
- Create: `sonic/configs/experiments/stage_a.json`
- Modify: `sonic/README.md`
- Modify: `sonic/python/mm_sonic/metrics.py`

**Interfaces:**
- CLI subcommands are non-interactive and return exit 0 only for a completed passing gate, 2 for configuration/integration failure, 3 for scientific failure, and 4 for `not_run` due to absent required external capability.
- `preflight` performs hashes, commit/source checks, joint certification, basis fixtures, model/scene load, geom registry, and CUDA/provider availability without stepping a trial.
- `stage-a --mode mm-reference` generates the full flat MM file reference and runs kinematic replay.
- `stage-a --mode known-good-file` drives the official known-good reference through the unmodified policy and registered flat simulator, recording target/actual metrics.
- `stage-a --mode known-good-stream` decodes the same official CSVs into the canonical buffer, publishes frame zero plus 20-frame chunks through this branch's ZMQ sink, and runs the same simulator setup.
- The file and stream runs use identical policy, encoder, observation config, G1 model, initial qpos, scene, and reference values. A decoded value mismatch prevents launch.
- Stage B/C commands refuse to run until the matching Stage A evidence hashes pass.

#### Corrected asynchronous scoring contract (independent-review amendment)

The pinned unmodified SONIC controller runs its 50 Hz policy loop from a
wall-clock recurrent thread; MuJoCo advances independently at 500 Hz.  A parent
request for 200 simulator steps therefore does **not** prove that exactly 20
policy actions were produced or that each action was held for exactly ten
steps.  Stage A must not infer policy-frame alignment from simulator step
counts.

For the known-good file and stream gates, launch a fresh SONIC process only as
far as `WAIT_FOR_CONTROL` while an explicitly unscored simulator epoch supplies
low state.  In stream mode, enable the ZMQ input and preload frame zero followed
by the exact 20-frame logical chunks while control is still inactive.  A padded
transport publication followed by a one-frame receipt fence may be used to
prove that the last logical chunk and required future horizon were assigned;
neither is a scored logical chunk.  In file mode, reset and
arm the loaded motion at frame zero while control is still inactive.  Then:

1. prove that no policy target/state row exists while SONIC remains in
   `WAIT_FOR_CONTROL`;
2. stop the SONIC process group while it is still in authenticated
   `WAIT_FOR_CONTROL`, reset MuJoCo to the registered initial qpos and a
   distinct scored log epoch, and prime fresh low state while SONIC remains
   stopped;
3. resume SONIC, activate `CONTROL`, and do not stop it again until scoring
   ends; a stdout transition marker followed by an external stop is not a
   valid pre-tick barrier;
4. drive the simulator until the official
   target log contains the exact canonical frames `0..440`, rejecting an
   overshoot, duplicate, omission, timeout, or post-stop row;
5. score joint positions and base orientation from SONIC's own authenticated
   `q.csv` and `base_quat.csv` rows.  The state files share exact indices and
   timestamps.  The target file has values only, so pair it positionally using
   the clean lifecycle, exact 441-row cardinality/order, and authenticated
   one-state-row/one-target-row source order after applying the pinned joint
   permutation;
6. retain MuJoCo state/contact logs and the actual simulator-step/time totals as
   asynchronous physical-cadence evidence.  Do not label them as lockstep
   policy rows or require a fabricated 10:1 tick-to-step coupling.

The file and stream runs must use the same cold pre-control lifecycle.  A
MuJoCo-only reset after policy readiness is not an acceptable scored reset,
because it leaves policy/history state warmed by an unaudited prior control
epoch.

- [ ] **Step 1: Write RED CLI parsing, exit-code, prerequisite, and mode tests**

Patch all process/simulator dependencies with fakes. Require exact required arguments, no implicit external path, output confinement, `not_run` semantics, mode-specific prerequisites, stable command manifest, correct exit codes, and refusal to promote a known-good file pass when stream delivery failed.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_cli -v
~~~

Expected: missing CLI module.

- [ ] **Step 3: Implement the CLI and Stage A evidence registry**

The CLI never shells into the external repository for mutation. It may execute its official deployment binary/scripts and import its simulator read-only. Record exact argv, environment allowlist, process hashes, and output files.

The Stage A registry has these gates in order:

~~~json
{
  "schema": "mm-sonic-stage-a/v1",
  "gates": [
    "external_identity",
    "joint_projection_round_trip",
    "basis_and_scene_alignment",
    "flat_mm_kinematic_replay",
    "known_good_file_dynamic",
    "known_good_stream_delivery",
    "known_good_stream_dynamic"
  ]
}
~~~

Known-good metrics are keyed by policy, encoder, observation config, external commit, official model XML, generated flat scene, initial qpos, and reference-buffer hashes. Never reuse metrics across a hash mismatch.

- [ ] **Step 4: Run CPU GREEN, then execute Stage A on the configured GPU machine**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_cli -v
test -d "$SONIC_GEAR_CHECKOUT"
test -f "$SONIC_POLICY"
test -f "$SONIC_OBS_CONFIG"
test -f "$SONIC_SOURCE_MJCF"
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli preflight \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-a \
  --mode mm-reference \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-a \
  --mode known-good-file \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
~~~

Expected: all seven Stage A gates pass. If an environment variable is absent or an LFS/checkpoint/GPU dependency is unavailable, the run writes `not_run` and execution stops without a feasibility claim.

- [ ] **Step 5: Commit code and compact Stage A summary only**

~~~bash
git add sonic/python/mm_sonic/cli.py sonic/python/mm_sonic/metrics.py \
  sonic/configs/experiments/stage_a.json tests/python/test_sonic_cli.py \
  sonic/README.md
git commit -m "feat: add staged SONIC integration preflight"
~~~

Do not commit `sonic/runs`. If Stage A executed, commit only a compact result JSON whose hashes point to retained immutable local evidence; otherwise do not fabricate one.

---

### Task 14: Run the deterministic 12-second flat MM plus SONIC gate

**Files:**
- Create: `sonic/configs/experiments/stage_b.json`
- Create: `sonic/python/mm_sonic/operator.py`
- Create: `tests/python/test_sonic_stage_b.py`
- Create: `tests/python/test_sonic_operator.py`
- Modify: `sonic/python/mm_sonic/cli.py`
- Modify: `sonic/python/mm_sonic/metrics.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Stage B uses `sonic-flat-baseline`, terrain weight `0.0`, 30 chunks, 601 target frames including frame zero, and exactly 12.0 seconds of MuJoCo time.
- Before dynamic tracking, replay the full reference kinematically in the same generated G1 model/scene.
- Dynamic pass requires complete command/frame coverage, no forbidden contacts, local pelvis height at least `0.45` m, pelvis up-dot at least `0.5`, joint-position RMSE no more than 1.5 times known-good, pelvis-orientation RMS no more than 1.5 times known-good, and no chunk/delivery failure.
- When a known-good metric is at most `1e-8`, candidate error must also be at most `1e-8` instead of dividing by zero.
- Manual steering is disabled until this scripted gate passes and is never entered into the scored verdict. Once enabled, `manual` samples keyboard/gamepad state only at a 0.4-second boundary, queues input received during generation, and otherwise uses the identical coordinator, bridge, scene, gate, and run-artifact path.
- The initial keyboard contract is `W/S` forward/back, `A/D` lateral, `Q/E` desired-heading increments, space to stand, and `X` to terminate cleanly. Velocity is capped to the source database's registered walk/side/back limits; the operator remains responsible for obstacle avoidance.

- [ ] **Step 1: Write RED Stage B exact-frame, threshold, and prerequisite tests**

Use fake canonical/reference/sim logs to require 601 frames, 30 advances of exactly 0.4 seconds, 12.0 total sim seconds, exact command phase boundaries, Stage A hash match, kinematic pass before dynamic launch, every safety threshold, tracking ratio behavior, and failure attribution among conversion, delivery, reference, and SONIC tracking. Operator tests require boundary-only latching, queued mid-generation input, normalized diagonals, heading increments independent of velocity, speed caps, clean termination, and refusal without matching Stage B evidence.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_stage_b \
  tests.python.test_sonic_operator -v
~~~

Expected: missing Stage B registry/entry point.

- [ ] **Step 3: Implement the Stage B runner and report**

The machine-readable result includes:

~~~json
{
  "schema": "mm-sonic-trial-verdict/v1",
  "stage": "B",
  "scene_id": "sonic-flat-baseline",
  "terrain_weight": 0.0,
  "expected_frames": 601,
  "expected_sim_time_s": 12.0,
  "integration_pass": false,
  "kinematic_pass": false,
  "dynamic_pass": false,
  "failure_layer": null,
  "metrics": {},
  "timings": {},
  "evidence_hashes": {}
}
~~~

Booleans begin false and are set only after their complete gates. A crash or partial file cannot look like a pass. Implement `OperatorSampler.sample_boundary()` as a pure state-to-`CommandSample` conversion; terminal/joystick polling stays in a small adapter so it can be replaced later without touching MM or reference semantics.

- [ ] **Step 4: Run GREEN unit tests and the real Stage B gate**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest \
  tests.python.test_sonic_stage_b \
  tests.python.test_sonic_operator -v
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-b \
  --stage-a-evidence "$SONIC_STAGE_A_EVIDENCE" \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
~~~

Expected: unit tests pass. The real command must pass Stage B before Stage C or manual steering is enabled; otherwise preserve its attributed failure and stop.

After a pass, run the non-scored steering demonstration with:

~~~bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli manual \
  --stage-b-evidence "$SONIC_STAGE_B_EVIDENCE" \
  --scene grail-curb-low \
  --terrain-weight 4 \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
~~~

Expected: each visible motion segment corresponds to one recorded 0.4-second command, and the run is labeled `demonstration`, never `scored`.

- [ ] **Step 5: Commit**

~~~bash
git add sonic/configs/experiments/stage_b.json \
  sonic/python/mm_sonic/operator.py sonic/python/mm_sonic/cli.py \
  sonic/python/mm_sonic/metrics.py tests/python/test_sonic_stage_b.py \
  tests/python/test_sonic_operator.py sonic/README.md
git commit -m "feat: add scripted flat MM SONIC gate"
~~~

---

### Task 15: Execute the matched terrain-aware versus blind causal matrix

**Files:**
- Create: `sonic/python/mm_sonic/matrix.py`
- Create: `tests/python/test_sonic_matrix.py`
- Modify: `sonic/python/mm_sonic/cli.py`
- Modify: `sonic/python/mm_sonic/metrics.py`
- Modify: `sonic/README.md`

**Interfaces:**
- Stage C refuses to run without matching Stage A and passing Stage B evidence.
- For each scene and condition, generate one deterministic MM source/target reference from the frozen command sequence, assert repeat generation hash equality, replay it kinematically, and retain it immutable.
- Run one nominal dynamic smoke trial and ten scored perturbation trials by streaming the same retained condition reference chunk-by-chunk. The MM virtual root/reference is unchanged across perturbations; only physical initial qpos differs.
- The safe dynamic trial still runs when kinematic replay reports a reference collision; that report controls attribution, not suppression.
- Trial order is deterministic and paired by scene/perturbation, alternating which condition runs first. Every trial is its own immutable run. Matrix resume may schedule only wholly missing trial IDs with matching experiment/reference hashes; it never resumes a partial trial.
- Per class, H1 terrain awareness is supported only if aware succeeds at least 8/10 and beats blind by at least 3. If both succeed at least 8/10, composition feasibility is supported but the awareness effect is inconclusive.

- [ ] **Step 1: Write RED matrix enumeration, pairing, caching, resume, and aggregation tests**

Require exactly 3 scenes × 2 conditions × (1 nominal + 10 scored) = 66 dynamic trials, six kinematic reports, exact perturbation order, unchanged reference hashes within a condition, different weight identity between conditions, physical-only reset, paired condition alternation, safe discovery of missing trials, rejection of stale/partial/mismatched evidence, and all approved aggregate outcomes.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_matrix -v
~~~

Expected: missing matrix module.

- [ ] **Step 3: Implement deterministic matrix scheduling and aggregate reports**

Use these stable IDs:

~~~python
def trial_id(
    scene_id: str,
    condition: str,
    perturbation_id: str,
) -> str:
    return f"stage-c__{scene_id}__{condition}__{perturbation_id}"
~~~

The aggregate report lists, for each class, aware/blind successes, integration failures separately from scientific failures, kinematic defect counts, pass margin, composition verdict, awareness verdict, and evidence hashes. Overall H1 is supported only if all three class verdicts meet the aware and margin thresholds. Never average success counts across terrain classes.

- [ ] **Step 4: Run GREEN unit tests, nominal smoke tests, then the full matrix**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_matrix -v
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli matrix \
  --mode nominal \
  --stage-a-evidence "$SONIC_STAGE_A_EVIDENCE" \
  --stage-b-evidence "$SONIC_STAGE_B_EVIDENCE" \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli matrix \
  --mode scored \
  --stage-a-evidence "$SONIC_STAGE_A_EVIDENCE" \
  --stage-b-evidence "$SONIC_STAGE_B_EVIDENCE" \
  --gear-checkout "$SONIC_GEAR_CHECKOUT" \
  --policy "$SONIC_POLICY" \
  --observation-config "$SONIC_OBS_CONFIG" \
  --encoder "$SONIC_ENCODER" \
  --source-mjcf "$SONIC_SOURCE_MJCF" \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs
~~~

Expected: unit tests and six nominal smoke trials complete before the 60 scored trials. Report the hypothesis only from the complete scored aggregate; a partially completed matrix has `status=incomplete`.

- [ ] **Step 5: Commit code and, only if complete, the compact aggregate summary**

~~~bash
git add sonic/python/mm_sonic/matrix.py sonic/python/mm_sonic/cli.py \
  sonic/python/mm_sonic/metrics.py tests/python/test_sonic_matrix.py \
  sonic/README.md
git commit -m "feat: add matched terrain MM SONIC matrix"
~~~

Do not commit raw runs, policies, external code, generated scenes, target logs, or videos.

---

### Task 16: Add one reproducible verification entry point and perform final review

**Files:**
- Create: `sonic/scripts/test_cpu.sh`
- Create: `sonic/scripts/check_scope.py`
- Create: `tests/python/test_sonic_scope.py`
- Modify: `sonic/README.md`
- Modify: `docs/superpowers/specs/2026-07-15-g1-sonic-scene-aware-baseline-design.md`

**Interfaces:**
- `test_cpu.sh` builds all headless C++ targets/tests and runs every `test_sonic_*` Python module without external policy/GPU requirements.
- Real-artifact tests run only when `SONIC_TERRAIN_DIR` and `SONIC_SOURCE_MJCF` are explicitly set. GPU tests remain separate.
- Scope checking forbids runtime/package imports or identifiers for depth, lidar, DAgger, learned student, action chunking, measured-root feedback, and obstacle planning. These words may appear only in design/README non-goals.
- The design status changes from “awaiting written-spec review” to “approved; implementation plan linked” without changing the approved scientific contract.

- [ ] **Step 1: Write RED scope and verification-script tests**

Require the scope checker to catch a synthetic forbidden runtime import and ignore the approved non-goal prose. Require the shell script to use `set -euo pipefail`, compile outside the source tree or beneath ignored `sonic/build`, run all focused C++ tests, run Python discovery for `test_sonic_*.py`, and avoid mutation of external/user artifact paths.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  -m unittest tests.python.test_sonic_scope -v
~~~

Expected: missing scope checker.

- [ ] **Step 3: Implement the verification entry point and final documentation**

The script begins:

~~~bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mkdir -p sonic/build/tests
make -C sonic/cpp clean all
PYTHONPATH=sonic/python python -m unittest discover \
  -s tests/python -p 'test_sonic_*.py' -v
~~~

Then compile/run every focused C++ test listed in this plan with `-Wall -Wextra -Werror -pedantic`. Add an ASan/UBSan pass for state clone, runtime, projection, and protocol. The README gives the exact ladder: CPU tests, file preflight, Stage A, Stage B, manual demo, nominal terrain, scored matrix, evidence interpretation.

The required C++ executables are exactly:

~~~bash
cpp_tests=(
  test_g1_controller_state
  test_g1_runtime
  test_g1_joint_projection
  test_mm_chunk_protocol
  test_sonic_flat_scene
)
for name in "${cpp_tests[@]}"; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${name}.cpp" -o "sonic/build/tests/${name}"
  "sonic/build/tests/${name}"
done
~~~

- [ ] **Step 4: Run full verification**

~~~bash
bash sonic/scripts/test_cpu.sh
git diff --check
PYTHONPATH=sonic/python python sonic/scripts/check_scope.py
git status --short
git diff --stat c1f97c201228376b22a7a114dfd4c757090383b4
~~~

Also rerun the root controller parity command from Task 3 and the real-artifact MM server test from Task 4. If GPU evidence exists, run `evaluate` against every completed Stage A/B/C run and compare the aggregate digest to the retained compact summary.

Expected: all CPU tests, sanitizer tests, scope check, controller parity, server integration, and diff checks pass. `git status` contains only intentional branch changes and never shows the user's primary terrain workspace.

- [ ] **Step 5: Request code review and fix only verified findings**

Use the `requesting-code-review` skill against the full branch diff. Check especially transactional failure paths, quaternion/order correctness, process cleanup, external path writes, target frame ownership, and metric threshold inequalities. Re-run Step 4 after any fix.

- [ ] **Step 6: Commit**

~~~bash
git add sonic/scripts/test_cpu.sh sonic/scripts/check_scope.py \
  tests/python/test_sonic_scope.py sonic/README.md \
  docs/superpowers/specs/2026-07-15-g1-sonic-scene-aware-baseline-design.md
git commit -m "docs: finalize SONIC baseline verification workflow"
~~~

---

## Implementation handoff gates

- Tasks 1–12 are CPU/framework gates and may be completed without claiming that SONIC tracks MM.
- Task 13 is the first external-policy gate. No Stage B/C claim exists unless every Stage A hash and result is present.
- Task 14 answers whether vanilla flat MM references are in distribution enough for unmodified SONIC.
- Task 15 answers the narrow terrain-aware composition hypothesis. Its timing evidence is descriptive only and does not answer the later real-time/action-chunking hypothesis.
- If implementation is blocked by missing policy/encoder/GPU assets, stop after writing a truthful `not_run` bundle. The framework can still be complete while the experiment remains unexecuted.
