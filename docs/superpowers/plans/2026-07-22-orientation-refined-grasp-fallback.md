# Orientation-Refined Grasp Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fallback pickups actively refine full wrist orientation while retaining hard position, approach-axis, and full-body clearance requirements.

**Architecture:** Extend trajectory selection and shaping with a soft full-quaternion fallback score and an achieved-orientation diagnostic. Tighten object collision exemption in the trajectory library, then have the viewer rank accepted fallback results and use a conservative clearance configuration.

**Tech Stack:** C++17, existing G1 arm IK and trajectory library, raylib viewer, Python `unittest`, GNU Make.

## Global Constraints

- Exact-pose search remains unchanged and always ranks first.
- Fallback retains hard 4 cm position and 15 degree approach-axis acceptance.
- Fallback uses orientation weight `0.10`, at least 16 IK iterations, and a 60 degree achieved-orientation ceiling.
- Fallback retrieval adds a `0.25` normalized full-quaternion cost term.
- Only the final active wrist sphere and wrist-pitch-to-wrist segment are exempt from object collision after Contact.
- Viewer collision radii gain 2 cm; library defaults remain unchanged.
- Keep Enter-only search and the existing 12-result fallback target.
- Do not launch mesh or terrain renderers.

---

### Task 1: Refine and measure fallback orientation

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `GraspOrientationMode::ApproachAxis`, `IKConfig`, and the requested full grasp rotation.
- Produces: `ShapedHandTrajectory::achieved_orientation_error_radians` and orientation-refined fallback poses.

- [ ] **Step 1: Write failing shaping tests**

Extend the existing reachable fallback fixture by rotating its source wrist 30 degrees around its local approach axis while leaving the requested grasp rotation unchanged. Measure the mapped Contact quaternion error before shaping, then require that shaping reduces that error while retaining accepted position and axis. Reuse that fixture with a 75-degree source twist and set `IKConfig::maximum_step_radians = 1.0e-6F`; require rejection with `Reason::CorrectionLimit` because all 16 iterations leave the achieved result above the 60-degree ceiling:

```cpp
const auto shaped = interaction::shape_hand_trajectory(
    database, selected.front(), query);
require(shaped.contact_accepted, "refined fallback was rejected");
require(shaped.achieved_orientation_error_radians < 0.523598776F,
        "fallback did not improve the 30 degree source twist");
require(length(shaped.path.hands[selected.front().contact_point].position -
               query.grasp_world_position) <= 0.04F,
        "orientation refinement lost contact position");

interaction::IKConfig no_refinement{};
no_refinement.maximum_step_radians = 1.0e-6F;
const auto excessive = interaction::shape_hand_trajectory(
    database, excessive_selected.front(), excessive_query, no_refinement);
require(!excessive.contact_accepted,
        "fallback accepted more than 60 degrees of achieved error");
require(excessive.reason == interaction::ShapedHandTrajectory::Reason::CorrectionLimit,
        "fallback used the wrong orientation-ceiling rejection reason");
```

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because the achieved orientation field is absent.

- [ ] **Step 3: Add the soft fallback retrieval term**

For `ApproachAxis`, retain the hard 25-degree axis gate and calculate cost as:

```cpp
trajectory.cost = position_error / position_normalizer +
    axis_error / orientation_normalizer +
    0.25F * quaternion_angle(
        mapped_contact.rotation, query.grasp_world_rotation) / pi;
```

Exact and position-only cost behavior remains unchanged.

- [ ] **Step 4: Refine fallback orientation and apply hard final gates**

Apply the same Reach-to-Contact full quaternion correction used by exact mode to `ApproachAxis`. For fallback set `maximum_request_orientation_radians = pi`, `orientation_scale_m_per_radian = 0.10F`, and `maximum_iterations = max(16, config.maximum_iterations)`, while retaining the caller's 15-degree solver acceptance. Preserve the solver's best pose even when its full quaternion result is not accepted. At Contact calculate position, approach-axis, and full quaternion errors from the solved wrist; accept only position `<= config.accepted_position_m`, axis `<= config.accepted_orientation_radians`, and full error `<= 1.047197551F`.

Initialize `achieved_orientation_error_radians` to `0.0F` in the struct and overwrite it from the solved Contact wrist versus the requested grasp quaternion for every orientation mode.

- [ ] **Step 5: Verify and commit**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git diff --check
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: refine fallback grasp orientation"
```

### Task 2: Restrict Contact collision exemption

**Files:**
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `evaluate_shaped_trajectory_feasibility` and G1 wrist bone identities.
- Produces: object-contact exemption for only the final active wrist bone.

- [ ] **Step 1: Write failing collision tests**

Update the existing post-Contact exemption test so an elbow-to-wrist-roll or wrist-roll-to-wrist-pitch segment crossing the object returns `ObjectCollision`. Add a separate pose where only the final active wrist and its parent segment overlap the object and require `None`. Keep the inactive-hand collision assertion.

```cpp
require(interaction::evaluate_shaped_trajectory_feasibility(
            forearm_shaped, 0U, interaction::Hand::Right,
            object, environment).reason ==
        interaction::TrajectoryFeasibilityReason::ObjectCollision,
        "active forearm was exempted after Contact");
```

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories && ./build/tests/test_interaction_hand_trajectories`.

Expected: the active forearm test fails because three wrist-chain bones are currently skipped.

- [ ] **Step 3: Implement wrist-only exemption**

Replace `active_grasp_chain_bone` with `active_contact_wrist_bone`, returning true only for `LeftWrist` or `RightWrist`. In `skeleton_intersects_box`, skip the sphere and parent segment only when the current bone is that final wrist and the post-Contact exemption is enabled. Wrist roll, wrist pitch, forearm, elbow, upper arm, torso, legs, and inactive wrist remain tested.

- [ ] **Step 4: Verify and commit**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git diff --check
git add interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "fix: reject post-contact body penetration"
```

### Task 3: Rank refined options and add viewer clearance

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- Consumes: `achieved_orientation_error_radians` and `TrajectoryCollisionConfig`.
- Produces: orientation-ranked fallback display, HUD error in degrees, and 2 cm viewer clearance.

- [ ] **Step 1: Write failing viewer tests**

Require the source to contain `achieved_orientation_error_radians`, `orientation_error_degrees`, `std::stable_sort`, and assignments adding `0.02F` to `joint_radius_m`, `limb_radius_m`, and `torso_radius_m`. Preserve exact-first and Enter-only assertions.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v`.

Expected: the new refinement and clearance assertions fail.

- [ ] **Step 3: Apply viewer clearance and ranking**

Construct a viewer-only `TrajectoryCollisionConfig` with joint, limb, and torso radii increased by `0.02F`, and pass it to shaped feasibility evaluation. After candidate processing, `std::stable_sort` valid paths with a tier-aware comparator: exact paths compare equivalent so their retrieval order is preserved; exact always precedes fallback; two fallback paths compare achieved orientation error, then retrieval cost, then clip ID. Because processing stops at 12 total accepted paths, sorting does not increase IK work.

- [ ] **Step 4: Display achieved orientation error**

Convert the selected option's stored radians to degrees and add `orient %.1f deg` to the option HUD line. Keep `EXACT`, `AXIS-FALLBACK`, support, phase, and frame labels.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
git add hand_trajectory_viewer.cpp tests/python/test_hand_trajectory_viewer.py
git commit -m "feat: rank collision-safe grasp orientations"
```

### Task 4: Regression and live replacement

**Files:**
- Modify only files already listed if verification exposes a task-related defect.

**Interfaces:**
- Consumes: completed release viewer and mixed interaction pack.
- Produces: passing evidence and exactly one live refined viewer.

- [ ] **Step 1: Run the established branch gate**

```bash
python3 -m unittest tests.python.test_interaction_sources tests.python.test_interaction_build_cli tests.python.test_interaction_artifacts tests.python.test_hand_trajectory_viewer
make -B build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_trajectory_database
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
git status --short
```

Expected: 104 Python tests pass, both C++ binaries exit zero, the release viewer builds, and the worktree is clean.

- [ ] **Step 2: Replace only the exact viewer PID**

Resolve the single process matching `^./hand_trajectory_viewer$`, verify its cwd is this worktree, send SIGTERM only to that PID, and wait for exit. Launch on `DISPLAY=:1` with the existing `build/smart-pickup/table-ground-pack`. Do not touch controller PID `556617`.

- [ ] **Step 3: Verify live identity and bounded memory**

Confirm one matching viewer, matching disk/live SHA-256, correct `DISPLAY` and pack environment, initialized raylib window, and RSS below `1.1 GiB` over 10 seconds. Do not use screenshots.
