# G1 Mesh Locomotion Arm Reference Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the G1 mesh from inheriting a raised pickup-arm reference during ordinary locomotion.

**Architecture:** Add one geometry-preserving calibration function beside the existing flat/G1 adapter. The controller builds a mesh-only reference from its fixed flat locomotion reference and uses it only for mesh expansion; interaction handoff references remain unchanged.

**Tech Stack:** C++17, existing interaction pose adapter, Python source-integration tests, Raylib 6.

## Global Constraints

- Do not alter diffusion, locomotion selection, interaction ownership, or pickup timing.
- Preserve all non-root G1 local translations during mesh-reference calibration.
- Do not use external screenshot or X11 capture tools.

---

### Task 1: Geometry-preserving mesh reference calibration

**Files:**
- Modify: `interaction_controller_adapter.h`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`

**Interfaces:**
- Produces: `interaction::calibrate_flat_mesh_reference(const Pose&, const FlatControllerPose&) -> Pose`

- [ ] Add a focused test whose G1 reference has raised mapped arm rotations and whose flat reference has lowered arm rotations. Require mapped arm world rotations to match the flat reference and every non-root local translation to remain unchanged.
- [ ] Compile and run the focused adapter test; confirm the new symbol is missing.
- [ ] Implement calibration by obtaining absolute mapped rotations through the existing two-argument expansion, restoring G1 non-root local translations, and zeroing velocities and angular velocities.
- [ ] Re-run the focused adapter test and require exit zero.

### Task 2: Controller mesh-only reference

**Files:**
- Modify: `controller.cpp`
- Modify: `tests/python/test_diffusion_pickup_g1_mesh_integration.py`

**Interfaces:**
- Consumes: `calibrate_flat_mesh_reference`
- Produces: `mesh_reference_pose`, used only by the final mesh expansion

- [ ] Extend the integration test to require a separate mesh reference and reject use of `interaction_reference_pose` in the final mesh expansion.
- [ ] Run the integration test and confirm it fails on the old expansion.
- [ ] Initialize `mesh_reference_pose` after `interaction_flat_reference_pose`, then use it in the final post-IK mesh expansion.
- [ ] Re-run the integration test and require all cases to pass.

### Task 3: Verification and relaunch

**Files:**
- Modify: none

**Interfaces:**
- Consumes: fixed controller binary
- Produces: live mapped Raylib window for manual validation

- [ ] Run focused adapter, mesh renderer, and Python integration tests.
- [ ] Build `controller` with the pinned Raylib 6 dependency.
- [ ] Launch the controller on display `:1` with the current interaction pack and feature artifact.
- [ ] Verify using process and X11 window metadata only that the 1280x720 Raylib window remains mapped; do not capture the display.
