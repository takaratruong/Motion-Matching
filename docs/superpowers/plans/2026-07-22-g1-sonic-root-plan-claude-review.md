# Independent Review — G1 SONIC Local Root-Trajectory Conditioning

## Scope and method

This is an independent Claude audit of the approved design and its three
implementation plans against the two pinned codebases. A protected reviewer
rejected the worker's first B1 formulation because it overlooked explicit but
conflicting width comments in checked-in deployment YAML. The Codex foreman
corrected B1 below against both those comments and the controller-qualified live
deployment bundle; all other findings retain the worker's audit content.

Design under review:

- `docs/superpowers/specs/2026-07-22-g1-sonic-local-root-trajectory-design.md`

Plans under review:

- `docs/superpowers/plans/2026-07-22-g1-sonic-root-conditioned-training.md` (training)
- `docs/superpowers/plans/2026-07-22-g1-sonic-root-stream-runtime.md` (runtime/stream)
- `docs/superpowers/plans/2026-07-22-g1-sonic-root-terrain-evaluation.md` (evaluation)

Sources inspected:

- Motion Matching worktree at the frozen base revision (this repository, HEAD
  `196988f6c2051c04a43a421bc4e87c71a4fc2af4`), notably `sonic/cpp/mm_chunk_protocol.h`
  and `sonic/cpp/g1_joint_projection.h`.
- Read-only pinned GEAR checkout at
  `/home/ubuntu/projects/gear-sonic-worktrees/simulation-lowstate-wait-v6`
  (design pin commit `294110cedba01ad764f1e268d57ddf7c1bbf9523`).

Each finding is tagged **[FACT]** (directly observed in source) or
**[INFERENCE]** (reasoned from observed evidence). Blocking findings cite exact
paths/symbols and give a narrow correction. Optional improvements are separate.

### Correction of a prior false premise

A previous review attempt asserted the pinned GEAR checkout was inaccessible and
concluded the GEAR-dependent claims were unverifiable. **That accessibility
premise is withdrawn.** The checkout is readable, and this report re-derives
every GEAR-dependent claim directly from it. The checked-in YAML comments and
the controller-qualified deployed artifact disagree on the base encoder width;
B1 records that conflict explicitly rather than preferring either silently.

### Evidence note on the GEAR pin

**[FACT]** The worker sandbox could not traverse the worktree's external gitdir,
so the worker verified claims against the readable working-tree contents. The
Codex controller separately ran `git rev-parse HEAD` outside that sandbox and
confirmed `294110cedba01ad764f1e268d57ddf7c1bbf9523` with a clean status before
accepting source evidence.

---

## Requirement coverage map

| # | Design requirement | Plan location | Verdict |
|---|---|---|---|
| R1 | Frozen encoder ignores root-z; A1 negative control | training Global Constraints; eval A1 | Consistent; see F1/F10 |
| R2 | Canonical `r_i = inv(h_0)*(p_i − proj_xy(p_0))` | training Task 1; runtime Task 5 | **Verified in GEAR source**; see F2 |
| R3 | Physical pelvis source, not virtual root | runtime Task 1/3; training Task 5 | **Verified source and conversion**; see F3 |
| R4 | Versioned wire contract carrying `body_pos [N,3]` | runtime Task 2 | **Verified v4 is token-only → v5 free**; see F4 |
| R5 | GEAR merger writes `BodyPositions(frame)[0]` atomically | runtime Task 4 | **Verified symbols exist**; see F5 |
| R6 | 1762→1792 exported observation layout | training Task 6; runtime Task 6 | **Artifact-verified, docs conflict**; see B1 |
| R7 | 640→670 framewise checkpoint migration | training Task 3 | Arithmetic sound; shapes partly verified; see F7 |
| R8 | Stage-A freeze boundary | training Task 2/4 | Consistent; see F8/B2 |
| R9 | Dataset provenance / hashed mixed dataset | training Task 5 | Consistent; see B3 |
| R10 | A0/A1/B(/C) gates | evaluation plan | **Verified well-structured**; see F10 |

Every enumerated design requirement is mapped. No requirement is unaddressed by
the plans; the gaps below are correctness/reproducibility gaps within addressed
requirements.

---

## Verified consistency findings

### F1 — Frozen-encoder inertness is the correct premise (R1)

**[FACT]** The design's central claim is that the released G1 encoder is
insensitive to root-z (design lines 45–61), so plumbing alone cannot change
behavior and a new trained channel is required. The training and eval plans build
directly on this via the A1 negative control. This premise is consistent with the
GEAR source treating body-0 z as an *output/diagnostic* read
(`g1_deploy_onnx_ref.cpp:447`), not an encoder input in G1 mode.

### F2 — Canonical transform matches GEAR source exactly (R2)

**[FACT]** `gear_sonic/envs/manager_env/mdp/commands.py:1137`
`root_transforms_relative_to_first_frame` computes exactly the design formula:
it projects the first frame's root position to z=0
(`first_frame_pos_projected[..., 2] = 0`), extracts a heading-only quaternion
via `get_heading_q`, forms `delta_pos_w = root_pos_w - first_frame_pos_projected`,
and rotates the delta into the first-frame heading via `quat_apply`. It returns
`[N, F, 9]` = 3 position + 6D rotation.

**[FACT]** Because the first frame is projected to z=0 while later frames keep
their true z, and a yaw-only rotation preserves z, the position slice's third
component is the **absolute** pelvis height and `r_0 = [0, 0, p_0.z]` — precisely
the design's claim (design lines 74–86).

**[FACT]** Training Task 1 takes the `[..., 0:3]` slice of this tensor and
appends only those 3 position values, retaining the existing 6D orientation term
(`motion_anchor_ori_b_mf_nonflat`). This matches the design's "smallest change"
(design lines 93–98). The existing 9-value term is exposed by
`command_multi_future_root_transforms` (`observations.py:1141`), confirming the
6D rotation is not duplicated. **This is a genuine strength: the design's math is
already implemented in the pinned source.**

### F3 — Physical-pelvis vs virtual-root distinction is real in MM source (R3)

**[FACT]** `sonic/cpp/mm_chunk_protocol.h:67,69` defines both
`float physical_pelvis_position_holden[3]` and `float virtual_root_position_holden[3]`.
`sonic/cpp/g1_joint_projection.h:596` sets
`candidate.physical_pelvis_position_holden = global_positions(G1_Hips)`, i.e. the
physical pelvis is G1 hips from forward kinematics — matching the design's
"physical pelvis coherent with the 29 joint targets" (design lines 100–106). The
runtime plan's insistence on transmitting `physical_pelvis_position` and never
`virtual_root_position` (runtime Global Constraints) is therefore well grounded.

**[FACT]** The C++ field is Holden-frame (`_holden` suffix), and the Python
timeline converts it before publication: `sonic/python/mm_sonic/resample.py`
lines 359–366 applies `holden_to_mujoco_position` and
`holden_to_mujoco_quaternion_wxyz` before resampling; the initial-boundary path
does the same in `sonic/python/mm_sonic/timeline.py` lines 274–281. The final
`TargetChunk` receives `resampled.physical_pelvis_position` at timeline line
529. Runtime tests should still pin this with a known-axis/standing-height case.

### F4 — Protocol version choice is correct (R4)

**[FACT]** `zmq_endpoint_interface.hpp` handles protocol versions 1, 2, 3, and 4;
version 4 is explicitly "Token-Only Streaming" (comment at the v4 branch, ~line
688). No version 5 exists. The runtime plan's choice to add **v5** for the
root-capable motion message, leaving v1 byte-for-byte unchanged, is consistent
with the source (runtime Global Constraints, Task 2).

**[FACT]** The endpoint rejects a mid-session version switch:
`active_protocol_version_` is established on first message and a later mismatch is
signalled/rejected (~lines 702–706). This supports the design's requirement to
reject an older runtime for the root-aware checkpoint (design lines 151–153),
though negotiation there is version-monotonic rather than an explicit capability
handshake; see optional item O3.

### F5 — GEAR merger target symbols exist (R5)

**[FACT]** `MotionSequence::BodyPositions(frame)[0]` is a real accessor used
throughout `g1_deploy_onnx_ref.cpp` (e.g. lines 447, 1160, 1320), and body index
0 is already treated as the root (`motion_root_z_pos = BodyPositions(target)[0][2]`
at line 447). The merger's `IncomingData` struct
(`streamed_motion_merger.hpp:83`) currently carries `joint_pos`, `joint_vel`,
`body_quat`, and `frame_indices` but **no `body_pos` member** — exactly the gap
the runtime plan Task 4 fills. The merger already validates required fields and
rejects on mismatch (lines 217–242), so the plan's "reject partial/mismatched
root array atomically" extends an existing pattern rather than inventing one.

### F7 — 640→670 migration arithmetic is sound; shapes partly verified (R7)

**[FACT]** `10 × 64 = 640` and `10 × 67 = 670`; the training-plan framewise
remap copies each frame's 64-wide block into a 67-wide stride and zeroes the 3
new slots, and Task 3 Step 1 asserts exactly that. Internally consistent.

**[FACT]** `gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml` declares
`hidden_dims: [2048, 1024, 512, 512]`, so the encoder's first learned layer has
2048 output rows — matching the plan's hardcoded expected shape `[2048, 640]`
(training Task 3). The kinematic decoder's final-layer expected shape `[640, 512]`
with `[640]` bias is consistent with a 512-wide last hidden and 640 output.

**[INFERENCE / gap]** I confirmed the 2048 row count but not the literal `640`
input width or the exact state-dict key strings
(`actor_module.encoders.g1.module.0.weight`,
`actor_module.decoders.g1_kin.module.8.weight`) from the released checkpoint,
because the checkpoint is a downloaded artifact outside the audit's read scope.
The plan is fail-closed here (it errors unless each suffix matches exactly one
key of the exact shape), which is the correct mitigation. See B1.

### F10 — Evaluation gate structure is sound (R10)

**[FACT]** The evaluation plan implements the design's A0/A1/B(/C) matrix
faithfully: A0 = released model over v1, A1 = identical released hashes over v5
(causal negative control), B = Stage-A encoder over v5 (Architecture; Global
Constraints lines 14–17). It shares one command SHA, initial state, scene
(`grail-curb-low`), route (`curb-forward`), and terrain weight `4.0` across
variants and enforces this in a registry validator that requires *only* protocol
to differ A0→A1 and *only* encoder/config/root-range to differ A1→B, with a
**common decoder hash** (Task 1, line 98). This structurally prevents an A0/A1
comparison that silently changes the model.

**[FACT]** The primary gate requires exactly three consecutive `dynamic_pass`
low-curb trials plus one passing flat verdict, emits `stage_b_required = not
primary_pass`, and does **not** auto-start Stage B (Task 3 line 226; Global
Constraints line 25) — matching design lines 324–339 and 248–259.

**[FACT]** The A1 inertness probe asserts **maximum token delta exactly `0.0`**
(Task 5, line 325), and `verify_root_onnx.py` uses `INITIAL_TOKEN_ATOL = 0.0`;
these two agree, so the design's "exactly zero" invariant is enforced
consistently. (This resolves what would otherwise be a tolerance-mismatch risk.)

---

## Blocking findings

### B1 — Checked-in width comments conflict with the deployed artifact (R6)

**[FACT]** The checked-in release YAML headers state three different widths:
`observation_config.yaml:3` says total policy dimension 436,
`observation_config_low_latency.yaml:6` says encoder dimension 1247, and
`observation_config_sonic_release.yaml:7` says encoder dimension 1751. The
worker's original claim that these files contained no explicit widths was false,
and the protected reviewer correctly rejected it.

**[FACT]** The exact deployment bundle already qualified by the controller is
not the 1751-commented artifact. Its encoder ONNX input is `[1,1762]` and output
is `[1,64]`. Identities are encoder
`013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3`,
decoder `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`,
and config `466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1`.
That exact pair supports the plan's live `[1762,1792)` append, but the conflicting
source comments make a literal width unsafe for any other export.

**Impact:** Selecting the 1751-commented export while retaining `[1762,1792)`
would misalign every root value. Selecting the qualified 1762 model without
checking its config hash would make the same mistake silently after any artifact
update.

**Narrow correction:** Derive `base_width` from the selected ONNX input shape,
independently sum the runtime gatherer dimensions from the selected config, and
require equality before startup/export. Derive the root range as
`[base_width, base_width + 30)`. For the currently qualified hashes, assert
`base_width == 1762` and final width `1792`; record model/config hashes beside
the derived range. Never infer the live model width from a YAML comment alone.

### B2 — Freeze audit must gate the entrypoint before the first optimizer step (R8)

**[FACT]** Training Task 2 adds `trainable_parameter_groups()` and Task 4 Step 4
says "At initialization, assert the trainable audit has positive counts only for
`encoders.g1` and `decoders.g1_kin`; raise `RuntimeError` before rollout
otherwise." The design requires Stage A freeze the control decoder and all non-G1
components (design lines 232–238; plan Global Constraints).

**Impact / gap:** As written, the audit is a method plus a prose assertion. The
task does not name the exact call site (which function in
`train_agent_trl.py` / the module constructor) where the `RuntimeError` is raised,
nor a test that drives the *real* training entrypoint and asserts it raises
**before** any optimizer step or environment rollout when, e.g., `g1_dyn` is
accidentally trainable. A helper-only test (Task 2 Step 1 constructs a module and
checks the dict) does not establish the stop-before-side-effect boundary the
milestone requires.

**Narrow correction:** Add a focused test that constructs the Stage-A model with
an intentionally-unfrozen `g1_dyn`, invokes the training setup entrypoint, and
asserts `RuntimeError` is raised with no optimizer `.step()` and no rollout call
(assert downstream not invoked). Name the guarded function explicitly in Task 4.

### B3 — MM-clip validation should reject terrain-free (constant-Z) pelvis (R9)

**[FACT]** Training Task 5 rejects all-zero or non-finite physical pelvis
(`test_rejects_mm_reference_with_zero_or_missing_body_position`). The design's
whole hypothesis rests on a **terrain-varying** pelvis over the curb (design
lines 12–16, 278–280).

**Impact:** A flat/constant-Z pelvis clip (e.g. a flat-ground clip mislabeled as
curb, or a projection bug) passes the all-zero check yet carries no curb signal,
silently weakening Stage-A supervision and the `grail-curb-low` gate without any
failure.

**Narrow correction:** In the dataset validator, for any clip tagged as the
curb route assert nonzero pelvis-height variance (Task 5 Step 4 already *reports*
"nonzero pelvis-height variance" for the curb clip — promote that from a printed
observation to an enforced rejection with a named test).

---

## Optional improvements (non-blocking)

- **O1 (gatherer clamp parity).** Runtime Task 6 reuses
  `GatherMotionAnchorOrientationMutiFrame`'s clamp/step behavior. Recommend a test
  asserting the new gatherer and the anchor gatherer pick identical frame indices
  for the same play state, so future changes to the hold contract stay in lockstep.
- **O2 (capability negotiation).** GEAR rejects a version *switch* but the design
  wants an older runtime rejected when the root-aware checkpoint is selected.
  Recommend the plan state where the checkpoint↔protocol capability check lives
  (design lines 151–153, 224–226) rather than relying only on version monotonicity.
- **O3 (field-name aliasing).** GEAR endpoint decoding accepts both `body_pos`
  and `body_pos_w` for the v5 field (runtime Task 4 Step 4). Recommend the MM v5
  encoder emit exactly one canonical name to avoid a silent producer/consumer
  drift the parity fixture would not catch.

---

## Summary

- The design's core math (canonical local root transform) is **already present
  and correct** in the pinned GEAR source (F2); the physical-pelvis source, the
  v5 protocol slot, the merger target symbols, and the A0/A1/B gate structure are
  all **verified** (F3, F4, F5, F10). The 640→670 migration is arithmetically
  sound with a confirmed 2048-row first layer (F7).
- Three blocking gaps remain, all within addressed requirements: checked-in
  width comments conflict with the selected deployment artifact (B1), the freeze
  boundary is not proven at the real entrypoint (B2), and the dataset validator
  does not reject terrain-free constant-Z curb clips (B3). Each has an
  exact-path, narrow correction.
- The worker could not inspect the external gitdir or live ONNX inside its
  sandbox. The controller supplied those two identity checks and corrected B1;
  the physical-pelvis coordinate conversion is source-verified in F3.
