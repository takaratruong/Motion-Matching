# Final playable gate fix report

## Scope and outcome

- Worktree: `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching`
- Branch: `g1-manipulation-motion-matching`
- Starting HEAD: `ecccba436875d9a5d9909a9744b3bafb3d73f81b`
- Original implementation commit: `a88e0e765630c90ab9a21239a543830bd9c6657a`
- Correction design commits: `4a11549` and `9ff666a`
- Unconditional Carry seam commit: `a695476`
- Fast-math validation commit: `1a32ec5`
- Result: the corrected exact isolated `gate-playable-interaction` command completed with exit code 0, including the real graphical evidence validator.

The validated full pack under `resources/g1_interaction` was not rebuilt or modified. The protected untracked repo-root query probe was not accessed or changed; verification used only the permitted safe build probe and tracked interaction probe.

## Root causes and fixes

The original failure was the diagnostic five-source demo recipe omitting `--allow-rejections`. The builder correctly rejected `pickup_table__alcohol_0__001` with the closed, reviewed `interaction/no_distinct_lift_phase` code, but the Make recipe would not publish the remaining valid partition. The recipe now opts into the existing reviewed-exclusion path while retaining limit 5, 25 Hz, one held-out object, deterministic selection, validation, and transactional publication.

After that fix exposed the graphical stage, two Carry integration defects prevented the final evidence contract:

1. The controller handoff synchronized the 60 Hz simulation root from a cached 25 Hz layered-Carry pose, freezing live locomotion. Layered Carry now keeps the simulation root live; pre-Carry and recorded Carry retain root synchronization.
2. A Hold-to-layered-Carry pose seam exceeded the unchanged IK request bounds (0.12 m and 25 degrees), and rejection returned the entire stale pose, including its root. Layered Carry now remaps the last-safe pose onto the live root, inertializes non-root channels toward the layered target over bounded small steps, retains the accepted active-arm solution for IK, and backs off the transition step when necessary. Published object continuity is constrained by the existing 0.02 m / 10 degree grasp-drift limits. If every bounded step is rejected, the last-safe non-root pose and grasp are preserved on the live root.

No phase validation, rejection schema, IK request limit, evidence threshold, or validator was weakened.

Correction review found two further issues after the original implementation:

3. The original Hold-to-Carry smoothing ran only when full layered IK rejected. An IK-feasible layered candidate, an initial recorded selection, or a recorded retarget could still publish its complete non-root pose in one tick. Carry now has one publication-stage seam for every layered/recorded candidate. It uses a 0.50 s cumulative 0-to-1 transition, keeps all root channels live, distinguishes contiguous recorded playback from a new selection epoch, restarts on explicit mode/range/retarget switches, and retains the accepted active arm through bounded fallback attempts. The completion tick stays on the seam path before direct publication, preventing an active-arm boundary snap.
4. Carry used `std::isfinite` in a controller compiled with `-ffast-math`. Under the real optimized flags, NaN Hold and NaN `dt` inputs could pass those checks. Carry now uses IEEE-754 exponent-bit checks for float and double values. A dedicated always-active `-O3 -DNDEBUG -ffast-math` binary validates both rejection paths and is part of `test-interaction-safe`.

## TDD evidence

### Demo recipe RED

Command:

```sh
python -m unittest \
  tests.python.test_interaction_gate1.Gate1MakefileTests.test_demo_pack_permits_reviewed_exclusions
```

Before the Make change, the new test failed because `make -Bn demo-interaction-pack` did not contain `--allow-rejections`:

```text
AssertionError: '--allow-rejections' not found in 'python -m resources.build_g1_interaction_database ... --heldout-count 1\n...'
Ran 1 test
FAILED (failures=1)
```

### Layered-Carry controller handoff RED

The new `test_frame_handoff_keeps_layered_carry_simulation_root_live` initially failed against the old unconditional root synchronization:

```text
test_interaction_controller_adapter.cpp:1043:
Assertion `!layered.synchronize_simulation_root' failed.
```

### Hold-to-Carry continuity RED

The new carry test constructs a valid live pose deliberately different from the final Hold pose, then requires first-tick live-root ownership, a root-relative object seam within the existing contact tolerance, repeated small-step convergence, more than 0.20 m carried motion, and a genuine last-safe rejection path. Before the Carry change it failed at the first live-root assertion:

```text
test_interaction_carry.cpp:733:
Assertion `near(root_world(first), root_world(locomotion.pose), 2.0e-5F)' failed.
```

### Focused GREEN

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
make build/tests/test_interaction_controller_adapter && \
  build/tests/test_interaction_controller_adapter
python -m unittest tests.python.test_interaction_gate1.Gate1MakefileTests
```

All commands exited 0. The Make policy class ran 2 tests and reported `OK`. `git diff --check` also exited 0.

## Original exact final gate (superseded evidence run)

Fresh isolated command:

```sh
DISPLAY=:1 \
INTERACTION_DEMO_PACK=build/task12/final-demo-pack-ecccba4-warm-20260715 \
PLAYABLE_EVIDENCE_DIR=playable-evidence/final-ecccba4-warm-20260715 \
make gate-playable-interaction
```

Result: exit code 0.

- Safe Python suite: 225 tests passed, 2 skipped.
- Release-fast-math target validation: passed.
- Controller: rebuilt successfully; only pre-existing third-party compiler warnings were emitted.
- Evidence suite: 44 tests passed, including `test_real_playable_evidence`.

## Isolated artifact measurements

Builder and validator output:

```text
BUILT schema=1 fps=25 clips=1 frames=250 heldout_objects=1 rejected=1
VALID schema=1 fps=25 bones=31 features=71 clips=1 frames=250 heldout_objects=1
```

- Source clips: 5
- Included clips before split: 4
- Rejected clips: 1
- Rejection histogram: `no_distinct_lift_phase: 1`
- Exact rejection: sequence `pickup_table__alcohol_0__001`, object `alcohol_0`, stage `interaction`, code `no_distinct_lift_phase`
- Database sequence: `pickup_table__alcohol_10__000`
- Database object: `alcohol_10`
- Held-out object: `alcohol_0`
- Database clips / frames: 1 / 250
- Target FPS: 25
- Database SHA-256: `19ad444a45ac46bf0f636a48fd16852d26c096cbfc83860a4bab07308a89582d`
- Feature SHA-256: `d549daa6a08c1b128e1b8a42cb4292ccc5eda9fd488701b91f05cddca5135a70`
- Numeric validation bounds:
  - duration max error: 0.0 s
  - FK max error: `2.2552436012197768e-07` m
  - FK rotation max error: `0.09278683558103876` degrees
  - quaternion norm max error: `1.1920928955078125e-07`

Tracked `interaction_probe` output:

```text
bone_count=31 clip_count=1 feature_count=71 frame_count=250
first_source_frame=0 last_source_frame=249
max_quaternion_norm_error=4.97224599e-08
phase_counts=[114,25,14,23,74]
```

Runtime probe output:

```text
attach_frame=139 attached=true carry_mode=layered
carry_root_displacement_m=1.00000012 held_time_seconds=1
selected_clip=0 final_result=Succeeded final_reason=None
state_sequence=[Locomotion,Preflight,Align,PickupReplay,Hold,Carry]
```

## Graphical evidence measurements

- Log: `playable-evidence/final-ecccba4-warm-20260715/pickup.jsonl`
- Screenshot: `playable-evidence/final-ecccba4-warm-20260715/pickup.png`
- Records: 354
- Screenshot size: 116,075 bytes; visually inspected with Carry/Succeeded, attached Held object, and layered mode visible.
- State order: `Locomotion -> Preflight -> Align -> PickupReplay -> Hold -> Carry -> Locomotion(Reset)`
- Carry records: 152, all validated as Held, attached, successful, and pose-owning.
- Forward commands: exactly 150, followed by the required reset record.
- First Carry: render frame 201, runtime tick 85.
- Last Carry: render frame 352, runtime tick 147.
- Final Carry root displacement field: 1.756290 m.
- First-to-last Carry root displacement: 1.7562900338 m.
- First-to-last Carry object displacement: 2.7277622495 m.
- Hold-to-first-Carry root delta: 0.0006535580 m.
- Hold-to-first-Carry object delta: 0.0050708492 m.
- Hold-to-first-Carry object Y delta: 0.004864 m.
- Object Y range across all Carry records: 0.012767 m, below the existing 0.02 m grasp-drift tolerance and with no former large vertical excursion.
- Final reset: render frame 353, `result=Reset`, `reason=Reset`, object Free and detached.

## Correction TDD and optimized-validation evidence

The review fixtures first demonstrated all three direct-publication defects before production changes:

```text
test_interaction_carry.cpp:733: default layered LeftHipPitch snapped to the
complete +0.90 rad live target instead of alpha 0.08
test_interaction_carry.cpp:1119: initial recorded LeftHipPitch published the
complete recorded target instead of alpha 0.08
test_interaction_carry.cpp:1226: recorded range switch published the complete
new range instead of restarting at alpha 0.20
```

The default layered case mutates only the IK-feasible inactive `LeftHipPitch`, so it proves the seam is unconditional rather than an IK-rejection artifact. The recorded tests separately prove that same-range contiguous frame progression does not restart the epoch, while an explicit range/selection switch does.

The Make policy RED reported two expected failures: no `test-interaction-carry-release-fast-math` rule and no Carry optimized binary in `test-interaction-safe`. After adding the always-active test and actual optimized recipe, the production RED was:

```text
g++ ... -O3 -DNDEBUG -ffast-math \
  tests/cpp/test_interaction_carry_fast_math.cpp interaction_carry.cpp ... \
  -o build/tests/test_interaction_carry_release_fast_math
build/tests/test_interaction_carry_release_fast_math
make: *** [Makefile:200: test-interaction-carry-release-fast-math] Error 1
```

This was the unmodified `std::isfinite` implementation accepting at least one required non-finite case. After the bit-level finite change, the same optimized command exited 0. The normal Carry binary also exited 0, preserving its inclusive-threshold coverage, and all four `ReleaseFastMathMakefileTests` reported `OK`.

Focused correction GREEN:

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
make build/tests/test_interaction_controller_adapter && \
  build/tests/test_interaction_controller_adapter
make test-interaction-carry-release-fast-math
python -m unittest \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests
```

All commands exited 0. The seam implementation is commit `a695476`; the optimized-validation implementation is commit `1a32ec5`.

## Corrected safe and exact graphical gates

Static and safe verification:

```sh
git diff --check
make test-interaction-safe
```

Result: exit code 0. The safe Python suite ran 227 tests with 2 expected skips; all normal C++ binaries and both release-fast-math validation binaries passed.

Fresh commit-qualified graphical command:

```sh
DISPLAY=:1 \
INTERACTION_DEMO_PACK=build/task12/final-demo-pack-carry-seam-fastmath-1a32ec5-20260715 \
PLAYABLE_EVIDENCE_DIR=playable-evidence/final-carry-seam-fastmath-1a32ec5-20260715 \
make gate-playable-interaction
```

Result: exit code 0. The gate rebuilt and validated the pack, rebuilt the optimized controller, ran both tracked probes, collected real X11/OpenGL evidence, and ran 44 evidence tests with `test_real_playable_evidence` passing. Only pre-existing third-party compiler warnings were emitted.

Corrected builder and validator output:

```text
BUILT schema=1 fps=25 clips=1 frames=250 heldout_objects=1 rejected=1
VALID schema=1 fps=25 bones=31 features=71 clips=1 frames=250 heldout_objects=1
db_sha256=19ad444a45ac46bf0f636a48fd16852d26c096cbfc83860a4bab07308a89582d
feature_sha256=d549daa6a08c1b128e1b8a42cb4292ccc5eda9fd488701b91f05cddca5135a70
```

Corrected isolated artifact measurements:

- Manifest commit: `1a32ec5eb420e0c2917cec88ee784ed80c0ce93c`
- Source / included-before-split / rejected clips: 5 / 4 / 1
- Rejection histogram: `no_distinct_lift_phase: 1`
- Exact rejection: sequence `pickup_table__alcohol_0__001`, object `alcohol_0`, stage `interaction`, code `no_distinct_lift_phase`
- Database sequence/object: `pickup_table__alcohol_10__000` / `alcohol_10`
- Held-out object: `alcohol_0`
- Database clips / frames / feature dimensions: 1 / 250 / 71
- Database SHA-256: `19ad444a45ac46bf0f636a48fd16852d26c096cbfc83860a4bab07308a89582d`
- Feature SHA-256: `d549daa6a08c1b128e1b8a42cb4292ccc5eda9fd488701b91f05cddca5135a70`
- Numeric bounds: duration error 0.0 s; FK error `2.2552436012197768e-07` m; FK rotation error `0.09278683558103876` degrees; quaternion norm error `1.1920928955078125e-07`

Corrected tracked probe output:

```text
bone_count=31 clip_count=1 feature_count=71 frame_count=250
first_source_frame=0 last_source_frame=249
max_quaternion_norm_error=4.97224599e-08
phase_counts=[114,25,14,23,74]
```

Corrected runtime probe output:

```text
attach_frame=139 attached=true carry_mode=layered
carry_root_displacement_m=1.00000012 held_time_seconds=1
selected_clip=0 final_result=Succeeded final_reason=None
state_sequence=[Locomotion,Preflight,Align,PickupReplay,Hold,Carry]
```

## Corrected graphical evidence measurements

- Log: `playable-evidence/final-carry-seam-fastmath-1a32ec5-20260715/pickup.jsonl`
- Screenshot: `playable-evidence/final-carry-seam-fastmath-1a32ec5-20260715/pickup.png`
- Records: 354
- Log SHA-256: `12b55e41c7d30b4a3f67adb255cbc70b09f492bb984697b933f42afffbb82e70`
- Screenshot SHA-256: `c94717fbd7118709fcf5ecee5b6d9ba786e6979bf36c9102d166e68b45c811ae`
- Screenshot size: 116,319 bytes
- Visual inspection: the fresh screenshot shows `state=Carry`, `result=Succeeded`, `reason=None`, `owns=1`, `attached=1`, the Held object at the hand, and `carry=layered`
- State order: `Locomotion -> Preflight -> Align -> PickupReplay -> Hold -> Carry -> Locomotion(Reset)`
- State counts: Locomotion 32, Preflight 2, Align 32, PickupReplay 117, Hold 19, Carry 152
- Carry records: 152; all 152 use layered mode and validate as Held, attached, successful, and pose-owning
- Forward commands: exactly 150, numbered 0 through 149
- First Carry: render frame 201, runtime tick 85
- Last Carry: render frame 352, runtime tick 147
- Final Carry root displacement field: 1.756290 m
- First-to-last Carry root displacement: 1.7562900338 m
- First-to-last Carry object horizontal displacement: 2.7697128099 m; 3D displacement: 2.7737493793 m
- Hold-to-first-Carry root delta: 0.0006535580 m
- Hold-to-first-Carry object delta: 0.0050708492 m; horizontal component: 0.0014335327 m
- Hold-to-first-Carry object Y delta: 0.0048640000 m
- Object Y min / max / range across Carry: 1.062627 / 1.252193 / 0.189566 m
- Maximum adjacent Carry object Y change: 0.019415 m, within the existing 0.02 m per-update continuity bound
- Maximum adjacent Carry object 3D step: 0.1327102922 m between render frames 215 and 216
- Maximum adjacent Carry root step: 0.0553559078 m between render frames 292 and 293
- Final reset: render frame 353, runtime tick 148, `result=Reset`, `reason=Reset`, object Free and detached

The cumulative 0.189566 m object-height change is gradual and can represent the Hold-to-carry-height adjustment; it is not a one-frame Hold-to-Carry teleport. The first seam delta and each adjacent Y change remain bounded. The larger horizontal/3D object steps reflect the existing 25 Hz held-pose/object presentation against 60 Hz rendering and are recorded below as follow-up work rather than hidden by changing this gate's thresholds.

## Self-review and concerns

- The final tracked implementation diff contains only the Make recipe, Carry/controller integration, and their regressions; temporary diagnostics and exploratory controller changes are absent.
- `git diff --check` passed before the implementation commit.
- The exact final gate used new isolated artifact/evidence paths and did not overwrite prior evidence or the full pack.
- The demo database is intentionally small after the reviewed exclusion and object-disjoint split: four clips survive source validation, while one database clip remains after holding out `alcohol_0`. Both partitions are nonempty and independently validated.
- The successful demo uses layered Carry rather than a recorded Carry range. This is expected and now has direct continuity/rejection coverage plus real graphical evidence.
- The bounded transition constants are internal (`0.50` s and at most 8 backoff attempts); public frozen configuration defaults and IK request limits remain unchanged.
- The corrected implementation applies the publication seam to IK-feasible layered Carry, initial recorded Carry, and every recorded selection epoch, not only the old rejection path. Temporary diagnostics are absent.
- The dedicated Carry fast-math binary uses throwing checks under `NDEBUG`; normal `assert` elision cannot turn the safety validation into a false pass.
- The 25 Hz held-pose/object presentation still produces visible 60 Hz stutter: the corrected evidence measured a 0.1327102922 m maximum adjacent object step while the maximum adjacent root step was 0.0553559078 m. A separate 60 Hz visual handoff/continuity task should address interpolation without weakening Carry attachment, displacement, or evidence requirements.
