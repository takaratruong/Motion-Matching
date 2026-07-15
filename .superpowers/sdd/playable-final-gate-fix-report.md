# Final playable gate fix report

## Scope and outcome

- Worktree: `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching`
- Branch: `g1-manipulation-motion-matching`
- Starting HEAD: `ecccba436875d9a5d9909a9744b3bafb3d73f81b`
- Implementation commit: `a88e0e765630c90ab9a21239a543830bd9c6657a`
- Result: the exact isolated `gate-playable-interaction` command completed with exit code 0, including the real graphical evidence validator.

The validated full pack under `resources/g1_interaction` was not rebuilt or modified. The protected untracked repo-root query probe was not accessed or changed; verification used only the permitted safe build probe and tracked interaction probe.

## Root causes and fixes

The original failure was the diagnostic five-source demo recipe omitting `--allow-rejections`. The builder correctly rejected `pickup_table__alcohol_0__001` with the closed, reviewed `interaction/no_distinct_lift_phase` code, but the Make recipe would not publish the remaining valid partition. The recipe now opts into the existing reviewed-exclusion path while retaining limit 5, 25 Hz, one held-out object, deterministic selection, validation, and transactional publication.

After that fix exposed the graphical stage, two Carry integration defects prevented the final evidence contract:

1. The controller handoff synchronized the 60 Hz simulation root from a cached 25 Hz layered-Carry pose, freezing live locomotion. Layered Carry now keeps the simulation root live; pre-Carry and recorded Carry retain root synchronization.
2. A Hold-to-layered-Carry pose seam exceeded the unchanged IK request bounds (0.12 m and 25 degrees), and rejection returned the entire stale pose, including its root. Layered Carry now remaps the last-safe pose onto the live root, inertializes non-root channels toward the layered target over bounded small steps, retains the accepted active-arm solution for IK, and backs off the transition step when necessary. Published object continuity is constrained by the existing 0.02 m / 10 degree grasp-drift limits. If every bounded step is rejected, the last-safe non-root pose and grasp are preserved on the live root.

No phase validation, rejection schema, IK request limit, evidence threshold, or validator was weakened.

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

## Exact final gate

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

## Self-review and concerns

- The final tracked implementation diff contains only the Make recipe, Carry/controller integration, and their regressions; temporary diagnostics and exploratory controller changes are absent.
- `git diff --check` passed before the implementation commit.
- The exact final gate used new isolated artifact/evidence paths and did not overwrite prior evidence or the full pack.
- The demo database is intentionally small after the reviewed exclusion and object-disjoint split: four clips survive source validation, while one database clip remains after holding out `alcohol_0`. Both partitions are nonempty and independently validated.
- The successful demo uses layered Carry rather than a recorded Carry range. This is expected and now has direct continuity/rejection coverage plus real graphical evidence.
- The bounded transition constants are internal (`0.50` s and at most 8 backoff attempts); public frozen configuration defaults and IK request limits remain unchanged.
