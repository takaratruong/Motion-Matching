# Affordance Standoff Entry Arc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed straight interaction-waypoint translation with an affordance-sided arc waypoint that preserves the recorded pickup standoff exactly.

**Architecture:** Keep the mapped clip-0 Reach transform as the immutable evidence reference. Derive a clearance chord from retained object geometry and the grasp affordance, rotate the Reach-to-object planar radius by the corresponding positive and negative chord angles, and choose the hand-sided candidate. Ordinary left-stick navigation and one-way braking target that root-only point; matching, IK, collision, attachment, and placement remain unchanged.

**Tech Stack:** C++17 controller geometry, Python `unittest` evidence validation, Raylib/X11 graphical gate, native exact 25 Hz runtime.

## Global Constraints

- Controller, runtime, evidence, and video remain native exact 25 Hz; never add 60 Hz or 6 Hz conversion.
- Preserve the stable Reach limits of `0.15 m` and `20 degrees`, pickup standoff `[0.35 m, 0.45 m]`, matcher correction limits `0.25 m` and `25 degrees`, and the 250-tick approach bound.
- Do not change collision geometry/exemptions, matcher selection/costs, IK, playback, attachment, placement, or the canonical interaction pack.
- Do not initialize, relocate, or write the simulation/displayed root; placement mode continues to provide only live flat-controller snapshots.
- Do not merge or add terrain behavior.
- Never access, stat, hash, execute, modify, stage, or delete the protected repository-root artifact `interaction_query_probe`; use `git status --untracked-files=no`.
- This amendment stays inside the open Task 8 review unit. Do not create an intermediate product commit; the existing Task 8 final verified commit remains the product commit boundary.

---

### Task 1: Derive and Prove the Standoff-Preserving Entry Arc

**Files:**
- Modify: `tests/python/test_playable_placement_evidence.py`
- Modify: `controller.cpp`
- Update after verification: `.superpowers/sdd/task-8-report.md`

**Interfaces:**
- Consumes: stable mapped clip-0 Reach transform, retained source `InteractionTarget`, its single `GraspAffordance`, and ordinary placement auto-demo input state.
- Produces: `make_placement_autodemo_interaction_waypoint(Transform, const InteractionTarget&, float&)`, immutable `interaction_object_position`, independently recomputable arc evidence, and a collision-free live entry attempt without widening any bound.

- [ ] **Step 1: Change the evidence fixtures and tests first**

Add `interaction_object_position` to `REQUIRED_FIELDS` and every synthetic row. Replace the translated valid fixture with an independently constructed arc fixture. Add tests named:

```python
def _planar_distance(left, right):
    return math.hypot(left[0] - right[0], left[2] - right[2])


def _hand_sided_lateral(row):
    approach = pickup._quaternion_rotate(
        row["interaction_object_rotation"],
        row["interaction_approach_direction_object"],
    )
    length = math.hypot(approach[0], approach[2])
    right = [approach[2] / length, 0.0, -approach[0] / length]
    return [row["interaction_hand_side"] * value for value in right]


def test_interaction_waypoint_preserves_chord_standoff_and_hand_side(self):
    records = _valid_records()
    validate_evidence(records)
    row = records[0]
    assert math.isclose(
        _planar_distance(row["interaction_waypoint_position"],
                         row["reach_waypoint_position"]),
        row["interaction_lateral_offset_m"], abs_tol=2e-6)
    assert math.isclose(
        _planar_distance(row["interaction_waypoint_position"],
                         row["interaction_object_position"]),
        _planar_distance(row["reach_waypoint_position"],
                         row["interaction_object_position"]), abs_tol=2e-6)

def test_interaction_waypoint_rejects_straight_translation(self):
    records = _valid_records()
    row = records[0]
    lateral = _hand_sided_lateral(row)
    translated = [
        value + row["interaction_lateral_offset_m"] * axis
        for value, axis in zip(row["reach_waypoint_position"], lateral)
    ]
    for record in records:
        record["interaction_waypoint_position"] = translated.copy()
    with self.assertRaisesRegex(
        PlacementEvidenceValidationError, "standoff|arc"
    ):
        validate_evidence(records)
```

Extend the rotation/hand test to construct common-yaw-rotated right- and left-hand rows and assert that they select opposite candidates while preserving the same chord and object radius. Add corruption cases for object position, chord, dimensions, rotation, approach, clearance, hand sign, wrong arc candidate, and translated point. Extend the static policy test to require `std::asin`, both `+theta`/`-theta` candidates, object-centered construction, hand-sided scores, and explicit invariant checks.

- [ ] **Step 2: Run RED and verify the failure is about missing arc behavior**

Run:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
python -m unittest \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_interaction_waypoint_preserves_chord_standoff_and_hand_side \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_interaction_waypoint_rejects_straight_translation \
  tests.python.test_playable_placement_evidence.PlacementEvidenceValidatorUnitTests.test_interaction_waypoint_recomputes_rotation_and_hand_side \
  tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_terminal_pickup_waypoint_is_affordance_derived_and_root_only -v
```

Expected: FAIL because the current helper/validator uses a straight translation, does not preserve object-centered standoff, and does not log `interaction_object_position`. A syntax/import error is not an acceptable RED result.

- [ ] **Step 3: Implement the minimal C++ arc derivation**

Keep the public helper root-only:

```cpp
interaction::Transform make_placement_autodemo_interaction_waypoint(
    interaction::Transform reach_waypoint,
    const interaction::InteractionTarget& target,
    float& clearance_chord_m);
```

Retain the current approach/lateral and oriented-box support calculation, including exactly `0.001F` clearance epsilon. Replace only the final translation with:

```cpp
const vec3 object = target.object_world.position;
const vec3 radius(
    reach_waypoint.position.x - object.x,
    0.0F,
    reach_waypoint.position.z - object.z);
const float standoff_m = length(radius);
// Fail closed unless finite, standoff is [0.35, 0.45], chord is
// positive, chord <= 0.15, and chord <= 2 * standoff_m.
const float theta = 2.0F * std::asin(
    clearance_chord_m / (2.0F * standoff_m));
const vec3 plus_radius = quat_mul_vec3(
    quat_from_angle_axis(+theta, vec3(0.0F, 1.0F, 0.0F)), radius);
const vec3 minus_radius = quat_mul_vec3(
    quat_from_angle_axis(-theta, vec3(0.0F, 1.0F, 0.0F)), radius);
vec3 plus(object.x + plus_radius.x, reach_waypoint.position.y,
          object.z + plus_radius.z);
vec3 minus(object.x + minus_radius.x, reach_waypoint.position.y,
           object.z + minus_radius.z);
const vec3 hand_lateral = hand_sign * right_of_approach;
const float plus_score = dot(plus - reach_waypoint.position, hand_lateral);
const float minus_score = dot(minus - reach_waypoint.position, hand_lateral);
reach_waypoint.position = plus_score > minus_score
    ? plus
    : minus_score > plus_score
        ? minus
        : affordance.hand == interaction::Hand::Right ? plus : minus;
```

Fail closed on non-finite inputs/results, invalid dimensions/hand/approach, zero or out-of-band standoff, invalid chord, or failed invariants. Verify within `2.0e-5F` that planar `|P-R| == chord`, planar `|P-O| == |R-O|`, `P.y == R.y`, and rotation is unchanged. Do not clamp or retry.

Pass `state.interaction_waypoint.position` to the existing navigation and braking seams. Stable Reach/yaw evidence remains measured against `state.reach_waypoint`.

- [ ] **Step 4: Log and independently validate the immutable geometry**

In `write_placement_autodemo_record`, emit:

```cpp
"interaction_object_position": source_target.object_world.position
```

alongside the existing dimensions, rotation, approach, clearance, hand sign, chord, Reach transform, and interaction waypoint. Python `_expected_interaction_waypoint(record)` must independently implement the complete construction:

```python
def _expected_interaction_waypoint(record):
    approach = pickup._quaternion_rotate(
        record["interaction_object_rotation"],
        record["interaction_approach_direction_object"],
    )
    approach_length = math.hypot(approach[0], approach[2])
    approach = [approach[0] / approach_length, 0.0,
                approach[2] / approach_length]
    unsigned_lateral = [approach[2], 0.0, -approach[0]]
    lateral_object = pickup._quaternion_rotate(
        pickup._quaternion_inverse(record["interaction_object_rotation"]),
        unsigned_lateral,
    )
    support = 0.5 * sum(
        abs(axis) * dimension
        for axis, dimension in zip(
            lateral_object, record["interaction_object_dimensions"]
        )
    )
    chord = support + record["interaction_clearance_radius_m"] + 0.001
    reach = record["reach_waypoint_position"]
    object_position = record["interaction_object_position"]
    radius_x = reach[0] - object_position[0]
    radius_z = reach[2] - object_position[2]
    distance = math.hypot(radius_x, radius_z)
    theta = 2.0 * math.asin(chord / (2.0 * distance))

    def candidate(angle):
        cosine, sine = math.cos(angle), math.sin(angle)
        return [
            object_position[0] + cosine * radius_x + sine * radius_z,
            reach[1],
            object_position[2] - sine * radius_x + cosine * radius_z,
        ]

    plus, minus = candidate(+theta), candidate(-theta)
    hand_lateral = [
        record["interaction_hand_side"] * value
        for value in unsigned_lateral
    ]
    plus_score = sum(
        (value - origin) * axis
        for value, origin, axis in zip(plus, reach, hand_lateral)
    )
    minus_score = sum(
        (value - origin) * axis
        for value, origin, axis in zip(minus, reach, hand_lateral)
    )
    if plus_score > minus_score:
        expected = plus
    elif minus_score > plus_score:
        expected = minus
    else:
        expected = plus if record["interaction_hand_side"] == 1 else minus
    return expected, chord
```

It must recompute support/chord before the angle, apply the specified exact-tie hand rule, reject changed immutable inputs, and independently verify chord, preserved standoff, stable bounds, height, and rotation. It must never trust reported distance/error scalars as proof of the arc.

- [ ] **Step 5: Run focused GREEN, policy, compile, and unchanged collision regressions**

Run:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
python -m unittest tests.python.test_playable_placement_evidence -v

make build/tests/test_interaction_runtime
./build/tests/test_interaction_runtime
make controller
git diff --check
git status --short --untracked-files=no
```

Expected: every placement validator/policy test passes; runtime collision tests, including table and precontact object crossings, remain green without edits; controller compiles; diff check is empty. Remove all temporary `fprintf`/deadline/tail diagnostics once the behavioral cause is verified.

- [ ] **Step 6: Run one real native-25-Hz acceptance gate**

Run:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-placement
```

Expected: exit 0; atomic JSONL/PNG are published; the live default-spawn path has at least 25 approach ticks and 2 m displacement; one-way braking remains zero-input; five settled rows remain in the unchanged standoff/Reach/yaw/speed bounds; pickup reaches Carry without `BlockedPath`; ordinary Carry staging, Place, release, and seven-frame handoff complete. If this fails, record the exact state/result/reason and return to systematic root-cause investigation rather than tuning geometry.

- [ ] **Step 7: Record the amendment result without an intermediate product commit**

Append the RED/GREEN commands, exact outputs, derived chord/radius/selected side, graphical result, evidence paths, and any concerns to `.superpowers/sdd/task-8-report.md`. Leave product files for the existing Task 8 final verification, review, visual gate, and commit boundary.
