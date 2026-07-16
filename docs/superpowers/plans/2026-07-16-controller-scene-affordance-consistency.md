# Controller Scene Affordance Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the graphical controller and headless placement probe consume one certified contact/rest-derived grasp and destination that produces a runtime-owned reversed-pickup preview.

**Architecture:** Preserve the two public controller-scene constructors and place one private certified-source derivation behind both. Move the probe's reviewed rest-grasp and support-fit logic into those shared constructors, delete probe-only corrections, and enrich the existing fail-closed controller diagnostic.

**Tech Stack:** C++17 runtime/controller code, Python `unittest` source-policy checks, GNU Make safe targets, canonical native 25 Hz interaction data.

## Global Constraints

- Do not change selector logic/reason priority, IK, any correction/collision/fit tolerance, timing, native 25 Hz scheduling, terrain, locomotion, pack/evidence schema, or runtime ownership.
- Keep `make_controller_demo_target(const Database&)` and `make_controller_demo_destination_surface(const Database&, const InteractionTarget&)` as the public interfaces.
- The headless probe must use those public constructors unchanged; no probe-only grasp, destination, or support correction remains.
- Certification remains fail-closed and requires accepted preview, nonzero source/selection/fingerprint, and exact preview/candidate IK configuration.
- Never access, stat, hash, execute, modify, stage, or delete repository-root untracked `interaction_query_probe`; safe Make targets may use only `build/task12/interaction_query_probe_safe`.
- Preserve unrelated dirty work. Use only tracked-file status inspection; do not stage or commit before review.
- Do not run `make gate-playable-placement` or any graphical controller until the one implementation/review unit is approved.

---

### Task 1: Share Scene Authoring, Prove Headless Parity, and Enrich Failure Evidence

One implementer owns this coherent change; one fresh reviewer checks the full
diff and verification evidence before any graphical attempt.

**Files:**
- Modify: `interaction_controller_adapter.cpp:1373-1498`
- Modify: `interaction_place_probe.cpp:395-480,620-705`
- Modify: `controller.cpp:7170-7182`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp:4414-4555`
- Modify: `tests/python/test_playable_placement_evidence.py` (`Task8PlacementPolicyTests`)

**Interfaces:**
- Consumes: clip-0 range/phases, active hand, contact hand pose, contact-minus-one object pose, source table, object bounds, existing `evaluate_placement_fit`, and unchanged `RuntimeConfig`.
- Produces: unchanged public `make_controller_demo_target(...)` and `make_controller_demo_destination_surface(...)`; private `CertifiedControllerSceneSource certified_controller_scene_source(const Database&)`; enriched certification text only.

- [ ] **Step 1: Write the regressions first**

Extend `test_demo_target_preserves_object_in_table_transform` to assert:

```cpp
const Transform expected_grasp = compose(
    inverse(rest_object), contact_hand);
assert_transform_near(target.affordances.front().hand_in_object,
                      expected_grasp);
const PlaceAffordance& slot = destination.affordances.front();
assert(evaluate_placement_fit(
    destination, slot, target.object_bounds).accepted);
```

Add these policy tests:

```python
def test_headless_probe_consumes_shared_controller_scene_unchanged(self):
    source = Path("interaction_place_probe.cpp").read_text()
    self.assertNotIn("make_headless_target(", source)
    self.assertNotIn("make_headless_destination_surface(", source)

def test_certification_failure_reports_all_preview_conjuncts(self):
    source = Path("controller.cpp").read_text()
    for token in ("accepted=", "reason=", "source_id=", "selection_id=",
                  "ik_fingerprint=", "preview_ik_exact=",
                  "candidate_ik_exact="):
        self.assertIn(token, source)
```

- [ ] **Step 2: Run RED and record the expected failures**

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
python -m unittest tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests -v
```

Expected RED: the synthetic target grasp does not equal the contact/rest-derived
transform; the probe still contains both private helper names; and the generic
controller error lacks the seven diagnostic labels. No unrelated failure is an
acceptable RED.

Then switch only the probe call sites to the shared constructors and run:

```bash
make interaction_place_probe
./interaction_place_probe resources/g1_interaction --json
```

Expected RED before the shared constructor correction: exit `1` with
`runtime placement preview was rejected: CorrectionLimit`.

- [ ] **Step 3: Implement the minimal shared correction**

In `interaction_controller_adapter.cpp`, add private
`certified_controller_scene_source` to validate and return the first Contact,
its preceding rest object, active hand/contact hand, and source table geometry.
Use it in the target constructor for `inverse(rest_object) * contact_hand` and
in the destination constructor for rest-derived `object_in_surface` and
`support_point_object`. Centralize the existing vertical-clearance correction,
call `evaluate_placement_fit` again, and throw `FormatError` unless accepted.

Delete `make_headless_target` and `make_headless_destination_surface`; construct
the probe target and destination directly with the shared public functions.
In `controller.cpp`, retain the same predicate but build its exception with the
seven tested fields and the existing reason-name mapping. Do not change control
flow or acceptance conditions.

- [ ] **Step 4: Run focused GREEN**

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
python -m unittest tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests -v
make interaction_place_probe
./interaction_place_probe resources/g1_interaction --json
```

Expected: all commands exit `0`; the probe emits one successful JSON record
with `"mode":"reversed_pickup"`, nonzero runtime-owned selection identity,
and no private fixture correction.

- [ ] **Step 5: Run full safe GREEN and review**

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
make gate-place-headless
```

Expected: safe Python/C++ suites, release-fast-math checks, real database
validation, headless place probe, and its validator all pass with no protected
root-probe dependency.

The reviewer checks the spec exclusions, exact RED/GREEN logs, tracked diff,
probe parity, centralized fit validation, and diagnostic-only controller change.
Leave the implementation uncommitted for that review. Only after approval may
a separately authorized single graphical placement gate be scheduled.
