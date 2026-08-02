# G1 Terrain Motion Quality Oracle

Date: 2026-08-01

## Decision

The next correction should be **contact-space trajectory composition**.

Do not spend the next iteration tuning search weights, adding a gait-phase-only filter, expanding the corpus indiscriminately, or attaching IK as a final cleanup pass. The exhaustive evidence shows that the current rigid root placement and root/joint-space inertialization destroy contact geometry even when the source action is good.

The replacement should treat entry support and the planned swing-foot landing as constraints of the action representation:

1. fit the source action to current support feet rather than anchoring it only at the root;
2. distribute the planned swing-foot endpoint correction through the source swing interval;
3. decay transition offsets in a contact-preserving space, keeping active stance feet fixed while allowing the root and free joints to absorb the correction;
4. retain the source timing and support sequence, and use no unconstrained post-hoc foot lock.

This is one representation/composition change, not a separate planner plus IK patch. A small IK residual may later implement the contact constraint, but IK is not the algorithmic fix by itself.

## Qualified evaluator

The oracle consumes the frozen six-route multi-horizon baseline and reconstructs source contacts from exact saved `(clip path, source frame)` identities. It captures reset, command-boundary, source-transition, split-height, and elevated direction-change states. For each state it searches every complete one-landing contact action, including exact sagittal mirrors, and keeps independent landing, continuity, combined, and exact-source-successor rankings.

Inventory and state counts:

- source dataset: `build/torch-grail-terrain-feedback-v2`
- baseline: `build/multi-horizon-terrain-skills/qualified-v2-final`
- source clips: 760
- complete contact actions searched per state: 9,004
- frozen difficult states: 92
- capture reasons: 61 source transitions, 13 command boundaries, 6 resets, 6 split-height stances, and 6 elevated direction changes
- native acceptance thresholds: median plus two median absolute deviations over all 9,004 actions
  - stance drift: 0.0150659 m
  - boundary motion: 7.20219

The initial evaluator was run twice and produced the same hash. Its route-wide source classification was then rejected because emitted contact disagreement is not evidence that the selected source action is corrupt. A regression now requires `source` to be determined only from native actions containing the selected source frame. Target-conditioned corpus and placement acceptance additionally require landing, root displacement, and facing agreement.

The corrected evaluator was run twice from raw clips:

- first v2 hash: `c967791f7d7fbaca068dfa3b291ea05cac8f72d33e540e6255ca4436d4924e73`
- repeat v2 hash: `c967791f7d7fbaca068dfa3b291ea05cac8f72d33e540e6255ca4436d4924e73`
- first artifacts: `build/terrain-motion-quality-oracle-v2`
- repeat artifacts: `build/terrain-motion-quality-oracle-v2-repeat`

The non-timing identities, all 92 state IDs, rankings, metrics, classifications, and top-level hashes match.

Focused oracle, report, preview, state, artifact, contact-quality, contact-search, composer, and CLI tests pass. Full repository discovery was attempted but is not green independently of this change: `test_g1_kinematic_contract_ownership` finds a pre-existing duplicated `0x3d23d70a` literal in `g1_ik_runtime.h`, and `test_sonic_artifacts` plus `test_sonic_metrics` cannot import the environment's missing `jsonschema` package. The duplicate discovery run was stopped during the existing 760-clip/640-frame directional integration test after more than 18 minutes because another copy of the same suite was already consuming the cluster. No oracle-related failure appeared before termination.

## Results

No frozen state reached `qualified`, and no state had an acceptable composed candidate.

| Earliest failing boundary | States | Meaning |
|---|---:|---|
| Source | 14 | The selected native source phase itself exceeded robust contact/boundary limits. |
| Corpus/target coverage | 33 | No native complete-step action simultaneously matched support, planned landing, command displacement/facing, and native-quality limits. |
| Rigid placement | 28 | A matching native action existed, but none survived current rigid placement and terrain constraints. |
| Composition | 17 | A good native and placed action existed, but root/joint-space inertialization broke stance contact. |
| Search | 0 | No evidence that a good sustained candidate merely lost the online ranking. |
| Representation | 0 | This label was not reached because earlier boundaries already failed. |
| Qualified | 0 | No candidate survived the complete pipeline. |

The result is not a shortlist artifact. Across the exhaustive frozen-state queries, hard rejection totals were:

- entry support mismatch: 497,276
- entry foot mismatch: 279,211
- landing height: 13,268
- stance height: 13,023
- joint velocity: 2,011
- height deformation: 674
- landing edge margin: 667
- swing penetration: 313
- joint position: 304

The stage-isolation result is the clearest causal evidence. For the 62 states with renderable ranked candidates, the median minimum source-conditioned stance drift was:

| Stage | Median best drift | 90th percentile | Maximum |
|---|---:|---:|---:|
| Native | 0.00260 m | 0.02025 m | 0.02815 m |
| Rigidly placed | 0.00260 m | 0.02025 m | 0.02815 m |
| Inertialized/composed | 0.03856 m | 0.08445 m | 0.10727 m |

Rigid placement preserves the source stance trajectory numerically. Composition increases even the best candidate's median stance motion by roughly 15 times. The contact sheets visually agree: feet that are stationary or nearly stationary in native and placed stages drift during the decaying joint/root offset.

The baseline still contains real steps—11 to 16 complete source-conditioned steps per route—but route-level emitted/source contact agreement is only 0.341 to 0.659. That metric must not be mistaken for source corruption; it is consistent with contact being lost during selection and composition.

## Interpretation

More corpus coverage could eventually help the 33 target-coverage states, but it is not the first controlled intervention. The present corpus already offers 9,004 complete contact phases, and 28 additional states fail because root-only placement cannot align their support geometry. More importantly, all states fail after composition. Adding clips without changing the representation would feed more candidates into the same broken boundary.

Gait phase is useful as a hard compatibility condition, but phase alone cannot specify where the stance foot must remain or where the swing foot must land. The action index already uses complete unload-to-land phases and exact entry support. The remaining defect is geometric.

The previous one-shot support-foot root lock is not a valid substitute. It corrected one frame after composition and worsened slide elsewhere. The contact constraint must be part of placement and offset decay across the entire support interval.

## Next experiment

Implement contact-space trajectory composition behind an offline-only ablation and rerun the same 92 frozen states before changing the online matcher.

Acceptance criteria:

1. exact support order and complete source phase remain unchanged;
2. the entry-foot-error rejection total decreases without increasing stance-height, landing-height, swing-penetration, or edge failures;
3. for candidates accepted in native and placed stages, composed stance drift is at most 0.03 m per action and does not exceed placed drift by more than 0.01 m;
4. source joint/root trajectories remain unchanged when no placement or transition correction is required;
5. deterministic first/repeat hashes match;
6. contact sheets show actual stair touchdowns and stationary stance feet for side approach, diagonal ascent/descent, turning on a tread, reversal, and side exit.

Only after this ablation passes should the corrected kinematics return to the interactive MuJoCo viewer or Sonic tracking.
