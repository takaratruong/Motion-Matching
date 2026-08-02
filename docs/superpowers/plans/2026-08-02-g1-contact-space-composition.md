# G1 Contact-Space Trajectory Composition Plan

## Goal

Replace root-only placement and unconstrained root/joint inertialization with one offline contact-space composition path, then evaluate it on the unchanged 92-state terrain quality corpus before exposing it to the interactive matcher.

## Constraints

- Keep source action timing, joint ordering, and support sequence unchanged.
- Use the planned first foothold already frozen by the quality oracle.
- Do not change online search weights or add corpus clips during this ablation.
- Do not treat a terminal IK foot lock as the solution; contact constraints apply over the full support interval.
- Preserve existing unrelated landing-bridge and FK working-tree changes.

## Task 1: Contact-anchored rigid placement

- Add `torch_terrain_contact_composition.py` and focused tests.
- Fit action entry support feet to current support feet.
- For single support, retain current root yaw and translate the full action so the stance foot matches exactly in XYZ.
- For double support, solve the deterministic planar least-squares rigid fit and mean vertical translation; report the residual rather than hiding incompatible foot separation.
- Verify unchanged pairwise action geometry and strictly lower entry support error than root anchoring on a displaced synthetic state.

## Task 2: Contact target trajectory

- Construct immutable per-frame foot targets from the source support mask.
- Hold each supported foot at its contact-onset world anchor.
- During swing, distribute the planned landing displacement with a zero-slope smoothstep from unload to touchdown.
- Preserve the source swing trajectory exactly when the desired endpoint already matches.

## Task 3: Contact-projected inertialization

- Build the ordinary 50 Hz inertialized root/joint trajectory first.
- Project every supported frame back to its contact target with authoritative G1 leg IK; project the swing foot only along the planned warp.
- Record IK residual, joint deformation, root deformation, stance drift, and failure reason for every frame.
- Fail closed on unreachable frames; do not silently return the unprojected pose.

## Task 4: Frozen-state ablation

- Add contact-anchored placement and projected composition as additional oracle stages without altering v2 evidence.
- Re-evaluate all 9,004 actions at all 92 frozen states.
- Compare entry-foot rejection counts and native/placed/composed stance drift against v2.
- Render contact sheets from saved qpos for the same ranked identities.

## Task 5: Decision

- Require deterministic first/repeat hashes.
- Accept only if entry-foot errors decrease without terrain-safety regressions and composed stance drift is at most 0.03 m per action and within 0.01 m of placed drift.
- If contact projection passes but coverage remains low, the next isolated experiment is swing-endpoint action deformation; if projection fails, reject this representation and do not integrate it online.
