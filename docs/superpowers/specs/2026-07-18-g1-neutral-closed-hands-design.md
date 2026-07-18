# G1 Neutral Closed Hands Design

## Purpose

Make both Dex3 hands hold a relaxed closed posture during motion-matching locomotion, while preserving an explicit per-chunk override for later grasping motions. The posture must be commanded through the real GEAR-to-MuJoCo control path and visible in physics-derived video; changing only initial state or replay rendering is not acceptable.

This change also corrects an actuator-routing defect discovered during visual review. The run-local robot XML lists all 29 body actuators before the 14 hand actuators, while the pinned GEAR MuJoCo simulator routes torques with joint-traversal indices. MuJoCo joint traversal interleaves the left hand between the left and right arms. Consequently, the current scene sends right-arm commands to left-hand actuators and left-hand commands to right-arm actuators. Walking remains genuine MuJoCo physics and the lower-body route is aligned, but the full robot control path is not correct until this mismatch is removed.

## Scope

The implementation will:

- normalize actuator order in the generated run-local MuJoCo XML so it matches MuJoCo joint traversal;
- publish an explicit relaxed-fist target for all seven joints of each hand on every ZMQ pose chunk;
- allow a future grasping source to replace either hand target without changing the motion-matching body interface;
- produce close-up visual evidence before running the longer locomotion qualification; and
- retain the official source robot and pinned GEAR/SONIC checkouts unchanged.

The implementation will not add grasp planning, object interaction, learned hand motion, hardware deployment, or a renderer-only pose correction. The motion-matching server remains responsible for the 29 body joints only.

## Control-path correction

The scene builder operates on the copied run-local robot XML. It loads the model structure, determines the MuJoCo traversal order of actuated joints, and reorders existing `<motor>` elements by their referenced joint. All motor attributes are preserved; only element order changes.

Normalization fails closed if an actuator references an unknown joint, if a joint has zero or multiple motors where one is expected, or if the loaded model does not produce a one-to-one actuator-to-joint mapping. The original robot XML is never modified. The normalized scene and its SHA-256 identity are recorded with run provenance.

After normalization, actuator slot `joint_id - 1` names the motor for that one-degree-of-freedom joint, which is the invariant assumed by the pinned `BaseSimulator`. The implementation verifies this invariant from the fully loaded run-local model rather than inferring it from XML text alone.

## Neutral hand profile

The default posture is the midpoint-based relaxed close already defined by the official Dex3 `close()` behavior, rather than the full-limit fallback used when hand fields are absent. In GEAR command order, the targets are:

| Hand | Joint order | Target radians |
| --- | --- | --- |
| Left | thumb0, thumb1, thumb2, index0, index1, middle0, middle1 | 0.000, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875 |
| Right | thumb0, thumb1, thumb2, index0, index1, middle0, middle1 | 0.000, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875 |

The implementation derives these targets from named joint limits/constants where available and tests the resulting vectors, instead of relying on positional literals at multiple call sites. Every value must be finite, within the corresponding joint range, and exactly seven elements long.

## Transport and future grasp override

The body target and hand targets remain separate until the existing GEAR pose message is assembled:

```text
motion matching: 29 body joints ---+
                                     +--> ZMQ pose --> GEAR --> DDS --> MuJoCo physics
neutral/grasp source: 7 + 7 hands --+
```

The publisher includes `left_hand_joints` and `right_hand_joints` descriptors and payloads for every emitted chunk. With no grasp source, both fields use the neutral profile. A caller may supply a validated seven-value override for either hand on a chunk; the other hand continues using neutral. An override is data, not a persistent hidden mode, so removing it deterministically restores neutral on the next chunk.

Run commands, summaries, and evidence identify the selected hand profile and hash the commanded vectors. This makes the posture explicit and auditable rather than depending on GEAR's missing-field fallback. Existing body-motion payload values are not reordered or altered by the hand addition.

## Visual-first development order

The first executable check is a short, close-up MuJoCo physics render that clearly shows both hands, both forearms, and finger articulation. It must make asymmetry, saturation, and arm/hand cross-wiring visually obvious. The image or clip is inspected before spending time on the full walking run.

Once that view passes, a short locomotion run uses the same normalized scene and explicit hand commands. Its video must include at least one view where both hands are readable while the robot walks. Formal automated qualification follows the visual checks; it does not substitute for them.

## Verification

The implementation is accepted only when all of the following hold:

- the loaded model has exactly one actuator per expected actuated joint and every actuator slot resolves to the joint assumed by `BaseSimulator`;
- a distinct sentinel command for every body and hand joint reaches the intended actuator, including the previous left-hand/right-arm boundary;
- ZMQ encode/decode tests preserve both seven-value hand vectors and reject missing-width, non-finite, or out-of-range overrides;
- a close-up physics render is visually inspected and shows two relaxed, substantially symmetric fists with no flailing, hard-limit pinning, or unintended arm response;
- after settling, each hand joint's median absolute tracking error over the final simulated second is at most 0.20 radians, and no joint is at a limit unless its commanded target is at that limit;
- the locomotion physics run remains fall-free, with minimum root height at least 0.65 m, minimum pelvis-up dot product at least 0.90, and stopped-command drift at most 0.25 m;
- the recorded 29-joint body pose stream still matches the accepted motion-matching command stream frame-for-frame; and
- the official source XML and pinned external checkouts remain byte-identical.

The sentinel routing test is mandatory because ordinary gait metrics did not expose the actuator mismatch. Visual inspection is mandatory because the earlier automated suite also allowed an obviously wrong hand motion to pass.

## Failure and rollback behavior

Scene creation stops before simulation when actuator normalization cannot prove its mapping. Hand publishing stops before transport when a profile or override is invalid. Neither case silently falls back to the current mismatched routing or to GEAR's implicit full-close posture.

Each run receives its own normalized XML, leaving the source asset available for comparison and rollback. The neutral-hand default can be removed from a future run without changing motion-matching output, but the actuator-order correction remains a required simulator compatibility invariant.
