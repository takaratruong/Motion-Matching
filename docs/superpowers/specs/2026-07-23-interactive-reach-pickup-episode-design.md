# Interactive Reach Pickup Episode

Date: 2026-07-23

Status: approved for implementation planning

## Goal

Build a small flat viewer that proves the current grasp-relative reach search can
drive a complete playable pickup:

1. the user moves the G1 with WASD;
2. pressing F freezes one object and grasp request;
3. motion matching chooses a reusable left- or right-hand reach;
4. ordinary motion-matched locomotion walks to that reach's entry;
5. a deterministic bridge hands locomotion to the corrected reach;
6. the object attaches at contact;
7. the pose blends into hand-correct carry locomotion; and
8. WASD control resumes while the object remains attached.

This milestone is the deterministic baseline. Diffusion will later replace only
the transition bridge, after the full episode can be tested reliably.

## Scope

The first viewer contains one movable box on flat ground and one known 6-DoF
grasp. It supports pickup and controlled carrying. It does not yet support
placement, live GraspNet inference, obstacle navigation, or diffusion.

While no attempt is active, arrow keys translate the box in the ground plane,
U/J move it up and down, and Q/E rotate it about world up. Moving the box does
not search; F freezes its current pose and starts exactly one search.

The viewer must use the G1 mesh and the same pose contract as the existing flat
coverage viewer. It must not load the terrain viewer or terrain scene.

## Selected Architecture

Create a dedicated `g1_interaction_episode_viewer` instead of adding runtime
control to the coverage-analysis viewer or changing the production controller.
The executable reuses existing modules for:

- flat G1 locomotion and native pose publication;
- contact-anchored bilateral reach search;
- posture-aware IK and swept body/object collision checks;
- object attachment; and
- left/right carry pose generation.

Keeping the viewer separate makes failures attributable to the interaction
handoffs and avoids destabilizing the production controller during iteration.
The episode coordinator is a small state machine; it does not duplicate motion
matching, IK, attachment, or carry algorithms.

## Grasp Provider Boundary

The viewer requests grasps through a narrow interface:

```cpp
struct GraspCandidate {
    interaction::Transform hand_world;
    vec3 approach_world;
    interaction::Hand preferred_hand;
    uint64_t grasp_id;
};

class GraspProvider {
public:
    virtual ~GraspProvider() = default;
    virtual std::vector<GraspCandidate> query(
        const ObjectSnapshot& object) const = 0;
};
```

The first implementation is `KnownGraspProvider`, which transforms a grasp
stored in object-local coordinates into world space. A later
`GraspNetProvider` can produce the same values without changing episode
planning or playback.

The first provider emits one grasp that permits both hands. `preferred_hand`
is therefore advisory; the search evaluates captured left reaches and mirrored
right reaches and retains the better feasible candidate.

## Frozen Interaction Attempt

Pressing F during free locomotion captures an immutable attempt containing:

- object identity, generation, world transform, and dimensions;
- the selected grasp transform and approach direction;
- the displayed G1 root, pose, and velocity;
- the selected reach clip, hand, corrected path, contact frame, and entry root;
- all collision and IK settings used to certify the result.

Object movement, reset, cancellation, or generation mismatch invalidates the
attempt. The planner never silently retargets an active attempt.

## Reach Selection

Search begins from the frozen grasp, not from an object-specific motion label.
Both hands and the full reach pack are considered. Each candidate is
contact-anchored to the requested wrist transform, posture-aware IK corrects
the final reach, and swept collision checks include the hand, arms, torso,
legs, object, table, shelf, and lower table geometry.

Hard rejection precedes ranking. The search rejects candidates with invalid
transforms, unreachable IK, excessive correction, object penetration,
environment penetration, or an entry root that locomotion cannot reach.

Feasible candidates are ranked in this order:

1. entry-root travel distance and facing compatibility from the frozen G1;
2. grasp position and orientation error after IK;
3. path directness, including backtracking and excess wrist travel;
4. deformation from the recorded pose; and
5. deterministic clip identity as the final tie break.

This preserves diverse reusable reaches while preventing a beautiful reach
whose entry is on the wrong side of the scene from winning the playable
episode.

## Runtime State Machine

The viewer owns these states:

- `FreeLocomotion`: WASD drives ordinary motion matching.
- `Plan`: F freezes the request and runs reach search once.
- `Approach`: locomotion follows the selected entry root and facing.
- `Bridge`: a short deterministic pose-and-root-velocity blend transfers
  authority from locomotion to the first corrected reach frame.
- `Reach`: the corrected full reach plays at 25 Hz.
- `Attach`: contact is validated and the object becomes hand-relative.
- `CarryBlend`: the contact pose blends into the existing carry controller
  while preserving the active wrist/object transform.
- `Carry`: WASD again controls locomotion; the active hand maintains the
  grasp and the inactive arm follows locomotion.
- `Failed`: the reason is shown and control returns to free locomotion.

F is edge-triggered. Repeated F presses do not restart a committed attempt.
Escape cancels before attachment. R resets the viewer and returns the box to
its initial pose.

## Handoffs

### Locomotion to reach

The approach ends by spatial tolerance, not elapsed time. The root must be
close to the selected entry position and yaw and below a capture-speed
threshold for a short stable interval. The bridge preserves the current
displayed root and fades local rotations and velocities into the corrected
reach start. It must not teleport to the source clip origin.

### Reach to attachment

The object remains fixed until the certified contact frame is presented. At
that frame, the final wrist/object error and collision status are checked
again. On success, the object-to-hand transform is frozen and all subsequent
object motion comes from the displayed active wrist.

### Attachment to carry

The current reach pack ends at contact. Although the source recordings contain
post-grab motion, their annotations do not identify a paired neutral-return
boundary; automatic return detection is ambiguous for many samples. The first
milestone therefore uses a short deterministic, hand-constrained blend from
the contact pose into the existing hand-specific carry controller. The object
must remain attached without visible drift throughout the blend.

Extracting and validating paired recorded return segments is a later data pass.
It is not required to prove the pickup episode.

## Rendering and Diagnostics

The flat viewer renders:

- G1 mesh;
- the movable box;
- floor, table, shelf, and lower table;
- the selected entry root and facing;
- the selected root approach path;
- the corrected active-wrist path;
- current state, selected hand, clip, search time, and rejection reason; and
- position/orientation error at contact.

The overlay makes origin snapping, wrong-hand carry, stalled approach, and
object drift immediately visible.

## Performance

F performs one bounded search against the frozen request. Search may run on a
worker thread so rendering and input remain responsive, but there is no live
recomputation while the object moves. The viewer shows `SEARCHING` and the
elapsed time. The initial target is under one second on the current machine;
the hard acceptance limit is three seconds. A timeout returns to free
locomotion without changing the object.

No diffusion model is loaded in this milestone.

## Failure Handling

Before attachment, any failed search, blocked entry, approach timeout,
excessive bridge mismatch, collision, or invalidated object cancels cleanly and
restores free locomotion. After attachment, the object remains attached unless
the user resets; carry failures fall back to a hand-constrained procedural
carry pose instead of dropping or teleporting the object.

Every failure has one visible reason code. The previous displayed pose remains
the blend source so failure recovery cannot jump to the origin.

## Verification

Automated tests must prove:

- the known grasp transforms correctly with object translation and rotation;
- one frozen F press produces one immutable plan;
- both left and mirrored right reaches can be selected;
- ranking considers entry compatibility before directness;
- approach completion is spatial and cannot rubber-band to the origin;
- bridge endpoints preserve root continuity and bounded joint motion;
- attachment occurs only at certified contact;
- the attached object follows the active wrist through `CarryBlend` and
  `Carry`;
- left/right selection propagates into the corresponding carry controller;
- cancellation and timeout leave the object free;
- the viewer target builds without terrain dependencies; and
- existing reach coverage, IK, collision, and interaction regressions remain
  green.

The playable acceptance test is:

1. start at several positions and headings;
2. walk with WASD;
3. press F once;
4. observe continuous approach, reach, contact, and carry;
5. resume WASD movement while the object stays in the selected hand; and
6. repeat until both left and right selections have been observed.

## Deferred Work

- Placement and release.
- Live GraspNet integration and multiple grasp candidates.
- Diffusion-based locomotion/reach and reach/carry stitching.
- Extracted paired post-grab return clips.
- Multi-object selection and obstacle-aware navigation.
- Production-controller integration.
