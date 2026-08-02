# Contact-Sequence Terrain Transition Design

## Status

Approved direction for the second terrain motion-matching pass. The existing
multi-horizon matcher is preserved as the v0 baseline. Sonic and physics
tracking remain out of scope; this pass evaluates kinematic motion in MuJoCo.

## Problem

The v0 matcher usually makes progress and can select terrain-compatible chunks,
but the feet visibly float, slide, or fail to take a convincing step on stairs.
The current search represents gait phase only indirectly through normalized foot
positions and velocities. It does not explicitly require the current support
state to match the candidate entry state or reason about which foot unloads and
lands next.

The transition composer also places a selected clip from the root and
inertializes root and joint offsets independently. It does not preserve a
planted foot in world space during the blend.

Evidence from the frozen six-route baseline supports separating these causes:

- 63 of 67 selected entries begin in double support;
- two begin in single support and two begin with neither foot supported;
- 73--93 percent of measured stance slide occurs within the first 15 emitted
  frames after a chunk transition on each route.

Explicit contact compatibility will remove the four unambiguous phase errors,
but the concentration of slide after mostly double-support transitions means a
phase gate alone is insufficient. Candidate transition quality and contact-aware
placement must be evaluated together.

## Considered Approaches

### 1. Scalar gait phase

Assign each frame a normalized cyclic phase and penalize phase differences.
This is inexpensive and useful for regular flat locomotion. It is ambiguous for
starts, stops, turns, asymmetric stair contacts, long double-support intervals,
and clips in which the gait is not periodic. It is not selected as the primary
representation.

### 2. Contact-state gate only

Require the current and candidate entry support masks to match. This immediately
rejects double-support-to-swing transitions, but many frames share the same
support mask while leading to different next steps. It is retained as the first
layer, not as the complete solution.

### 3. Contact-sequence search with emitted-window validation

Represent each candidate by its entry contact state and upcoming contact events,
then evaluate the actually placed and blended transition window. This directly
models which foot unloads, which foot lands, when it lands, where it lands, and
whether the emitted feet respect the query terrain. This is the selected design.

## Design

### Contact descriptors

Precompute one descriptor for every terrain horizon entry. The descriptor covers
the first 15 source frames and contains:

- the two-bit support mask at entry;
- the first foot to unload after entry, or `none`;
- the first foot to land after entry, or `none`;
- frames until the first unload and first landing;
- each foot's local displacement from entry to the landing event;
- each foot's maximum clearance above its source surface before landing;
- the support-mask sequence across all 15 frames.

The support mask is the primary phase representation. Event timing and leading
foot disambiguate frames within double support. A scalar phase may be recorded
later for diagnostics, but it does not participate in v1 selection.

### Current contact state

At each legal search boundary, derive the current support state from the emitted
kinematics and query terrain using the same clearance and vertical-speed
thresholds as the qualification metrics. Retain the previous support state for
one-frame hysteresis so measurement noise cannot flip a planted foot on and off.

The current state includes:

- support mask;
- planted-foot world positions;
- per-foot world velocity;
- current root pose and velocity;
- any pending command change.

### Layered selection

Selection remains deterministic and uses four layers:

1. **Contact gate:** require an exact entry support-mask match. If no candidate
   survives, allow only a one-bit relaxation and record the structured fallback.
   Never admit a no-support entry from a supported current state.
2. **Intent and terrain gate:** retain the v0 turn, progress, surface-sign, local
   exclusion, and supported-foot terrain-trace gates.
3. **Contact-sequence cost:** compare unload foot, landing foot, event timing,
   local landing displacement, and swing clearance. When a command requests
   movement, prefer a sequence that actually produces an unload and landing.
4. **Emitted-window validation:** place and inertialize the top ranked candidates
   for 15 frames without committing them. Use FK and the query terrain to reject
   penetration, unsupported hovering, insufficient swing clearance, planted-foot
   drift, or a missing commanded step. Choose the lowest-cost valid candidate.

The existing multi-horizon displacement, yaw, height, duration, and stall terms
remain responsible for longer-term intent. Contact terms decide whether the
transition into that outcome is physically and visually coherent.

### Contact-aware placement

Place the candidate using its supported feet rather than root translation alone:

- in single support, align the candidate's planted foot to the current planted
  foot in world XY and preserve its query-surface clearance;
- in double support, solve one least-squares planar translation for both feet;
- retain the selected yaw alignment;
- derive the placed root from the foot anchor solution.

Root and joint inertialization continue to provide pose continuity, but a
candidate fails emitted-window validation if the blend moves a planted foot
beyond the transition drift threshold. This pass does not add runtime IK or
procedural foot locking; those mechanisms would obscure whether search and
placement are selecting coherent source motion.

### Failure handling

The matcher must not silently hold or choose an airborne entry when no valid
candidate exists. It returns a structured coverage failure containing counts for
contact, event, terrain, clearance, drift, and legacy gates. The live viewer
continues running so a new command or reset can recover.

## Evaluation

The v0 baseline artifacts remain unchanged. The second pass runs the same six
frozen routes and adds transition-local metrics:

- entry contact mismatch count;
- unload/landing event mismatch count;
- maximum and accumulated planted-foot drift during the first 15 transition
  frames;
- swing-foot clearance and penetration during those frames;
- moving transitions with no unload-and-land event;
- stance slide split into transition-neighborhood and steady-source components.

The interactive stress test remains omnidirectional walking on the same stair:
side approach, lateral traversal with feet on different treads, diagonal ascent
and descent, turning in place, reversal on a riser, and side exit.

The second pass qualifies for viewer feedback only if:

- every supported transition enters a matching support state;
- no selected transition begins with neither foot supported;
- every moving transition contains an unload-and-land event unless the selected
  chunk is an explicitly classified turn-in-place sequence;
- transition-neighborhood stance slide is reduced by at least 50 percent on at
  least four of the six routes without reducing the existing 5/6 route outcome
  pass count;
- support-height p95 remains at or below 0.03 m;
- all failures are structured and the viewer remains recoverable.

## Scope Boundaries

This pass does not add Sonic tracking, dynamics, a learned controller, depth
input, runtime IK, procedural foot locking, or new corpus data. Those are later
experiments. The immediate question is whether contact-sequence-conditioned
search and contact-aware placement can make the existing terrain motion corpus
produce convincing kinematics.
