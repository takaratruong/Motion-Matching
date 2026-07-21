# Holden Movement Model Design

Date: 2026-07-21

## Purpose

Add an opt-in, acceleration-limited movement model before motion-matching
trajectory prediction. The experiment tests whether a physically feasible
brake-through-zero reference improves SONIC tracking during abrupt forward to
backward and left to right reversals.

The existing raw command path remains the default and the control condition.
The experiment must produce honest comparative evidence even if the shaped
profile performs worse.

## Observed Problem

The current operator path may change a requested planar velocity from roughly
`+0.9 m/s` to `-0.9 m/s` at one boundary. The MM runtime immediately uses the
new requested velocity for all future desired-velocity samples, then applies
its existing root-trajectory spring (`0.27 s` half-life) and pose
inertialization (`0.10 s` half-life).

Live 0.2-second-prefix evidence shows that MM-generated motion usually points
in the new direction within one to three prefixes, while SONIC's physical
motion takes about six prefixes to remain aligned. Manually braking toward zero
before requesting the opposite direction produces more reliable motion.

## Scope

The first experiment supports exactly two session profiles:

- `raw`: existing behavior and default.
- `holden-v1`: acceleration-limited intermediate velocity with fixed,
  recorded parameters.

`holden-v1` uses:

- acceleration: `1.5 m/s^2`;
- deceleration: `2.0 m/s^2`;
- directional acceleration: disabled;
- turn strength: disabled;
- the existing MM root spring and pose inertialization unchanged.

This phase does not add parameter tuning through the CLI, post-MM kinematic
warping, learned prediction, obstacle behavior, or changes to SONIC weights.
It does not claim to implement every feature in Daniel Holden's New Movement
Model; it tests the acceleration/deceleration intermediate-velocity layer that
is missing from the current runtime.

## Architecture

The data flow is:

```text
raw operator velocity
  -> terrain/traversability limiter
  -> session movement model
  -> predicted desired-velocity samples
  -> existing MM root trajectory spring
  -> motion matching and inertialization
  -> SONIC reference tracking
```

The movement model belongs in the C++ MM runtime, not Python. Its persistent
intermediate velocity is stored in `g1_controller_state`, which is already
cloned for candidate generation and swapped only on commit. Consequently:

- abort discards every tentative movement-model update;
- regenerate from the same active state is bit-identical;
- superseded commands cannot leak shaped velocity into the next candidate;
- reset initializes the intermediate velocity to zero.

The Python layer selects and records the session profile but does not own the
model state.

## Movement Update

Let `v` be the persistent intermediate velocity and `target` the
traversability-limited command. For each 25 Hz MM step (`dt = 0.04 s`):

1. Compute `difference = target - v`.
2. Use acceleration when `|target| + epsilon > |v|`; otherwise use
   deceleration.
3. Limit the change magnitude to `limit * dt`.
4. Snap to the target when the remaining difference is within that limit;
   otherwise advance along the normalized difference.

For an equal-speed 180-degree reversal, the update first uses deceleration as
the velocity approaches zero, then acceleration as its magnitude grows in the
new direction. It cannot overshoot the target.

The movement-model update produces the applied velocity for the current frame.
The raw request remains available as intent and evidence.

## Future Trajectory Prediction

Motion matching must see the upcoming braking profile, not only the current
shaped velocity. In `holden-v1`, the direct trajectory predictor starts from a
copy of the current intermediate velocity and advances it toward the limited
target at each trajectory sample time (`1/3 s`). Those predicted intermediate
velocities replace the current constant raw-command samples before the existing
root position/velocity spring runs.

The persistent state advances only once by the real MM step `dt`; prediction
uses a local copy and cannot mutate committed state.

In `raw`, future desired-velocity samples remain exactly the requested limited
velocity, preserving current behavior.

## Traversability and Blocking

The movement model operates after terrain/traversability command limiting so it
cannot accelerate toward a velocity rejected by scene safety. A blocked stop
clears planar intermediate velocity along with the existing simulated planar
dynamics. This prevents stale velocity from reappearing after a blocked frame.

## Protocol and Configuration

The manual demo exposes:

```text
--movement-model {raw,holden-v1}
```

The default is `raw`. The selected profile is session-scoped and is sent in
the MM reset request. The server:

- advertises `supported_movement_models: ["raw", "holden-v1"]` in `hello`;
- validates the requested profile before reset mutation;
- publishes the selected profile and fixed numerical parameters in reset
  evidence;
- keeps the profile fixed until the next reset.

An omitted profile is interpreted as `raw` for compatibility with existing
clients. Unsupported profiles fail closed before controller state or scene
state changes.

## Evidence

Existing source-chunk evidence already distinguishes:

- the raw `requested_velocity_holden`; and
- per-step `applied_velocity_holden`.

For responsive evidence, each boundary trace additionally records the first
and last applied velocities transformed into MuJoCo coordinates. The manual
summary records the selected profile and fixed parameter values. This provides
the chain:

```text
raw request -> shaped/applied velocity -> MM root displacement
            -> observed physical root displacement
```

No missing measurement is represented as zero; unavailable physical evidence
remains explicitly unavailable under the existing trace contract.

## Failure and Transaction Semantics

- Unknown profiles and non-finite or non-positive fixed parameters fail before
  reset publication.
- Non-finite movement state or prediction fails candidate generation before
  publish, commit, or physics release.
- Candidate abort and supersession leave active movement state unchanged.
- `raw` remains the fallback control profile; the system never silently falls
  back from an explicitly requested `holden-v1` profile.
- The existing generate, validate, prepare, supersession, publish, dual-commit,
  physics-release ordering remains unchanged.

## Testing

Automated tests are written and observed failing before implementation.

Pure C++ movement-model tests prove:

- zero to `0.9 m/s` respects the `1.5 m/s^2` acceleration bound;
- `+0.9 m/s` to `-0.9 m/s` brakes through zero, respects the deceleration and
  acceleration bounds, reaches the target, and never overshoots;
- 90-degree changes remain finite and bounded with directional acceleration
  and turn strength disabled;
- prediction does not mutate persistent state.

Runtime and protocol tests prove:

- reset initializes the intermediate velocity;
- clone/swap/abort/regenerate preserve transactional identity;
- `raw` retains existing generated arrays and diagnostics;
- `holden-v1` reports bounded per-step applied velocity and shaped future
  samples;
- hello/reset schema, default compatibility, and unsupported-profile rejection;
- Python schema, manual flags, summary, and responsive trace evidence.

The existing warning-strict C++ tests, real MM server tests, Python affected
suites, scheduler/boundary/coordinator tests, and default 10-interval tests
remain green.

## Live A/B Experiment

Run the same flat-scene 0.2-second-prefix setup twice, changing only the
movement profile. Apply abrupt W to S and A to D reversals with sufficiently
long holds. Record shared-display contamination if present rather than hiding
it.

Compare:

- signed wrong-direction physical displacement after each reversal;
- time and simulated duration to sustained positive projection;
- generated-versus-observed displacement error;
- completion, fall/stability, and evidence availability;
- MM time, completed-prefix latency, and real-time factor.

The hypothesis is supported if `holden-v1` reduces wrong-direction physical
displacement or tracking error without instability. It is rejected if tracking
or stability worsens. Longer intentional time-to-target is not alone a failure:
the model is designed to replace an infeasible instantaneous request with a
trackable reference.

## Deliverable Boundary

The deliverable is an opt-in, transaction-safe movement profile plus automated
and live comparative evidence. Learned models, action chunking, controller
retraining, and parameter optimization require separate hypotheses after this
result.
