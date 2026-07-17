# SONIC Elastic-Band Lifecycle Design

## Problem

The pinned GEAR MuJoCo configuration constructs an enabled `ElasticBand` with
`kp_pos = 10000` and target height `z = 1`. At the registered G1 reset height
of `z = 0.793`, the band applies about 2,070 N upward to a 36.165 kg robot.
That predicts `+0.949 m/s` vertical velocity after 20 ms; the rejected rollout
recorded `+0.911 m/s`. This is why the robot remained airborne and never made
terrain contact.

The band cannot simply be disabled when the external simulator is constructed.
GEAR needs it during the unscored cold-start because policy control is not yet
active. Disabling it at construction let the unpowered robot fall and invoke
GEAR's automatic reset after 1.3 seconds. Setting the configuration flag to
false is also invalid in the pinned checkout because `sim_step()` accesses an
`elastic_band` attribute that is then never created.

## Approaches Considered

1. Disable `ENABLE_ELASTIC_BAND` in the loaded GEAR configuration. Rejected:
   the pinned `DefaultEnv` crashes because it still reads the absent attribute.
2. Disable the band immediately after simulator construction. Rejected: the
   unpowered cold-start falls before the controller becomes ready.
3. Make band state an explicit reset input. Selected: bootstrap resets keep the
   band enabled; scored resets disable it before resetting and priming physics.

## Design

The internal gated-simulator reset protocol gains one required boolean field,
`elastic_band_enabled`. The value crosses every layer without a default:

`GatedSimulatorClient.reset` -> JSONL reset request ->
`GatedSimulatorRunner.reset` -> `SimulatorBackend.reset_from_qpos`.

`ExternalGearBackend.reset_from_qpos` sets the pinned simulator's existing
`sim_env.elastic_band.enable` field to the requested value before
`mj_resetData`, qpos restoration, and `mj_forward`. The reset response repeats
the exact boolean so the client and run evidence can bind the applied state.
Missing fields, non-boolean values, a missing band object, or a response that
does not echo the requested value are integration failures.

The orchestration has exactly two policies:

- the initial unscored bootstrap reset passes `elastic_band_enabled=True` and
  stores its reset response in bootstrap evidence;
- `_reset_and_prime_scored_epoch` passes `elastic_band_enabled=False`, stores
  that response in prime evidence, advances exactly one 5 ms LowState step,
  and only then activates control.

No MM reference, policy input, gains, scene geometry, timing, or qualification
threshold changes in this patch.

## Verification

Unit boundaries must prove strict boolean validation, exact client/server
transport, backend application on successive enabled/disabled resets, bootstrap
use of `True`, scored use of `False`, and evidence echoing. A controller-owned
evaluator outside the candidate worktree independently checks these behaviors.

After integration, a fresh real known-good run is the physical smoke oracle.
Before any full MM rollout, its scored log must show terrain contact and must
not reproduce the prior `+0.91 m/s` first-row launch. A short replay and montage
must be visually inspected before proceeding.
