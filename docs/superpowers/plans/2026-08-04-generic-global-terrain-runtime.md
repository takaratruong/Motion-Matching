# Generic Global Terrain Runtime

## Scope

This is the globally privileged, purely kinematic ceiling for G1.  The caller
provides an exact terrain USD, its world transform, and a world-space route.
The runtime may use global root pose to capture a precompiled portal.  It does
not receive target motion identity, target robot states, or a clean reference
trajectory.  Ordinary flat locomotion remains under the official MotionBricks
two-stick controller.

This is not a SONIC or physics qualification.  Its purpose is to establish the
best motion-generation ceiling before replacing global scene/root information
with causal robot-centred perception and state estimation.

## Method

1. Ray-cast the requested route against the exact mesh and classify it as flat,
   continuous slope, monotonic steps, alternating curb/step course, or mixed.
2. Split a mixed route into separated terrain events.  MotionBricks retains the
   flat gaps; each event includes only the support context needed for a safe
   handoff.
3. Retrieve coherent clean G1 motion from the appropriate bank:
   - stairs: source-contiguous GRAIL stair blocks plus compact fragment fallback;
   - ramps: 89 clean C490 windows, 50 uphill and 39 downhill;
   - curbs: 65 coherent C490 support transfers, 38 up and 27 down.
4. Warp the selected source support trajectory to the target geometry, preserving
   authored pose timing and solving bounded leg corrections against complete
   sole targets.
5. For alternating obstacles, beam-search phase-compatible transfer chains,
   trim shared support phases, inertialize pose residuals, and retarget internal
   seams to safe source footfalls.
6. Retain coherent source gait context at the two external portal boundaries.
   A source window is eligible only when that context remains on the same flat
   support level; context may not silently include the next obstacle from a
   long curb course.  If a consecutive source obstacle begins immediately
   after an otherwise valid landing, keep those consecutive authored frames
   and normalize the next obstacle's terrain-height profile to flat.  Anchor
   the shared frame to the already accepted pose and decay its bounded joint
   correction back to the authored gait rather than phase-blending an unrelated
   flat clip.
7. Reject unless the complete rendered G1 passes the exact target mesh audit:
   no forbidden-body penetration and at most 5 mm foot penetration.  Root,
   rotation, joint-step, and acceleration checks reject pose pops and teleports.
8. Jointly rank the retained terrain primitive and MotionBricks gait phases to
   compile a flat-to-terrain-to-flat portal course.  Save accepted courses in a
   single manifest consumed directly by the browser/Switch viewer.

## Current evidence

- Stairs: 4/4 leave-one-source-out exact meshes accepted, including descent and
  an oblique ascent.  Representative full course:
  `artifacts/generic_stair_route/canary_target223_mesh_only_v1/motionbricks_course`.
- Ramps: 4/4 leave-one-source-out exact meshes accepted, two uphill and two
  downhill.  Representative full courses:
  `artifacts/generic_terrain/slope_heldout4_mesh_only_v1`.
- Curbs: both short held-out exact meshes accepted; target 27 also has an
  accepted MotionBricks-flat/curb/flat course.  The two long alternating routes
  are retained as stress tests for tall-transfer coverage and internal seam
  compatibility rather than counted as accepted coverage.
- Unified route: one 10 m mesh-only scene discovers an uphill ramp and a separate
  up/down curb.  The ramp passes at 2.47 mm maximum foot penetration and 0.082
  rad maximum joint step; the curb passes at 4.38 mm and 0.134 rad.  Both have
  zero forbidden-body collision:
  `artifacts/generic_terrain/unified_mixed_smoke_v1/generation`.
- Unified runtime: the same scene now compiles both events into one accepted
  MotionBricks portal manifest.  The full ramp course passes at 2.03 mm foot
  penetration, zero body collision, and 0.118 rad maximum joint step.  The
  full curb course passes at 2.78 mm, zero body collision, 0.230 rad maximum
  joint step, and 22.83 m/s^2 root acceleration.  The curb landing bridge
  normalizes the adjacent 8.285 cm source rise to flat and overlaps the
  accepted descent by 0.13 mm at the root with zero joint gap.  Videos and
  dense seam sheets are under:
  `artifacts/generic_terrain/unified_mixed_smoke_v1/portals_landing_bridge`.
- Curved global showcase: a 70.8 s planned course combines an oblique stair
  ascent, left S-curve, turned stair descent, right arc, ramp ascent, right
  S-curve, turned ramp descent, diagonal-left travel, and an alternating curb
  traverse.  It travels 24.47 m with 9.73 m lateral span, zero unsupported
  frames, 4.33 mm maximum complete-foot penetration, and zero forbidden-body
  penetration.  Dense foot and external-seam sheets pass visual review:
  `artifacts/global_terrain_showcase/coherent_curved_v13_extended_gauntlet`.

## Main commands

Generate one arbitrary route:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.generate_generic_terrain_route \
  --terrain-usd /path/to/terrain.usd \
  --terrain-position 0 0 0 \
  --terrain-quaternion-wxyz 1 0 0 0 \
  --start-xy START_X START_Y \
  --end-xy END_X END_Y \
  --output-dir /path/to/output
```

Compile accepted events into MotionBricks portals:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.compose_generic_terrain_portals \
  --route-summary /path/to/output/summary.json \
  --output-dir /path/to/portals
```

Launch the browser/Switch viewer from the compiled scene manifest:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.motionbricks_global_terrain_viewer \
  --terrain-route-manifest /path/to/portals/portal_manifest.json
```

## Evaluation branches

- Clean primitive fails exact collision: improve source retrieval or support
  warp; do not debug MotionBricks or SONIC.
- Primitive passes but an internal curb seam fails: add a phase-compatible
  transfer candidate only at that seam, then rerun the exact whole-course audit.
- Primitive passes but flat portal composition fails: search another complete
  terrain primitive, a source transfer with genuine flat boundary context, or
  broader MotionBricks flat motion; do not loosen terrain collision thresholds.
- Kinematics and portal pass but SONIC fails: tracker/style adaptation is the
  isolated blocker.
- Global runtime passes: replace exact global pose/mesh incrementally with local
  height observations and state estimation while retaining the same audit set.

## Known limitations

- Geometry outside the clean source primitive distribution can still be
  rejected; exact scene knowledge does not manufacture a dynamically plausible
  gait.
- Steering is live on flat and route-selective at portal capture, but a committed
  terrain traversal is not yet re-planned mid-course.
- The curved showcase is a globally planned composition of qualified coherent
  traversals.  It demonstrates non-straight travel and repeated terrain
  handoffs, but it is not evidence of arbitrary joystick redirection while the
  robot is already on a stair, ramp, or curb.
- The accepted mixed manifest covers one ramp and one up/down curb; it is a
  unified-dispatch canary, not exhaustive evidence over arbitrary event order,
  dimensions, or approach angles.
- The accepted evidence is clean kinematics.  Tracking, contacts under dynamics,
  actuator limits, and sim-to-real robustness remain separate tests.
