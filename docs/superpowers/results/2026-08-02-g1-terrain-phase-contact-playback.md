# G1 terrain phase/contact playback result — 2026-08-02

## Outcome

The recommended kinematic demo is now the fixed-horizon terrain-skill matcher
with layered source-contact phase filtering and bounded, inertialized foot
cleanup. It passes all six frozen adversarial routes. The previous qualified
baseline passed five.

Run it with:

```bash
PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -B \
  -m mm_sonic.torch_terrain_live_viewer \
  --dataset build/torch-grail-terrain-feedback-v2 \
  --config sonic/configs/experiments/torch_grail_multi_horizon_skills.json \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --device cuda:0 --multi-horizon --contact-phase-gate --foot-lock
```

This remains kinematic-only: no physics and no Sonic tracking are active.

## Frozen route comparison

| Variant | Route outcomes | Total stance slide | Worst route p95 penetration | Maximum joint speed |
|---|---:|---:|---:|---:|
| Previous multi-horizon baseline | 5/6 | 2.882 m | 0.175 m | 10.85 rad/s |
| Contact-phase gate only | 6/6 | 3.418 m | 0.094 m | 14.53 rad/s |
| Phase gate + capped foot cleanup | **6/6** | **2.451 m** | **0.093 m** | **12.00 rad/s** |

The clean committed combined version reduces aggregate stance slide by 14.9% relative to the
previous baseline while fixing the failed side exit. It is not uniformly
better: riser-reversal slide changes from 0.453 m to 0.664 m. Cross-tread
improves from 0.508 m to 0.480 m, turn-on-stair from 0.627 m to 0.275 m,
diagonal descent from 0.948 m to 0.761 m, and side exit from 0.320 m to
0.244 m.

Artifacts:

- baseline: `build/multi-horizon-terrain-skills/qualified-v2-final`
- phase-only: `build/multi-horizon-terrain-skills/contact-phase-v1`
- deterministic phase repeat: `build/multi-horizon-terrain-skills/contact-phase-v2-repeat`
- recommended clean combined: `build/multi-horizon-terrain-skills/phase-foot-lock-capped-clean-v2`
- deterministic clean repeat: `build/multi-horizon-terrain-skills/phase-foot-lock-capped-clean-v3-repeat`

The two phase-only matrices have the identical deterministic SHA-256
`dbb4e87fc7252bc5cd0212218b9b7148cd86a812894e67510535aabe92207e54`.
The two clean combined matrices have the identical deterministic SHA-256
`1e0c5cc86ac65e74101720b0282ce4be45b001d268643c27fe5c15a1839d5fb7`.

## What changed

1. At a terrain skill transition, rank normally, then validate candidates in a
   strict layer that preserves the current left/right support pattern. If that
   layer has no terrain-compatible candidate, fall back to the original
   terrain-only layer. This makes phase a reachability constraint rather than
   another arbitrary scalar weight.
2. During sequential playback, source-supported feet acquire persistent world
   locks projected onto the query terrain. Fixed-root leg IK supplies the
   desired correction.
3. Cleanup is inertialized, unlocks when the desired contact becomes too far or
   unreachable, and caps emitted joint speed at 12 rad/s. The hard-lock trial
   was rejected because it produced 35–37 rad/s spikes.
4. Live display-marker sampling now clamps only visualization queries at the
   finite terrain boundary. The authoritative matcher queries remain strict.
   This fixes the reproducible viewer crash when the operator walks outside the
   height-grid domain.

## Ablations rejected

- Hard foot locking reduced slide on five routes but caused 35–37 rad/s joint
  discontinuities and worsened cross-tread slide.
- Very soft foot locking preserved baseline continuity but made almost no
  measurable improvement.
- Adding a horizontal source-foot speed gate changed the selected skill chain,
  raised cross-tread slide to 0.931 m, and introduced 0.127 m p95 penetration.
- Relaxed local contact composition with five candidates accepted 12/92 frozen
  difficult states. Doubling search to ten candidates only reached 14/92, so
  brute-force breadth is not the main bottleneck.

## Literature alignment

- Daniel Holden's production note describes contact acquisition, inertialized
  locking, automatic unlock by distance, and final IK; it also warns that an
  aggressive lock can damage the pose more than mild sliding:
  https://theorangeduck.com/page/code-vs-data-driven-displacement
- Learned Motion Matching outputs contact and applies IK for foot sliding, and
  adds future terrain-under-toe features for rough terrain:
  https://static-wordpress.ubisoft.com/montreal.ubisoft.com/wp-content/uploads/2020/07/09154101/Learned_Motion_Matching.pdf
- Gait-cycle feature matching supports explicit spatiotemporal gait features in
  a motion-matching query:
  https://doi.org/10.1111/cgf.14988
- UniAct reports G1 motion-matching features that include gait phase derived
  from foot contacts and segments clips at gait boundaries:
  https://awfuact.github.io/static/publications/arxiv25_uniact/paper.pdf
- Environment-aware Motion Matching uses layered environment validation,
  sequential playback between searches, and emphasizes data coverage:
  https://joseluisponton.com/assets/pdf/emm_siggraphasia2025.pdf

## Remaining limitation and next experiment

The riser-reversal route passes, but its slide regression indicates that an
exact binary support-pattern gate is still too coarse during reversal. The next
high-value experiment is a short contact-horizon descriptor: current support,
time-to-release, next touchdown foot, touchdown XY/Z, and a small transition
penalty. It should be evaluated as a layered gate/penalty against this frozen
6/6 result, not merged speculatively.

## Verification

- 92 focused unit/integration tests passed in the working tree.
- 65 relevant tests passed from a clean detached worktree at commit `2ec4a31`.
- The clean six-route matrix at `bc93c5e` passed 6/6 twice with an identical hash.
- An automated live run walked beyond the finite height-grid boundary, entered
  a recoverable search-failure state, and reset without terminating the viewer.
- No physics or Sonic dependencies were added to the kinematic viewer.
