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
  --device cuda:0 --multi-horizon --contact-phase-gate --foot-lock \
  --swing-clearance-margin-m 0.03 \
  --foot-correction-halflife-s 0.025
```

This remains kinematic-only: no physics and no Sonic tracking are active.

## Frozen route comparison

| Variant | Route outcomes | Total stance slide | Worst route p95 penetration | Maximum joint speed |
|---|---:|---:|---:|---:|
| Previous multi-horizon baseline | 5/6 | 2.882 m | 0.175 m | 10.85 rad/s |
| Contact-phase gate only | 6/6 | 3.418 m | 0.094 m | 14.53 rad/s |
| Phase gate + capped foot cleanup | **6/6** | **2.451 m** | **0.093 m** | **12.00 rad/s** |
| Phase + foot cleanup + 3 cm swing clearance | **6/6** | **2.320 m** | **0.093 m** | **12.00 rad/s** |
| Recommended: 3 cm clearance + 25 ms correction half-life | **6/6** | **2.024 m** | **0.093 m** | **12.00 rad/s** |

The clean committed combined version reduces aggregate stance slide by 14.9% relative to the
previous baseline while fixing the failed side exit. It is not uniformly
better: riser-reversal slide changes from 0.453 m to 0.664 m. Cross-tread
improves from 0.508 m to 0.480 m, turn-on-stair from 0.627 m to 0.275 m,
diagonal descent from 0.948 m to 0.761 m, and side exit from 0.320 m to
0.244 m.

The optional 3 cm swing-clearance layer is the new recommended visual demo.
On the clean committed solver it lowers aggregate slide another 5.4%, from
2.451 m to 2.320 m, and reduces source-unsupported samples below terrain from
8.13% to 6.29% (22.7% relative). Five routes improve or remain close; the
mixed route's slide rises from 0.027 m to 0.100 m and turn-on-stair rises from
0.275 m to 0.295 m. The maximum route p95 penetration remains 0.093 m, so this
is a bounded improvement rather than a claim that clearance is solved.

Shortening the correction half-life from 40 ms to 25 ms is the final
recommended setting. It lowers aggregate slide another 12.8%, from 2.320 m to
2.024 m, while retaining all six outcomes, the 0.093 m worst-route p95
penetration, the same command-tracking and stall metrics, and the 12 rad/s
output cap. A clean bracket at 20, 22.5, 25, 27.5, 30, and 35 ms had a clear
minimum at 25 ms; two exact repeats reproduced the same matrix hash.

Artifacts:

- baseline: `build/multi-horizon-terrain-skills/qualified-v2-final`
- phase-only: `build/multi-horizon-terrain-skills/contact-phase-v1`
- deterministic phase repeat: `build/multi-horizon-terrain-skills/contact-phase-v2-repeat`
- recommended clean combined: `build/multi-horizon-terrain-skills/phase-foot-lock-capped-clean-v2`
- deterministic clean repeat: `build/multi-horizon-terrain-skills/phase-foot-lock-capped-clean-v3-repeat`
- recommended clean swing clearance: `build/multi-horizon-terrain-skills/phase-foot-lock-swing-clearance-m030-clean-v1`
- deterministic swing-clearance repeat: `build/multi-horizon-terrain-skills/phase-foot-lock-swing-clearance-m030-clean-v2-repeat`
- recommended 25 ms clean run: `build/multi-horizon-terrain-skills/phase-foot-lock-swing-m030-halflife-h025-clean-v1`
- deterministic 25 ms repeats: `build/multi-horizon-terrain-skills/phase-foot-lock-swing-m030-halflife-h025-clean-v2-repeat` and `build/multi-horizon-terrain-skills/phase-foot-lock-swing-m030-halflife-h025-clean-v3-repeat`

The two phase-only matrices have the identical deterministic SHA-256
`dbb4e87fc7252bc5cd0212218b9b7148cd86a812894e67510535aabe92207e54`.
The two clean combined matrices have the identical deterministic SHA-256
`1e0c5cc86ac65e74101720b0282ce4be45b001d268643c27fe5c15a1839d5fb7`.
The two clean 3 cm swing-clearance matrices have the identical SHA-256
`e104fd51c3429b1ab85e7bfa07b9fbb483e170ab9ce7a20a6ce3612c7cc0f6f6`.
The three clean 25 ms matrices have the identical SHA-256
`78a69c669138d0547e7a60622cc49a35652326e34589d0013ac3ad24f02e9b91`.

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
5. An optional swing-clearance pass samples the query surface under source-
   unsupported ankles, raises only penetrating targets, and uses the same
   bounded IK, inertialization, correction ceiling, and 12 rad/s output cap.

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
- A hard time-to-next-contact gate failed every tested tolerance from 8 to 28
  frames (best: 5/6 routes). A soft timing cost passed 6/6 only at weights too
  small to change any selected clip; the first effective weights lost routes.
- Reweighting the existing entry/outcome terms with the phase gate did not
  help: legacy entry weight passed 2/6, entry-only 3/6, and outcome-only 4/6.
- Swing-clearance margins from 0 to 5 cm all retained 6/6, but the clean sweep
  had a clear optimum at 3 cm. Larger margins increased slide.
- Layered candidate swing-clearance gates at -5, -3, and -1 cm were rejected:
  they passed 5/6, 5/6, and 4/6 routes respectively, and the stricter versions
  produced worse penetration tails. Hard deletion cannot substitute for a
  contact-space swing trajectory.
- Causal swing-foot lookahead at 20, 40, 60, and 80 ms preserved 6/6 outcomes
  and reduced maximum single-frame penetration by about 3%, but increased
  aggregate slide to 2.417--2.541 m and raised joint-speed p95 by roughly
  14--17%. It was removed: lifting sooner is not equivalent to optimizing a
  phase-consistent swing trajectory.

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
- Perceptive BFM's terrain-conformal reference synthesis preserves raw contact
  timing but optimizes a smooth mid-foot swing trajectory with clearance and
  edge costs, reconstructs root height from support, and solves ankle/toe/heel
  targets jointly. This is the closest published diagnosis of our remaining
  reactive-clearance limitation:
  https://arxiv.org/html/2606.08059
- Recurrent Transition Networks uses a root-relative 13 x 13 terrain patch and
  reports that terrain conditioning helps for longer, obstacle-relevant
  transitions, while short 30-frame transitions often contain too little
  terrain interaction for the signal to matter:
  https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/uXaq0e2LE5W4DPR7KEW14/b40aa8a55f5b94b175e520b285c12e20/RecurrentTransitionNetworks_007.pdf

## Remaining limitation and next experiment

The timing and lookahead experiments show that neither a scalar phase
descriptor nor reactive pre-lifting is the missing signal. The remaining
long-tail penetration and visual floating point instead to an explicit,
phase-preserving swing path with foot geometry. The next high-value experiment
is therefore a TCRS-style mid-foot trajectory optimizer with fixed liftoff and
landing timing, smoothness/clearance/edge costs, support-aware root height, and
toe/heel-aware IK, evaluated against this frozen 6/6 result. Learned Motion
Matching's rough-terrain setup similarly queries terrain under future toes
rather than relying on a denser root-centered height grid.

## Verification

- 92 focused unit/integration tests passed in the working tree.
- 65 relevant tests passed from a clean detached worktree at commit `2ec4a31`.
- 78 terrain matcher/viewer tests passed from the clean detached worktree at
  commit `13d3c0a`.
- 90 terrain matcher/viewer tests passed from the clean detached worktree at
  final commit `2a06f15`.
- The clean six-route matrix at `bc93c5e` passed 6/6 twice with an identical hash.
- The clean 3 cm swing-clearance matrix at `13d3c0a` passed 6/6 twice with an
  identical hash; maximum joint speed was 12.00 rad/s.
- The clean 25 ms correction matrix at `cf60556` passed 6/6 three times with an
  identical hash; aggregate slide was 2.024 m and maximum joint speed was
  12.00 rad/s.
- An automated live run walked beyond the finite height-grid boundary, entered
  a recoverable search-failure state, and reset without terminating the viewer.
- No physics or Sonic dependencies were added to the kinematic viewer.
