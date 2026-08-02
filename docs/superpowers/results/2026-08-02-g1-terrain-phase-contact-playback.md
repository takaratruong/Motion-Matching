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
  --swing-clearance-margin-m 0.025 \
  --foot-correction-halflife-s 0.025 \
  --swing-plan-sigma-frames 2.5 \
  --maximum-source-contact-p95-m 0.033
```

This remains kinematic-only: no physics and no Sonic tracking are active.

## Frozen route comparison

| Variant | Route outcomes | Total stance slide | Worst route p95 penetration | Maximum joint speed |
|---|---:|---:|---:|---:|
| Previous multi-horizon baseline | 5/6 | 2.882 m | 0.175 m | 10.85 rad/s |
| Contact-phase gate only | 6/6 | 3.418 m | 0.094 m | 14.53 rad/s |
| Phase gate + capped foot cleanup | **6/6** | **2.451 m** | **0.093 m** | **12.00 rad/s** |
| Phase + foot cleanup + 3 cm swing clearance | **6/6** | **2.320 m** | **0.093 m** | **12.00 rad/s** |
| 3 cm clearance + 25 ms correction half-life | **6/6** | **2.024 m** | **0.093 m** | **12.00 rad/s** |
| Recommended: source-path swing plan | **6/6** | **1.923 m** | **0.093 m** | **12.00 rad/s** |
| Source plan + 3.3 cm source-quality ceiling | **6/6** | **1.732 m** | **0.093 m** | **12.00 rad/s** |

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

The final source-path swing plan looks through the selected clip's remaining
unsupported interval, places that future foot path in the emitted root frame,
measures terrain clearance along it, and spreads required lift backward with a
short Gaussian envelope. At a 2.5 cm clearance margin and 2.5-frame sigma it
lowers aggregate slide another 5.0%, from 2.024 m to 1.923 m. All six route
slides improve or tie, worst-route p95 penetration remains 0.093 m, maximum
single-frame penetration slightly improves from 0.16622 m to 0.16613 m, and
worst-route joint-speed p95 improves from 1.9284 to 1.9238 rad/s. Predicted
samples outside the finite terrain grid fall back to reactive clearance rather
than terminating the viewer.

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
- recommended source-path plan: `build/multi-horizon-terrain-skills/h025-m025-plan-s025-clean-v1`
- post-audit final: `build/multi-horizon-terrain-skills/h025-m025-plan-s025-clean-v4-final`

The two phase-only matrices have the identical deterministic SHA-256
`dbb4e87fc7252bc5cd0212218b9b7148cd86a812894e67510535aabe92207e54`.
The two clean combined matrices have the identical deterministic SHA-256
`1e0c5cc86ac65e74101720b0282ce4be45b001d268643c27fe5c15a1839d5fb7`.
The two clean 3 cm swing-clearance matrices have the identical SHA-256
`e104fd51c3429b1ab85e7bfa07b9fbb483e170ab9ce7a20a6ce3612c7cc0f6f6`.
The three clean 25 ms matrices have the identical SHA-256
`78a69c669138d0547e7a60622cc49a35652326e34589d0013ac3ad24f02e9b91`.
Four clean source-path-plan matrices, including the post-audit build, have the
identical SHA-256
`40b1fa392c324954b379873b50a4c2a1de710affccff9497b935b67e79f1d1ae`.

The final source-quality layer excludes GRAIL clips whose authenticated source
foot/terrain contact-fit p95 exceeds 3.3 cm. It preserves 6/6 frozen and 16/21
broad outcomes, keeps the same worst penetration values, reduces frozen slide
by 10.0% (1.923 to 1.732 m), and reduces broad slide by 6.2% (7.051 to
6.616 m). Two exact frozen runs have SHA-256
`46d7877d73218a267da22be221a44f9a71d5293d3a3d36fea76eeee32f4023f3`.
The useful interval is narrow and discrete: 3.5 cm reproduces the prior
baseline, while thresholds at or below 3.2 cm lose route coverage.

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
6. Optional source-path planning evaluates the remainder of each selected
   swing interval in the emitted root frame and smoothly anticipates measured
   clearance deficits without changing liftoff, landing, or clip selection.
   Grid-boundary prediction failures degrade to the reactive pass.

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
- The source-path planner is also bounded: smoothing scales of 3--6 frames
  retained route completion but raised slide to 2.338--2.440 m. The useful
  basin was 2--2.75 frames, with a clean optimum at 2.5 frames and 2.5 cm.
- Online support-aware root-height reconstruction was rejected and removed.
  Both the reactive and source-path variants lost the mixed and side-exit
  routes, raised aggregate slide above 3.3 m, and produced root-jerk maxima
  above 3,600 m/s^3. Root reconstruction must be solved over a whole contact
  interval with continuity constraints, not applied as a frame-local offset.
- Carrying the absolute integrated command path across skill boundaries was
  rejected and removed. It dropped the frozen gate from 6/6 to 3/6 and the
  broad matrix from 16/21 to 8/21. The clips contain systematic root drift
  that is compatible with their contacts and the route contracts; treating
  the raw command integral as an exact global path over-pulled the selector,
  losing cross-tread, turn, mount, and mixed routes. The experiment confirms
  that the contact-graph oracle's route cost does not transfer as a hard
  displacement-target shift in the live matcher.
- Hard four-corner sole preview and oriented mid-foot preview were both
  rejected and removed. Both retained 6/6 routes, but every tested clearance
  margin from 0 to 2.5 cm increased slide. The best hard-corner result was
  2.291 m and the best mid-foot result was 2.276 m versus the retained
  1.923 m; peak-penetration improvements were below 0.5 mm. Foot geometry
  without a whole-swing smoothness and endpoint solve simply over-lifts the
  leg near stair edges. This supports implementing the complete trajectory
  objective rather than adding denser pointwise clearance constraints.
- Whole-swing symmetric smoothing and touchdown-frame preview were rejected
  and removed. The full-interval variant retained 6/6 and 16/21 outcomes, but
  increased slide by 2.8% on the frozen set and 1.1% on all routes; shorter
  smoothing scales from 1.0 to 2.25 frames did not beat the retained planner.
  Touchdown-only preview increased frozen slide by 19% with no penetration
  improvement. The source gait's landing acquisition should remain governed
  by contact onset; forcing the query height into its raw swing endpoint makes
  the ankle IK fight the clip.
- Caching one fixed correction plan per source swing was rejected and removed.
  It retained 6/6 routes but raised slide by 19% and worsened peak penetration.
  A plan anchored at swing onset becomes stale as the composed root/foot
  placement evolves; the retained receding source-path query is beneficial
  because it re-anchors the same phase-preserving objective every frame.
- Future left/right terrain heights sampled at predicted foot locations were
  rejected, both as a direct query and as a second search constrained to the
  baseline-selected horizon. The direct query had no scalar gain that improved
  all routes: gains up to 0.12 reproduced the baseline exactly, while the first
  effective gain changed a shared early decision and lost two routes. Holding
  the baseline horizon still dropped the frozen gate to 4/6 and the broad gate
  from 16/21 to 14/21, with aggregate frozen slide rising from 1.923 m to
  2.207 m. Per-foot height is useful evidence, but a greedy query cannot know
  which future contact is reachable and compatible with the next command.
- Double-support gait context was tested as a layered hard gate using the foot
  that just landed, the foot that swings next, and both together. Every form
  dropped the frozen gate from 6/6 to 4/6 and increased aggregate slide. Some
  individual maneuvers improved (notably reversal with previous-contact context
  and diagonal descent with next-contact context), confirming that gait context
  is informative, but a hard identity constraint overfits sparse transitions.
- Raw corpus expansion was also tested. GRAIL exposes 1,769 curb candidates
  and 6,094 candidates in each stair partition, versus the working corpus's
  760 total clips. A balanced 512-per-partition build published 1,415 accepted
  clips, but using it as a replacement passed only 2/6 frozen routes. Quantity
  alone does not replace the targeted lateral-exit and turning coverage in the
  curated corpus; the follow-up evaluates their deduplicated union.

## Overnight corpus and motion-warp continuation

The deduplicated union contains 1,967 clips: the curated 760-clip set plus a
balanced sample from all curb and stair partitions. Freezing feature
normalization to the curated corpus removed corpus-statistics drift. The union
then passed 6/6 frozen routes with 1.591 m stance slide and zero worst-route
p95 penetration, but it passed only 13/21 broad routes. It lost both upper
180-degree turns and the middle-right 90-degree turn. Restricting only turning
chunks to curated clips still passed 13/21 and lost the frozen mixed route.
This rejects both raw corpus replacement and per-chunk specialist routing:
extra clips change the state that arrives at a later turn, so filtering only
the turn itself is too late.

A separately gated endpoint-warp experiment smoothly deforms each selected
skill toward the bounded command target over its actual playback duration.
The entry pose and velocity remain continuous, and translation/yaw corrections
are independently capped. Unrestricted 2.5 cm / 0.1 rad warping fixed both
previously failing 45-degree stair turns, proving that endpoint lateness and
command-path drift are causal. It was not qualified: the broad matrix fell to
11/21 because root-only warping changes contact geometry.

Restricting warp to target height changes at or below 2 cm recovered 6/6
frozen routes, fixed both 45-degree turns, and reduced frozen slide from 1.732
to 1.389 m. On the full matrix it remained 16/21: it exchanged the two fixed
turns for new `side-mount-left` and `turn-90-middle-right` failures. Broad
slide improved from 6.616 to 4.972 m and worst p95 penetration improved from
0.192 to 0.147 m, but peak penetration increased from 0.273 to 0.385 m.
Translation-only warping produced the behavioral gains; yaw-only warping did
not. The endpoint-warp controls remain available for research in the runner
and live viewer, but the zero-warp source-quality configuration remains the
recommended demo.

The causal conflict is important for manual control: before a command change,
the matcher cannot know whether the same forward stair approach will later
turn, reverse, or exit sideways. A rigid root-path correction that prepares
one continuation can make another unreachable. The next representation must
therefore deform stance and swing contacts together, or synthesize the pose
conditioned on the newly observed command, rather than applying more scalar
candidate gates.

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
- The same Perceptive BFM paper also provides a direct later-stage answer to
  the proposed MM-to-depth/kinematics distillation hypothesis: it trains a
  blind teacher on terrain-conformal references, expresses the teacher's
  effective PD target in the raw-reference action frame, and anneals teacher
  control during DAgger-style vision-student rollouts. Its deployed actor uses
  proprioceptive history, a 21-step reference window, and a torso-centered
  17 x 11 local height scan. That is a strong architecture candidate after the
  privileged kinematic teacher is good enough.
- Recurrent Transition Networks uses a root-relative 13 x 13 terrain patch and
  reports that terrain conditioning helps for longer, obstacle-relevant
  transitions, while short 30-frame transitions often contain too little
  terrain interaction for the signal to matter:
  https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/uXaq0e2LE5W4DPR7KEW14/b40aa8a55f5b94b175e520b285c12e20/RecurrentTransitionNetworks_007.pdf

## Remaining limitation and next experiment

The source-path result confirms that phase-preserving future foot geometry is
useful, but the current planner remains an ankle-center clearance envelope.
The next high-value experiment is the full TCRS structure: optimize a mid-foot
trajectory with fixed liftoff and landing timing, smoothness/clearance/edge
costs, support-aware root height, and toe/heel-aware IK. This should target the
remaining visual floating and the mixed route's 0.093 m penetration tail.
Learned Motion Matching's rough-terrain setup similarly queries terrain under
future toes rather than relying on a denser root-centered height grid.

## Verification

- 92 focused unit/integration tests passed in the working tree.
- 65 relevant tests passed from a clean detached worktree at commit `2ec4a31`.
- 78 terrain matcher/viewer tests passed from the clean detached worktree at
  commit `13d3c0a`.
- 90 terrain matcher/viewer tests passed from the clean detached worktree at
  final commit `2a06f15`.
- 92 terrain matcher/viewer tests passed from the clean detached worktree at
  final code commit `0908d59`.
- The clean six-route matrix at `bc93c5e` passed 6/6 twice with an identical hash.
- The clean 3 cm swing-clearance matrix at `13d3c0a` passed 6/6 twice with an
  identical hash; maximum joint speed was 12.00 rad/s.
- The clean 25 ms correction matrix at `cf60556` passed 6/6 three times with an
  identical hash; aggregate slide was 2.024 m and maximum joint speed was
  12.00 rad/s.
- The clean source-path matrix at `0908d59` passed 6/6 after the boundary
  fallback audit and reproduced the same hash as three preceding clean runs.
  Aggregate slide was 1.923 m and maximum joint speed was 12.00 rad/s.
- An automated live run walked beyond the finite height-grid boundary, entered
  a recoverable search-failure state, and reset without terminating the viewer.
- No physics or Sonic dependencies were added to the kinematic viewer.
