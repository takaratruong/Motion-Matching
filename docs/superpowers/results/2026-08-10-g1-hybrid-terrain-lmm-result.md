# G1 hybrid terrain LMM overnight result

Date: 2026-08-10 (America/Los_Angeles)

## Outcome

The delivered proof of concept is a steerable, terrain-aware hybrid motion-matching runtime for G1:

- exact normalized 31-D command/terrain search over the complete range-safe corpus;
- a trained latent-32 neural compressor/decompressor for displayed articulation;
- command-owned planar placement with live terrain resampling;
- W/S speed, A/D steering, Space hard stop, R reset, and optional Linux evdev gamepad input;
- native G1 joint-limit validation before commit, with deterministic exact next-candidate selection when a learned candidate is mechanically invalid;
- authenticated MuJoCo G1, terrain, corpus, and model identities.

This is deliberately labeled **HYBRID TERRAIN LMM POC (EXACT SEARCH + LEARNED GENERATOR; SUPPORTED TERRAIN ONLY)**. It is not a learned projector, a dynamics policy, or evidence of unrestricted terrain generalization. The displayed MuJoCo path is kinematic; the terrain geometry used by this viewer is not a physics-contact claim.

## Corpus

Strict primary cache:

- path: `sonic/runs/g1-hybrid-terrain-lmm/corpus-v2-strict`
- manifest SHA-256: `e6fbe9413d4cbca58697b90a92f12a1832d28b961972624dd373bc72fa45494e`
- 3,970,932 rows, 15,816 ranges, 25 Hz
- train/evaluation: 3,573,688 / 397,244 rows
- Takara temporal boundary: row 15,688, with range-local derivatives and future horizons
- normalization fit on train rows only

Authenticated PFNN supplement:

- path: `sonic/runs/g1-hybrid-terrain-lmm/pfnn-supplement-v1`
- manifest SHA-256: `618ff022ed781ffcace6d7cc95ca74420468e9be2882903548b8b103b5cc7db8`
- 2,125 rows, 17 ranges (750 flat/turning rows and 1,375 stair/terrain rows)
- derived from 17 authenticated training clips and 53 authenticated terrain fits
- all 17 PFNN ranges are fit-only; none is in the held-out mask

Final combined cache:

- path: `sonic/runs/g1-hybrid-terrain-lmm/corpus-primary-pfnn-v2-strict`
- manifest SHA-256: `084c168b473e730ec24419a49f4be1526226e5f95a2dd81ae4d367c729889cdb`
- 3,973,057 rows, 15,833 ranges
- train/evaluation: 3,575,813 / 397,244 rows
- family rows: flat 18,182; curb 442,250; slope 464,250; stair 3,048,375

The 397,244-row source-held-out set is entirely from the primary Takara/GRAIL
corpus.  The metrics below test primary-corpus generalization; they do not claim
held-out PFNN generalization.

## Model and evaluation

Source-held-out selection model:

- path: `sonic/runs/g1-hybrid-terrain-lmm/selection-combined-v2-strict-latent32-visual-v1`
- manifest SHA-256: `0f8d36395012d59deac2c54a88ba64252fc959fae00cd9255a4545479b49fa90`
- latent 32, width 512, seed 1234, visual-articulation loss, 30,000 post-coverage steps
- held-out 397,244 rows: joint geodesic MAE 0.008949 rad; frame-max p95 0.066416 rad; FK p95 0.022377 m; support-foot p95 0.021489 m; contact F1 0.999979 / 0.999969

Final all-row refit:

- path: `sonic/runs/g1-hybrid-terrain-lmm/final-combined-v2-strict-latent32-visual-v1-allrows`
- manifest SHA-256: `f200db342dd514df6deb6203f3be014054efd4f9038dc0486cc3a4b274a6d826`
- bound selection evaluation SHA-256: `959f65141346a0dbd322472470414cb7f29fb1d070206cb16b0fc7ee8c08d494`
- diagnostic held-out metrics after refit: joint MAE 0.012557 rad; frame-max p95 0.084931 rad; FK p95 0.030048 m; support-foot p95 0.028566 m; contact F1 0.997467 / 0.999995

A read-only all-row native conversion diagnostic covered all 3,973,057 stored row-aligned feature/latent reconstructions and canonical source poses. It found zero non-finite conversions. At the runtime 1e-5-rad tolerance, 6,127 learned rows (0.1542%) and 2,706 canonical source rows (0.0681%) were outside at least one native range. This diagnostic is not a formal receipt and does not cover arbitrary live-terrain substitutions; it motivated exact mechanically-valid candidate selection rather than relaxed limits.

This is an explicit variance from the original design gate requiring every
train and held-out decoded row to be natively joint-limited.  That model-level
gate was not met and is not retroactively weakened.  PoC acceptance is scoped
to the authenticated scripted supported-terrain routes below: unsafe exact
candidates are retried in deterministic score order, and each committed pose
must pass native limits and MuJoCo forwarding with zero clamp or fallback.

## Formal MuJoCo evidence

Current formal receipts use schema `hybrid-terrain-lmm-mujoco-smoke/v2` and
acceptance contract
`strict-combined-frozen-authorities-scripted-runtime/v2`. Earlier v1 receipts
are retained only as historical diagnostics because they predate the exact
class/count/frozen-authority acceptance gates.

Authenticated ramp, full exact search:

- receipt: `sonic/runs/g1-hybrid-terrain-lmm/evidence/final-combined-v2-strict-latent32-visual-v1-allrows-ramp10-full-1000-formal-v2.json`
- receipt SHA-256: `0ad82ae339c6f55d7a4af52b18548a09d4b208624fa359832f99f0d88ab25f66`
- accepted; 1,000/1,000 learned MuJoCo forwards
- zero fallback, zero clamp, zero native-limit violations
- two unsafe candidate rejections; first row 168; maximum one rejection in any step
- all four motion families and 39 ranges selected; flat and hill terrain classes observed
- full 3,957,224-row range-safe search; exact tree/brute and manifest-row retrieval audits passed

Authenticated stairs, full exact search:

- receipt: `sonic/runs/g1-hybrid-terrain-lmm/evidence/final-combined-v2-strict-latent32-visual-v1-allrows-stairs-standard-full-1000-formal-v2.json`
- receipt SHA-256: `0c2f1de3e65af7fcec60cbb1cfbd001b605439a5d28aefe654698e00e8426a44`
- accepted; 1,000/1,000 learned MuJoCo forwards
- zero fallback, zero clamp, zero native-limit violations
- 49 unsafe candidate rejections; maximum two in any step
- all four motion families and 38 ranges selected

Historical pre-v2 GRAIL curb diagnostic:

- receipt: `sonic/runs/g1-hybrid-terrain-lmm/evidence/final-combined-v2-strict-latent32-visual-v1-allrows-grail-curb-default-full-1000-candidate-safety-v1.json`
- receipt SHA-256: `a3e58c8a7293dd84a47a44ebe2ef58f00ed86fd06c1f5b3074e053f6b72ce5e9`
- schema: `hybrid-terrain-lmm-mujoco-smoke/v1`; not current formal evidence
- completed 1,000/1,000 MuJoCo forwards, but is correctly rejected
- 986 learned frames and 14 canonical fallbacks; zero clamps/native-limit violations
- the generic multi-turn smoke route left the finite supported scene corridor, so it does not meet the zero-fallback acceptance gate

The original ramp run before candidate-safety retry is retained as rejected evidence. It had one 0.001216-rad canonical left-ankle-roll clamp. The corrected runtime never relaxed the joint limit: it rejects the unsafe exact winner and selects the exact next-best safe candidate.

## Launch

From the repository root:

```bash
DISPLAY=:1 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -u \
  -m mm_sonic.hybrid_terrain_lmm_combined_viewer view \
  --cache sonic/runs/g1-hybrid-terrain-lmm/corpus-primary-pfnn-v2-strict \
  --model sonic/runs/g1-hybrid-terrain-lmm/final-combined-v2-strict-latent32-visual-v1-allrows \
  --scene ramp-10-up-down \
  --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --transition-penalty 0.1
```

Controls:

- W / S: forward / reverse speed
- A / D: steer left / right
- Space: hard stop (overrides gamepad)
- R: reset to authenticated scene spawn
- Esc: quit
- optional `--gamepad /dev/input/eventN` when a readable controller is attached

No joystick device was attached during overnight acceptance, so keyboard steering was exercised live and evdev behavior was covered by focused tests rather than a hardware claim.

Live keyboard evidence after a W+A steering sequence is retained at
`sonic/runs/g1-hybrid-terrain-lmm/evidence/final-live-ramp-steered-formal-v2-1280.png`
(SHA-256
`3d58b186d52ece20a8d5178c4769cbae55e114d4914620f874257404ace736b8`).
The overlay shows a learned pose selected from slope family/range 3191 on the
authenticated indexed ramp, full 3,957,224-row search, supported terrain, and
zero runtime fallback, canonical fallback, clamp, or candidate rejection.

## Verification boundary

Formal acceptance covers the two authenticated 1,000-frame scripted MuJoCo runs above. It does not claim physics contact, arbitrary unindexed scenes, a complete native-limit proof for every possible live-terrain-conditioned query, or generalization beyond the supported corpus envelope. Out-of-domain terrain is visible and fails over to a clearly labeled canonical diagnostic state rather than silently pretending to be learned output.
