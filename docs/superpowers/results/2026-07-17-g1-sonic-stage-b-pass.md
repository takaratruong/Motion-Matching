# G1 SONIC Stage B Motion-Matching Pass

Date: 2026-07-17 UTC

## Verdict

The registered Stage B experiment passes. The complete 601-frame flat
motion-matching reference was transported through the pinned, unmodified GEAR
ZMQ decoder, and the pinned GEAR policy dynamically drove the official G1
SONIC MuJoCo system for the registered 12.000-second CONTROL interval. All five
ordered gates and all ten primary qualification checks passed.

This is a scripted simulation qualification. It is not a physical-G1 safety
certificate, a hardware deployment, or qualification of the separate manual
operator workflow. Horizontal path drift remains a secondary diagnostic and
does not become a primary pass criterion in this experiment.

Qualified motion-matching commit:

```text
4c97fc26a94e38427fa796642082e09ef51b6f53
```

Pinned GEAR commit:

```text
60de0df7ffedeef415fe58d435e92cc5b01ba3d9
```

The sealed manifest records both repositories as clean.

## Immutable prerequisite and final run

Stage B authenticated the passing Stage A prerequisite at:

```text
sonic/runs/stage-a-official-sonic-f8abb37-20260717/stage-a/known-good-stream-20260717T161308279711Z-87d1d58d/stage-a-evidence.json
```

Its SHA-256 is:

```text
78de0087d841989acde414a0ebfeaf993d31d91e1ae89b5e8858b830f0cc443c
```

The final clean-SHA command was:

```bash
PYTHONPATH=sonic/python CUDA_VISIBLE_DEVICES=0 \
  sonic/.venv/bin/python -B -m mm_sonic.cli stage-b \
  --stage-a-evidence sonic/runs/stage-a-official-sonic-f8abb37-20260717/stage-a/known-good-stream-20260717T161308279711Z-87d1d58d/stage-a-evidence.json \
  --gear-checkout /tmp/groot-wbc-plan-inspect \
  --policy /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/model_decoder.onnx \
  --observation-config /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/observation_config.yaml \
  --encoder /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/model_encoder.onnx \
  --source-mjcf /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717
```

Immutable run:

```text
sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/stage-b/stage-b-20260717T190656038623Z-b746af3f
```

It was created at `2026-07-17T19:06:56.043876Z` and finalized at
`2026-07-17T19:08:19.967334Z`.

## Gate and coverage result

| Gate | Status |
|---|---|
| `external_identity` | pass |
| `joint_projection_round_trip` | pass |
| `basis_and_scene_alignment` | pass |
| `flat_mm_kinematic_replay` | pass |
| `stage_b_dynamic` | pass |

The command script was the registered `flat-12s` sequence with SHA-256
`8137b0a6d951da2a5ead427698479d490150e963c5a458cad11a766968de6d85`.
All 30 command indices `0..29` were accepted. The source reference contains
601 frames `0..600` and has canonical SHA-256
`7e35a05db7c8b8960835233fd666f0680fa5428842aece9221755187b6d2c4d0`.
Five float32 subnormals were normalized to positive zero at the documented GEAR
transport boundary, producing transport SHA-256
`1ee8613c7b2cf3233e60ea3e9c69c26eb64af8eb0ebdf0576fb6175645fac257`.

The delivery audit proves one readiness frame, 30 exact logical publications
covering 600 frames, one 46-frame padding publication, one receipt fence, 33
consumer markers, and 32 causal fences. The authoritative target contains
exactly 601 data rows. `q.csv` and `base_quat.csv` each contain one header plus
601 same-control-tick rows.

## Causal duration and safety evidence

The terminal fence closes the false-pass path found during review. It snapshots
the simulator before stopping GEAR, requires identical post-stop and final
snapshots, and never advances physics after the stop.

| Terminal-fence field | Value |
|---|---:|
| Required CONTROL steps | 2,400 |
| Requested before stop | 2,400 |
| Observed CONTROL steps | 2,400 |
| CONTROL duration | 12.000000000000476 s |
| Prime-inclusive total steps | 2,401 |
| Prime-inclusive total duration | 12.005000000000477 s |
| Target rows at stop | 601 |
| Contact rows | 2,401 |
| State rows | 600 |
| Post-stop snapshot stable | true |
| Final snapshot stable | true |

The contact log runs contiguously from step 1 at 0.005 seconds through step
2,401 at 12.005 seconds. The 50 Hz state log runs from step 4 at 0.020 seconds
through step 2,400 at 12.000 seconds. The evaluator verified every intermediate
step and timestamp, not only the endpoints and row counts.

All registered primary checks are true:

- exact command, frame, CONTROL-duration, and safety-log coverage;
- integration and both known-good-relative tracking ratios;
- no registered forbidden contact group;
- minimum pelvis local height `0.8003473735 m` against `0.45 m`;
- minimum pelvis up-dot `0.9714069648` against `0.5`.

The 601-frame joint-position RMSE is `0.8414116409 rad`, below the registered
`1.5 ×` known-good limit of `1.2169256648 rad`. Pelvis-orientation RMS is
`1.3748006007 rad`, below its `2.5139185636 rad` limit. The mean GEAR tick
period is `19.9999750002 ms` over 600 intervals.

Secondary diagnostics report zero swing-foot scuffs, no retained contact
impulses, and `3.8133427799 m` horizontal path drift. These measurements are
retained for the next research iteration and did not override the registered
primary gates.

## Scene, evidence, and verification

The Stage A and Stage B run-local XML paths differ, so raw generated-scene
digests differ. Stage B verifies the complete generated XML and robot include,
normalizes only the authenticated run-local include path, and binds the
canonical scene semantics to Stage A. The resulting cross-run semantic digest
is:

```text
12ed6359d8ae66dda36e41b4a2fe10dcf57e824d9aa192d8a1ae59cf06d1e112
```

The registered terrain geom is ID `0`; allowed foot geoms and forbidden pelvis,
knee, torso, and hand groups are sealed in the manifest.

Final evidence identities:

- `stage-b-evidence.json`:
  `7c6fd55a9ece636824a674e2f15e43ce3e6a7ce80f65d9225236703ca29a8d04`
- `manifest.json`:
  `3cd556d585f365ead75336b7b0b7c940b8f2b05f850362a2c4f4fc50af0468d9`
- `inventory.json`:
  `79d623a86993385f5ec88d90b7fa9605b6ec43b8d02906ddc98a7e8cbc6c27b7`
- `gates/stage_b_dynamic.json`:
  `a06db73494c5d9eba4812dfead9724c3e1f8605ad44ac667d24e68ed4c1bca04`

The finalized inventory contains 142 files, and a fresh
`verify_run_inventory(...)` call returned `True`.

The new batched-terminal adversary was RED against the earlier implementation:
it could falsely count 2,336 post-stop simulator steps as active CONTROL time.
The repaired test is GREEN and proves a terminal jump is rejected after only
64 genuinely active steps, with zero stopped advances. The warning-strict full
SONIC catalog passed 457 tests with 6 expected skips. Its live C++ projection
certificate covered 260 samples and all 29 source and target joints, with
maximum joint-angle reconstruction error `2.51585969e-07 rad`. The protected
controller lifecycle evaluator also passed and confirmed the unmodified
controller contract.

No GEAR controller, gated simulator, or motion-matching server process survived
finalization. No push, merge, physical-robot command, or runtime activation was
performed.

An independent final reviewer accepted r13 with no blocker or important issue.
The review rehashed every inventory entry and gate output, rebuilt all 33
transport publications bit-for-bit, reparsed all 601 target/control rows,
recomputed tracking and raw safety cadence, reauthenticated the Stage A and
semantic-scene bindings, reran the terminal adversary at the clean commit, and
confirmed that no scoped process survived.

## Research boundary

This result establishes the requested scripted composition: the registered
motion-matching output dynamically drives the pinned G1 SONIC simulator. The
next research questions are translational path tracking, scene-aware
perturbation/obstacle trials, and eventually a separately reviewed physical-G1
deployment gate. The manual operator workflow remains a distinct deferred
deliverable rather than evidence for this scripted pass.
