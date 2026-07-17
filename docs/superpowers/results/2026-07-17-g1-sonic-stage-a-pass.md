# G1 SONIC Stage A Baseline Pass

Date: 2026-07-17 UTC

## Verdict

The registered Stage A integration prerequisite now passes. The flat
motion-matching reference passes its kinematic replay, and the pinned,
unmodified G1 SONIC controller completes both dynamic paths using the canonical
official known-good reference. The unchanged `known-good-stream` command exited
`0` with `status=pass` and `stage_a_status=pass`; all seven ordered gates passed.

Stage A deliberately does not stream the generated motion-matching reference
through SONIC. That is the 601-frame, 12-second Stage B experiment. Therefore
this checkpoint clears the prerequisite for actual MM-to-SONIC dynamic testing;
it does not by itself establish that final composition claim.

This is a simulation qualification: the official SONIC policy and controller
drove the registered G1 MuJoCo system from both its file and ZMQ stream input
paths. It is not a claim that the policy has been deployed on a physical G1.
The retained tracking errors are diagnostic Stage A measurements, not a claim
of high-accuracy tracking or a physical-robot safety certificate.

Qualified production commit:

```text
f8abb375bff51b8d82cbbce7e562c95ade52428a
```

Pinned GEAR commit:

```text
60de0df7ffedeef415fe58d435e92cc5b01ba3d9
```

Both repositories were clean in the finalized manifest.

## Final blocker and repair

The last failed qualification had isolated an upstream timing race rather than
a policy or transport defect. GEAR's `WAIT_FOR_CONTROL` state calls
`CheckSafety()` every 20 ms and rejects a `LowState` older than 500 ms. The
stream preload authenticates 25 publications and takes about 0.57 seconds.
The simulator stopped publishing immediately after cold bootstrap, so stream
preload could consume the complete 500 ms freshness budget before the scored
reset. Publishing one scored-reset state while the entire GEAR process group
was stopped did not guarantee that its DDS callback would run before the
resumed control thread's safety check.

Commit `f8abb37` uses the existing bounded `_drive_simulator_until` primitive
around input preparation in both modes. File arming and the complete stream
enable/preload/consumer-wait sequence therefore receive continuous unscored
LowState publication while GEAR remains in authenticated `WAIT_FOR_CONTROL`.
The existing zero-row audit, stopped process-group reset, exact one-step scored
prime, CONTROL activation, asynchronous score counters, and exact 441-row
stop fence remain unchanged.

The finalized evidence separates this maintenance from scored physics:

| Mode | Unscored preparation maintenance | Scored prime | CONTROL-drive steps | Exact target rows |
|---|---:|---:|---:|---:|
| File | 20 steps / 0.120014076 s wall | 1 | 1,438 | 441 |
| Stream | 90 steps / 0.566605723 s wall | 1 | 1,403 | 441 |

The stream sequence now reaches the authenticated CONTROL transition instead
of exiting with `Lost LowState data connection from robot!`.

## Exact Stage A run

The only experimental change from prior trials was a fresh output root and the
qualified code revision:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python CUDA_VISIBLE_DEVICES=0 \
  sonic/.venv/bin/python -u -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout /tmp/groot-wbc-plan-inspect \
  --policy /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/model_decoder.onnx \
  --observation-config /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/observation_config.yaml \
  --encoder /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f/model_encoder.onnx \
  --source-mjcf /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/runs/stage-a-official-sonic-f8abb37-20260717
```

Immutable run:

```text
/home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/runs/stage-a-official-sonic-f8abb37-20260717/stage-a/known-good-stream-20260717T161308279711Z-87d1d58d
```

The run was created at `2026-07-17T16:13:08.287173Z` and finalized at
`2026-07-17T16:15:16.060332Z`.

## Seven-gate result

| Gate | Status |
|---|---|
| `external_identity` | pass |
| `joint_projection_round_trip` | pass |
| `basis_and_scene_alignment` | pass |
| `flat_mm_kinematic_replay` | pass |
| `known_good_file_dynamic` | pass |
| `known_good_stream_delivery` | pass |
| `known_good_stream_dynamic` | pass |

The dynamic file and stream runs used an identical launch identity
(`e2d0cf1766a8d1638e6c52cff9ab92da67621dc2f34c8b6308f9b78f9a2fa33f`).
Both authoritative target logs contain exactly 441 rows, frames 0 through 440,
and have the same SHA-256:

```text
744500f1d620f9a14f09d7f8df9dcac0b58c74281e187c469196bd17a9592450
```

Both target audits report `exact=true`, `post_stop_stable=true`, and
`final_close_stable=true`. Each `q.csv` and `base_quat.csv` contains one header
plus exactly 441 same-control-tick state rows. The stream delivery audit proves
22 logical publications covering 440 non-readiness frames, one readiness
publication, one 46-frame padding publication, one receipt fence, 25 exact
consumer markers, and 24 causal fences.

Retained dynamic measurements:

| Mode | Joint-position RMSE | Pelvis-orientation RMS | Mean tick period |
|---|---:|---:|---:|
| File | 0.8601554507 rad | 1.6764699122 rad | 19.9999818184 ms |
| Stream | 0.8112837765 rad | 1.6759457090 rad | 19.9999227264 ms |

The raw/search feasibility certificate also remained stable:

```text
frame_count=459682
raw_safe_count=458619
raw_unsafe_count=1063
search_safe_count=458274
mask_sha256=cce20d9b5d2dd4ed4013a1d6416f3402e564b013aebdfefc15bd1a468b494e25
```

## Immutable evidence identities

- `stage-a-evidence.json`:
  `78de0087d841989acde414a0ebfeaf993d31d91e1ae89b5e8858b830f0cc443c`
- `manifest.json`:
  `642b33456105a63b5393798d1d7e9b556317cf2dbbad709bf0812ac871baf159`
- `inventory.json`:
  `fa01b6574693037189ae3c8e7c333dd9596d092bad7172bb95827d6053121888`
- Gate 5, `known_good_file_dynamic.json`:
  `12436b1b279248453d9856434e4d665812ac86d58e8241658addaa4f786f5d81`
- Gate 6, `known_good_stream_delivery.json`:
  `e3f960ab33b1a0b3590914090e382aebe3d8b2284a0bec03373b8c7b3ae842af`
- Gate 7, `known_good_stream_dynamic.json`:
  `d151c6f5bc715745caa491d3a1ec768d494910dcb717659639f0779aa5f73aea`

The finalized inventory contains 165 files, and a fresh
`verify_run_inventory(...)` call returned `True`.

## Code and protected qualification

The focused lifecycle regression was first RED at parent `c8f07e9`: file input
preparation ran without simulator drive. It is GREEN at `f8abb37` and proves
the exact ordered phases:

```text
file-wait-for-control
file-input-preparation
stream-wait-for-control
stream-input-preparation
```

The warning-strict timing module passed all 21 tests. The complete supported
18-module Python catalog, with the exact prebuilt projection oracle bound by
`SONIC_PROJECT_CLI`, passed:

```text
Ran 437 tests in 48.492s
OK (skipped=6)
```

Its live C++ projection certificate covered 260 samples and all 29 source and
target joints, with maximum joint-angle reconstruction error
`2.51585969e-07` rad.

The controller-owned protected evaluator is independently RED at `c8f07e9`
and GREEN on the exact two-file patch. `git diff --check` passed, the Generic
Coordinator SHA-256 remained
`8aede89a11132674c62c0cc051a546103f0c4f56f9fbd3256701bf7e5599f92d`,
and the pinned GEAR checkout remained clean.

## Reliable Claude supervision

The exact detached `11f44df060fa011db504199917beb3e8bb5200dd` Reliable
Claude release supervised job `job-31f7d0a817a5e1436beb85fe8a129f4c` with
plugin `0.1.1+codex.20260716220019`. The single Claude attempt made no
repository progress in `184.46275671292096` active seconds. The supervisor
correctly raised durable `attention-001`; no repeated conversational retry was
used. Codex recorded a takeover, applied the exact qualified two-file patch to
the retained clean target, and reran the protected evaluator plus all 21 timing
tests successfully.

This is another data point that the Claude worker was not productive on the
motion-control implementation, while the updated supervisor was useful in
bounding the stall, preserving a clean handoff, and preventing it from delaying
the real acceptance run.

## Boundaries after qualification

No Stage A controller, gated simulator, `g1_deploy_onnx_ref`, or motion-matching
server process survived finalization. Unrelated GPU and Claude processes were
not modified or interrupted. No push, merge, tag, physical-robot command, or
runtime activation was performed.

The prerequisite that blocked the scripted flat MM-to-SONIC experiment is now
satisfied. Stage B remains the next required result before any claim that the
motion-matching output itself dynamically drives SONIC. Physical G1 deployment,
pick/carry/place, and obstacle-task expansion remain separate, explicitly gated
work.
