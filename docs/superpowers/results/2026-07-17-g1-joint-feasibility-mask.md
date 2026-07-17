# G1 Joint-Feasibility Mask Qualification

Date: 2026-07-17 UTC

## Verdict

The authenticated raw-frame feasibility mask is implemented and qualified. It
removed the original unsafe raw database progression from frame 866 to frame
867. The unchanged seven-gate Stage A baseline still does not pass: gate 4
stops at chunk 5 after 101 valid frames because a repeated transition into raw
frame 866 accumulates inertialization and drives `left_ankle_roll_joint` to
`-0.280130744`, below the registered lower limit `-0.261799991`.

This is a truthful next scientific result, not a Stage A pass. The database
certificate, masked search, Python identity binding, and live projection gate
all behaved as designed. The next intervention must address transition-time
inertialization feasibility without clipping poses, widening limits, removing
the hard projection gate, or weakening evidence contracts.

## Scope and commits

- Required base: `05014911828c3b52b38a1651fecf3760ef8f03ee`
- Implementation qualification HEAD: `0cc200ee57169eddf7d8b0e7f52987596a565ecf`
- Worktree: `/home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline`
- Branch: `g1-sonic-scene-aware-baseline`

| Task | Commit | Qualified behavior |
|---|---|---|
| Structured projection diagnostics | `d56fa29`, `6cd4752` | Transactional diagnostic API; non-finite positions remain fatal rather than maskable |
| Deterministic certificate | `42544ff` | One-time raw/search-safe scan, reconciled counts, successor-aware mask, stable SHA-256 |
| Masked matcher/runtime | `207027b` | Unsafe incumbents/candidates excluded; unsafe progression forces search; unmasked API unchanged |
| Server ownership/protocol | `989ec5b` | Certificate owned for server lifetime and published in strict protocol-v1 hello identity |
| Python orchestration binding | `0cc200e` | Strict seven-key parser, preflight/run identity cross-binding, retained evidence, exact scientific classification |

The live `sonic_project_pose` call remains enabled after selection and
inertialization. Its failure is what stopped the real run.

## Test-first implementation evidence

Each task began with the focused tests described in
`docs/superpowers/plans/2026-07-17-g1-joint-feasibility-mask.md` and observed the
expected missing-interface or behavioral RED before production changes.

- Task 1: the warning-strict projection test initially failed to compile
  because the structured overload did not exist. The GREEN binary passed all
  legacy and structured cases. Review then found non-finite position values
  could be misclassified as limits; the added regression was RED, commit
  `6cd4752` restored the fatal classification, and the protected evaluator
  passed.
- Task 2: the certificate test initially failed to compile because
  `g1_joint_feasibility.h` did not exist. The final warning-strict and
  ASan/UBSan certificate tests passed, including transactional fatal errors,
  range-end clamping, successor masking, count reconciliation, and digest
  sensitivity.
- Task 3: masked database and runtime tests initially failed against the
  unmasked interfaces. The final warning-strict and ASan/UBSan binaries passed,
  including masked incumbent rejection, count mismatch, forced search before
  unsafe progression, exact no-safe-candidate failure, and retention of the
  live inertialized projection gate.
- Task 4: hello tests were RED because `joint_feasibility` was absent. The final
  fake-server surface passed 14 tests with two guarded real-artifact skips;
  warning-strict server construction and Task 1--3 regressions passed.
- Task 5: schema, scene, and CLI tests were RED on the absent parser and binding.
  The required final suite passed 110 tests with seven guarded skips in
  `9.430s`; the adjacent surface passed 118 with two skips in `29.226s`; the
  protected Python 3.10 warning-strict run passed 110 with six guarded skips in
  `9.457s`. Independent review found no Critical, Important, or Minor findings.

Fresh protected evaluators after `0cc200e` all passed. Their bound production
and probe SHA-256 identities were:

- Projection: `sonic/cpp/g1_joint_projection.h`
  `feae1193e858fa173951f16d7ed3fd42a621a40a23bfab8ef107a55513611c11`;
  probe `52894f3ba675a345e98ec65eb262df7d15ab6c492d4a10d97dcbe2c07abcf58c`.
- Certificate: `sonic/cpp/g1_joint_feasibility.h`
  `cfb75c97e64b50562a0a293b0e5eaee6ae3f833810c604a4bf66e8728cd56221`;
  probe `b28709e1d5bbbb25b02ea15cf7140aac20a13f615c1f13b0f4cba39bba6e0edb`.
- Masked runtime: `database.h`
  `1a6495d938db97368f4f0f7f81b7ae8e62d969921588fc4b98c1176264e88f6b`,
  `sonic/cpp/g1_runtime.h`
  `4f83d92437f34837aece7ed8c3f18eb52737795ca226ebad7fadc254dcf73637`,
  and probe
  `2a6e71240033d49f4a0dc1795c739ffad38926fe5ab1c6c0dd9b72f1ed8bd8d8`.

## Final code qualification

The exact warning-strict C++ loop compiled and ran these four binaries with
exit 0:

```text
test_g1_joint_projection
test_g1_joint_feasibility
test_terrain_database
test_g1_runtime
```

The first attempt at the 18-module Python catalog used the isolated verifier
environment literally from the plan and traversed 360 tests, ending with one
failure, six errors, and six skips. Diagnosis showed environment preconditions,
not product failures: that venv lacked `jsonschema`, the clean build had removed
`route_schedule_cli`, and `SONIC_PROJECT_CLI` was unset. No result from that
attempt was counted as qualification.

After rebuilding `mm_chunk_server`, `route_schedule_cli`, and the warning-strict
`g1_project_pose_cli`, the same exact 18 modules ran in the supported project
Python 3.10.13 environment with `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONWARNINGS=error`, and the absolute `SONIC_PROJECT_CLI`. Result:

```text
Ran 411 tests in 50.957s
OK (skipped=6)
```

A fresh pre-commit rerun of the identical catalog also passed 411 tests with
six skips in `52.199s`; the C++ loop and all retained evidence assertions were
re-run in the same verification wave and exited 0.

Qualified executable SHA-256 identities:

- `mm_chunk_server`: `6ddca2f3d6d6fd7d07ba134059f0383d24c93f3acf4a6c998b67de9db9ace44a`
- `route_schedule_cli`: `b48208fe7eb4fbb3d2757698628bbee8fdcedfb2f6024d322323be36f1c3ba4b`
- `g1_project_pose_cli`: `a923539a7e130c4ee888ff363294d128b4afaffe85d7263606874af7acd031ad`

The server embeds implementation HEAD `0cc200ee57169eddf7d8b0e7f52987596a565ecf`.

## Real certificate

Two independent fresh real-server launches returned the same strict hello
identity:

```json
{
  "schema": "g1-joint-feasibility-certificate/v1",
  "frame_count": 459682,
  "raw_safe_count": 458619,
  "raw_unsafe_count": 1063,
  "search_safe_count": 458274,
  "joint_limit_violation_count": [0, 0, 0, 0, 20, 938, 0, 0, 0, 4, 14, 52, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
  "mask_sha256": "cce20d9b5d2dd4ed4013a1d6416f3402e564b013aebdfefc15bd1a468b494e25"
}
```

The 29 violation counts sum to 1,063. Both launches loaded the authenticated
459,682-frame database and produced the identical digest.

## Real Stage A run

The experiment changed only the output root from the prior known-good-stream
command:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=sonic/python \
  sonic/.venv/bin/python -B -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout /tmp/groot-wbc-plan-inspect \
  --policy /tmp/groot-wbc-plan-inspect/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx \
  --observation-config /tmp/groot-wbc-plan-inspect/gear_sonic_deploy/policy/release/observation_config.yaml \
  --source-mjcf /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --output-root /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/runs/stage-a-joint-feasibility-integrated-20260717
```

The command exited 3 with `status=scientific_failure` and
`stage_a_status=failed`. The immutable run is:

```text
/home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/runs/stage-a-joint-feasibility-integrated-20260717/stage-a/known-good-stream-20260717T104648535827Z-6a17fce5
```

Gates 1--3 passed. Gate 4 reported:

```text
status=scientific_failure
failed_chunk=5
failure_boundary=source_chunk
initial_boundary_accepted=true
completed_chunks=5
partial_frame_count=101
required_chunks=30
required_frame_count=601
generation_failed: joint left_ankle_roll_joint position -0.280130744 is outside range [-0.261799991, 0.261799991]
```

Gates 5--7 are `not_run` because gate 4 blocks them. Artifact SHA-256 values:

- `stage-a-evidence.json`: `bb939e7b351cbeff85627bfaef4a1e7e7d9451dd825f87ece204a663edd67025`
- `manifest.json`: `d84fab8eb8f8aab81040983b71c1b91d99ea61509d53d9272fead95f8ea21c61`
- `inventory.json`: `e6336158b0e646f5c34b75818040d4ece84ac8ee50287d7976a7c357e7cb767d`
- Gate 4: `a3cef3ba0c94381f362befd48052e5a07a05f75e4912a39847ff98ceb403c810`

`verify_run_inventory(...)` returned `True` for the finalized run.

## Failure localization

An out-of-tree, read-only C++ diagnostic harness replayed the public real adapter
at the failure boundary without changing production code:

| Step | Selected/state frame | Transition | Raw left ankle roll | Inertialized result |
|---|---|---:|---:|---:|
| 50 | 50 -> 51 | no | `-0.246654257` | `-0.246654257` (safe) |
| 51 | 865 -> 866 | yes | `-0.190088332` | `-0.254079133` (safe) |
| 52 | 865 -> 866 | yes | `-0.190088332` | `-0.280130744` (limit failure) |

The former selected-866-to-emitted-867 path is absent. Raw frame 866 projects
inside the registered limits on both repeated searches. The second transition's
inertialized pose, not the raw database pose or its successor mask, crosses the
limit. This localizes the next research problem to transition/inertialization
feasibility.

## Repository and runtime boundaries

At qualification HEAD, the motion-matching worktree and pinned GEAR checkout at
`60de0df7ffedeef415fe58d435e92cc5b01ba3d9` were clean. No
motion-matching `mm_chunk_server`, Stage A controller, gated simulator, or
`g1_deploy_onnx_ref` process survived. Unrelated Reliable Claude work in other
projects is outside this qualification and was not modified or interrupted.

No push or merge was performed.
