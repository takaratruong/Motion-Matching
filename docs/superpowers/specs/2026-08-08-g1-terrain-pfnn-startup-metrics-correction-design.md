# G1 Terrain PFNN Startup-Metrics Correction

## Status

Approved narrow correction cycle. This design supplements the physical-envelope
objective design; every objective, runtime, dataset, checkpoint, report, and
promotion contract remains unchanged.

## Evidence and root cause

The sole physical-envelope-v1 artifact attempt stopped correctly before update
step 1. The Task 7 artifact report records
`KeyError: 'joint_step_envelope'` from `one_step_metrics` while computing the
initial score. It produced no checkpoint, fitted report, evaluator callback,
promotion receipt, or immutable best. The failed directory is immutable
evidence and contains only its zero-byte `metrics.jsonl`.

An independent read-only audit confirmed the complete cause:

- `one_step_metrics` is a base-only, isolated-row evaluation path;
- it calls `pfnn_losses`, which intentionally returns the base losses plus
  `total`, never predecessor-relative envelope losses;
- it selects aggregation keys with `LOSS_WEIGHT_KEYS[:-1]`;
- physical-envelope work appended three envelope weights after
  `regularization`, so positional slicing now requests two envelope keys; and
- mocked trainer probes replaced `one_step_metrics` and returned all loss
  weights, masking the real call contract.

The failure is deterministic and occurs at both initial and final one-step
metric sites. It does not question the accepted recurrence-v2 dataset, target
feasibility, predecessor-pair population, physical-envelope objective, fitted
gate, or known-terrain evaluator.

## Selected correction

Inside `one_step_metrics`, define the aggregate names only as an explicit tuple
derived from `BASE_LOSS_WEIGHT_KEYS` while excluding the named
`"regularization"` member:

```python
names = tuple(
    name for name in BASE_LOSS_WEIGHT_KEYS if name != "regularization"
)
```

No positional slice is permitted. The resulting ordered tuple is exactly:

```text
trajectory_mse
body_mse
root_pose_mse
joint_mse
root_motion_mse
phase_mse
contact_bce
trajectory_direction
phase_nonnegative
fk_consistency
```

`one_step_score` remains the sum of those ten batch-size-weighted means at the
existing immutable weight `1.0`. Regularization remains an optimization term
but is excluded from evaluation as documented. Envelope objectives remain in
predecessor-aware one-step training, rollout training, complete fitted-pair
diagnostics, and the unchanged authorization gate; they are not reconstructed
from isolated evaluation rows.

## Alternatives rejected

- Adding envelope keys to `one_step_metrics` would require predecessor targets,
  change the score contract, and duplicate the complete fitted-pair gate.
- Catching `KeyError` or silently skipping missing keys would convert future
  loss-contract drift into an unreviewed score change.
- Replacing the slice with a different position-based slice would retain the
  same coupling that caused the failure.
- Resuming, deleting, or overwriting the failed v1 output would destroy the
  causal evidence and is forbidden.

## Scope and immutable contracts

The only source/test changes are:

- `sonic/python/mm_sonic/terrain_pfnn/training.py`: replace the aggregation-key
  selection in `one_step_metrics`;
- `tests/python/test_terrain_pfnn_training.py`: add a direct uneven-batch
  regression and tighten the mocked trainer probe to the real metric schema.

There is no refactor, new helper, fallback, schema migration, threshold change,
or artifact repair. In particular:

- model and layout remain `288 -> 512 -> 512 -> 268`;
- dataset remains `mm-sonic-terrain-pfnn-dataset/v2`;
- checkpoint remains `mm-sonic-terrain-pfnn-checkpoint/v6`;
- one-step report remains `mm-sonic-terrain-pfnn-one-step-report/v2`;
- fitted-transition report remains
  `mm-sonic-fitted-transition-report/v1`;
- promotion receipt remains `mm-sonic-pipeline-promotion-receipt/v3`;
- snapshot remains `mm-sonic-dataset-file-snapshot/v1`;
- objective constants, loss weights, pair receipts, target audit, sampler,
  recurrence, runtime, controller, raw outputs, and physical gates remain
  unchanged; and
- the sealed test split remains closed until original Task 9.

## Verification design

The regression must call the real `one_step_metrics` and real `pfnn_losses` on
three deterministic rows with batch size two. It independently computes the
same ten full-batch base means, requires the exact returned key order, requires
`samples == 3`, and requires the exact weighted score. The uneven final batch
proves aggregation is sample-weighted rather than batch-weighted.

The existing trainer probe may continue mocking `one_step_metrics` to isolate
pair-training flow, but its payload must contain only the ten base evaluation
keys plus `one_step_score` and `samples`. It must not advertise
`regularization` or any envelope key.

After the focused test and single source/test commit, run the complete
dependency-split 204-test PFNN gate, all static checks, and a prescribed
real-data GPU-1 smoke that invokes the real `one_step_metrics` on the exact
2,048-row recurrence-v2 subset without constructing an optimizer, taking an
update, or creating a run directory. An independent reviewer must approve both
the narrow diff and the full `d94b443..startup-fix-head` range before the source
is frozen.

## Controlled artifact rerun

The failed path
`sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1` remains
untouched. A successful review authorizes one fresh attempt only at:

```text
sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2
```

Fresh before/after dataset snapshots must prove the exact six accepted files
are unchanged around the exact builder `--resume`. GPU 1, frozen source hashes,
dataset digest, split isolation, target audit, objective digest, and pair
receipt must be reconfirmed before the command.

The attempt remains exactly 4,000 one-step updates plus 256 rollout16 updates
at seed 7. Training acceptance requires finite losses, safe v6 reload, exact
fixed-sample reproduction, and acceptance of all 2,011 fitted pairs before the
single exact 600-tick evaluator may run. Only an accepted evaluator may publish
`best.pt` and a v3 receipt.

At the first startup, training, reload, fixed-sample, fitted, or closed-loop
failure, record exact evidence and stop. There is no retry, resume, output
patch, threshold/margin/tail/weight change, source/test edit, or commit after
the frozen run boundary. A failure requires another separately approved cycle.

## Downstream handoff

If and only if the v2 artifact, promotion, safe reload, bound receipts, and
post-artifact tests all pass, independently re-review original Task 7 across
its full frozen range. A blocker-free review then resumes original Tasks 8 and
9 and ends with the already-required interactive W/A/S/D three-hill drive. No
downstream task may alter this frozen correction or the accepted checkpoint.
