# Full Walking Terrain LMM — Task 7 Report

## Scope

- Base: `6bf5e41`
- Branch: `feature/full-walking-task7`
- Worktree: `/home/ubuntu/worktrees/motion-matching-full-walking-task7`
- Implemented only Task 7 evaluation/viewer files and the strictly required
  legacy viewer rate bridge.
- No corpus, model, evidence, or interactive artifacts were launched.

## TDD evidence

The first focused run failed during collection exactly because both production
modules were absent:

```text
ModuleNotFoundError: No module named
'mm_sonic.full_walking_terrain_lmm_evaluation'
ModuleNotFoundError: No module named
'mm_sonic.full_walking_terrain_lmm_viewer'
2 errors in 0.21s
```

A second RED proved the existing interactive viewer had no matcher-rate bridge:

```text
ImportError: cannot import name '_runtime_step_hz'
1 error in 2.38s
```

## Implemented contract

- Immutable, validated continuous-time `FormalRoute` sampled with zero-order
  held commands at exact system rates.
- `SlipAccumulator` counts planar probe displacement only when a probe remains
  planted across both ends of an interval.
- `EvaluationSeries` validates post-`mj_forward` probes, contacts, world root,
  desired speed, query distance, canonical source, forward count, and safety
  counters.
- `compare_baseline_candidate` samples one route authority at frozen 25 Hz and
  candidate 60 Hz, publishes deterministic per-terrain/aggregate coverage,
  speed, slip, source diversity, command authority, and raw counts.
- The formal route evaluator performs real MuJoCo forwarding and samples the
  authenticated four-corner native ankle probes.
- Formal acceptance fails closed on full corpus/model provenance, all 15,918
  terminal identities, all 80 PFNN BVHs, split/test/refit/determinism receipts,
  exact full search, canonical XML/assets, all four authenticated scenes,
  motion-derived root ownership, finite retry budget, zero runtime safety
  events, 10,000 forwards, coverage/slip/speed thresholds, and two canonical
  identities per terrain.
- `smoke|view` CLI matches the Task 8 commands. Baseline evaluation consumes
  the candidate corpus's authenticated scene pack so both systems see the same
  terrain bytes.
- The legacy interactive viewer remains 25 Hz when no authenticated matcher
  rate exists and steps full walking matchers at their exact 60 Hz rate.

Task 6 interface alignment was checked directly against its in-progress
worktree: `fps`, `walking_speed_p95_mps`, `candidate_retry_budget`,
`candidate_exhaustion_count`, and `root_motion_source` are consumed without
editing Task 6 files.

## Fresh verification

```text
pytest evaluation + full viewer + legacy viewer: 40 passed in 8.41s
ruff check scoped production/tests: All checks passed!
py_compile scoped production/tests: exit 0
git diff --check: exit 0
```

The formal 10,000-frame run remains Task 8 and must use the final integrated
Task 4 corpus, Task 5 model, and Task 6 runtime artifacts.
