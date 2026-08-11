# Full-walking single-GPU search evidence

Status: **pre-final-fix diagnostic operational evidence**. The final listener
fix commit and its replacement-process validation are pending. This report does
not claim that post-fix validation has happened.

Date: 2026-08-10

## Scope and evidence boundary

This report durably preserves the real single-GPU benchmark and live-listener
evidence that existed at the `dcf2b59` code baseline. It is diagnostic-only:

- the full-row device score is FP32, not the formal CPU float64 search contract;
- the model's frozen test receipt remains red, and the viewer remains visibly
  labeled diagnostic and acceptance-ineligible;
- the `p95 <= 100 ms` threshold is an engineering latency target, not a formal
  model, motion-quality, or sim-to-real acceptance gate;
- exactly one physical GPU was selected. Multi-GPU execution was neither used
  nor tested, and this report makes no multi-GPU claim; and
- the live listener evidence below predates the pending final fix. It cannot be
  substituted for a fresh post-fix PID, window, and physical-input check.

No production file, test, threshold, corpus, model, existing task report, or
acceptance receipt is changed by this evidence report.

## Exact commit baseline through `dcf2b59`

The arrow-listener chain present in the pre-final-fix baseline is:

- `fcd7dbddc0ccc6e79b81ada761eb571a7a7489b1` — `docs: specify arrow-key viewer controls`
- `cc7cbb42ba97987c68a7ef39cca81303ba130cc0` — `docs: plan arrow-key viewer controls`
- `8f7acc55bad866aa8bcfb6c16e6657dc73b55bd8` — `fix: drive terrain viewer with arrow keys`

The single-GPU design, implementation, review-fix, viewer, and evidence chain is:

- `41c33bca2b204f5596a296d01142d049cbc42a28` — `docs: design single-gpu exact motion search`
- `f6f6d9aecde9206be5de548b57036888d95a767d` — `docs: plan single-gpu exact motion search`
- `60b7c90df6d6263353d4c82c557c92f5568d4b47` — `perf: avoid redundant contact exclusion sorting`
- `3be6fe515acc0210b25d5e4ab078ac7a8625c761` — `docs: avoid captured GPU search constants`
- `2b34dac6c327930732f7f5bdeb0e776bf4e73dd6` — `feat: add single-gpu exact motion scorer`
- `4fd33b8f01339423c8e51e8d2bdc56761a3b2c1b` — `feat: add single-gpu full-row motion scorer`
- `be30eae34626a86040b5a18956877f0aa15db947` — `fix: return full GPU candidate set`
- `fe1543db9bcee8475873cefd23af0abfc314dce6` — `docs: bind diagnostic GPU search to FP32`
- `0a6a26a348ea2abb224d8bba75d49ea5ef894fb4` — `test: verify GPU transition penalty minimum`
- `d2deaa24bb6e76cb1e29dc28b9dde6576f00c04e` — `feat: use single-gpu full-row motion search`
- `b9aca21e79050cc9904508f95782766d0b2216aa` — `fix: validate single-gpu runtime responses`
- `b97860011eb8f8ae58cec78193b6a94fed7e698d` — `fix: filter gpu successor retry exclusions`
- `3ff65b09f89f8b9217e17cbf4b85ebee7a808fb1` — `feat: expose single-gpu diagnostic motion search`
- `dcf2b59ce7fff4f3c82dfa3173bf8b7584407515` — `docs: record GPU5 viewer benchmark`

`dcf2b59ce7fff4f3c82dfa3173bf8b7584407515` is the last commit represented by
the operational evidence below. A final listener fix commit is pending and is
intentionally not invented or claimed here.

## Search architecture represented by this evidence

The viewer request `--search-device cuda:5` isolates physical GPU index 5
before JAX import. The lazy backend transfers the immutable native-FP32 feature
table, sorted searchable global rows, range IDs, and two-bit contact codes to
that single device. It does not use `pmap`, multi-device sharding, replication,
or another GPU.

Every eligible range-safe row is scored on GPU5 with FP32 subtraction,
squaring, and accumulation. The frozen `0.1` transition penalty is added
outside the current range; incompatible contacts and explicit retry exclusions
are masked. The scorer returns the complete finite stable device top-128 and a
separate FP32 near-minimum count. `HybridMatcher` then applies its existing CPU
float64 candidate score to all returned rows and selects the stable lowest
global row. If the near-minimum set overflows the returned set, the runtime uses
the existing complete CPU brute-force authority. A requested GPU failure is
explicit; it does not silently cap rows or change to the default CPU path.

The CPU cKDTree/float64 path remains the default when no search device is
requested. Runtime identity distinguishes `cpu-ckdtree-exact` from the measured
`single-gpu-full-row-fp32:cuda:5`. The live overlay recorded
`9741525/9741525` searchable range-safe rows from the 9,758,524-row corpus, so
the diagnostic GPU path was not a sampled or capped search.

## Previously recorded automated and static evidence

These are the exact results already recorded by the owning task reports. They
are historical verification evidence through `dcf2b59`, not a fresh test of
the pending final fix.

- Arrow-listener normalization at `8f7acc55bad866aa8bcfb6c16e6657dc73b55bd8`:
  the focused hybrid/full viewer run recorded `46 passed in 12.11s`. Ruff check
  reported `All checks passed!`, Ruff format reported `2 files already
  formatted`, `py_compile` exited zero, and `git diff --check` exited zero.
- Empty-exclusion fast path at `60b7c90df6d6263353d4c82c557c92f5568d4b47`:
  the focused regression recorded `1 passed, 40 deselected in 2.51s`; the full
  runtime file recorded `41 passed in 4.51s`. Ruff check/format and
  `git diff --check` were clean.
- FP32 scorer through `0a6a26a348ea2abb224d8bba75d49ea5ef894fb4`:
  the final scorer file recorded `15 passed in 6.06s`; scorer plus runtime
  recorded `56 passed in 10.95s`; post-commit aggregate verification recorded
  `56 passed in 10.80s`. Ruff check/format, `py_compile`, and
  `git diff --check` were clean. The GPU5 subprocess asserted one visible JAX
  GPU and covered 41 seeded CPU-rescore cases, all contact/range combinations,
  penalty and exclusion winner changes, duplicate ties, 140 exact device ties,
  and FP32 overflow cases.
- Runtime integration through `b97860011eb8f8ae58cec78193b6a94fed7e698d`:
  the final focused review-fix run recorded `9 passed, 41 deselected in 2.40s`;
  the complete runtime plus scorer run recorded `65 passed in 10.38s`. The
  recorded Ruff check/format, `py_compile`, and `git diff --check` were clean.
- Viewer exposure at `3ff65b09f89f8b9217e17cbf4b85ebee7a808fb1`:
  the full-viewer focused run recorded `17 passed in 2.35s`; the shared hybrid
  viewer recorded `37/37`; the combined scorer, runtime, hybrid-viewer, and
  full-viewer verification recorded `119 tests in 20.64 seconds`. Ruff
  check/format, `py_compile`, and `git diff --check` were clean.

## Real benchmark artifact provenance

The benchmark source was read directly in full before writing this report.

| Field | Exact value |
| --- | --- |
| Path | `/tmp/full-walking-gpu5-benchmark-20260810T184806.log` |
| Size | `3917` bytes |
| SHA-256 | `2bfac29a8fa2847cb6230347bc17a22b8bee3af7ca0f38a1814e255fa5a988fb` |
| File mtime | `2026-08-10 18:53:33.620653670 -0700` |
| Schema | `full-walking-single-gpu-search-benchmark/v1` |
| Benchmark PID | `1396969` |
| Backend | `single-gpu-full-row-fp32:cuda:5` |
| Query count | `25` |

### Complete selected rows

The log's complete `selected_rows` array, in query order, is:

```json
[72062, 5686, 2268948, 3648801, 255976, 246706, 2114886, 364403, 635794, 691048, 2679932, 1607958, 1415074, 994750, 1728969, 2114886, 364403, 635794, 691048, 2679932, 1607958, 1415074, 994750, 1728969, 2114886]
```

### CPU float64 brute-force parity samples

All three samples recorded `equal: true`, with identical row and distance on
the GPU-candidate/CPU-rescore path and full CPU float64 brute force.

| Query index | GPU row | CPU row | GPU distance | CPU distance | Equal |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `72062` | `72062` | `0.5644492261373896` | `0.5644492261373896` | `true` |
| `12` | `1415074` | `1415074` | `1.6329216054089495` | `1.6329216054089495` | `true` |
| `24` | `2114886` | `2114886` | `3.6107735601899673` | `3.6107735601899673` | `true` |

### Startup, device-search, and wall-step timings

All values below are milliseconds and retain the log's full numeric precision.

| Measurement | Exact value |
| --- | ---: |
| Matcher transfer/JIT startup | `3914.6297950064763` |
| Total cold startup, including corpus/model authentication | `318781.5433839569` |
| First runtime search | `4.64080402161926` |
| Device-search median | `4.395122989080846` |
| Device-search p95 | `4.868493741378186` |
| Device-search maximum | `5.498292041011155` |
| Wall-step median | `9.728798060677946` |
| Wall-step p95 | `13.555587455630299` |
| Wall-step maximum | `29.51011701952666` |

Complete `device_search_ms.samples`, in query order:

```json
[4.3753329664468765, 4.9663379322737455, 4.400082980282605, 4.467205959372222, 4.4771169777959585, 4.454665933735669, 4.37020102981478, 4.380922997370362, 4.467227030545473, 4.389032954350114, 4.3857620330527425, 4.2916679522022605, 4.426473984494805, 5.498292041011155, 4.42150398157537, 4.3673820327967405, 4.434484988451004, 4.34438104275614, 4.395122989080846, 4.384023020975292, 4.372722003608942, 4.343481035903096, 4.348512040451169, 4.405492916703224, 4.401843994855881]
```

Complete `wall_step_ms.samples`, in query order:

```json
[9.595870971679688, 12.401225045323372, 9.792761062271893, 10.132745956070721, 9.798550978302956, 9.944947087205946, 29.51011701952666, 10.216288967058063, 9.456856059841812, 9.671234991401434, 9.423753945156932, 10.020090965554118, 10.107653914019465, 13.844178058207035, 9.668175014667213, 9.733528015203774, 9.385191951878369, 9.117641020566225, 9.700365946628153, 9.728798060677946, 9.711347054690123, 9.371092077344656, 9.620013064704835, 9.80196101590991, 9.346061036922038]
```

The measured device-search p95 of `4.868493741378186 ms` is below the
diagnostic plan target of `100 ms`. This timing result does not alter the formal
acceptance boundary.

### Physical-GPU ownership and all before/after snapshots

Physical GPU5 is UUID
`GPU-6a819546-e32a-076b-a85e-0e539d6fd9df`. The benchmark ownership record
contains only PID `1396969` on that UUID with `2488` MiB process memory.

The complete instantaneous before/after snapshots are:

| Physical index | UUID | Before memory MiB | Before util % | After memory MiB | After util % |
| ---: | --- | ---: | ---: | ---: | ---: |
| `0` | `GPU-c5b2d7ba-9b1a-432e-db95-aa28fa96706f` | `6196` | `47` | `6196` | `48` |
| `1` | `GPU-b21c32d0-4182-695a-21f1-146bc75d1c6c` | `727` | `0` | `727` | `0` |
| `2` | `GPU-681a3aad-4716-3b53-c1ea-576c5745a703` | `793` | `0` | `793` | `0` |
| `3` | `GPU-87fb0777-4169-d4e9-12cc-382bdf7c0730` | `731` | `0` | `731` | `0` |
| `4` | `GPU-ffb6a882-6f1f-9860-54a2-e67ad10e2aab` | `1303` | `0` | `1303` | `0` |
| `5` | `GPU-6a819546-e32a-076b-a85e-0e539d6fd9df` | `117` | `0` | `2609` | `0` |
| `6` | `GPU-9eee69bf-e487-9730-1deb-f677aa1acfc4` | `117` | `0` | `117` | `0` |
| `7` | `GPU-67a0d2ad-30c6-fde9-8d1f-f1e7aee354ca` | `118` | `0` | `118` | `0` |

Only physical GPU5 gained memory in these snapshots (`+2492` MiB). The
ownership record's per-process value is `2488` MiB. The utilization columns are
instantaneous samples; a zero after-snapshot does not mean that timed kernels
did not run.

## Live arrow-listener exercise on PID 1401033

This is **pre-final-fix operational evidence**, not post-fix validation.

The live arrow listener was exercised on PID `1401033`. `Right` plus `Down`
changed the displayed motion family/range and recovered the displayed state
from `OUT OF TRAINED SUPPORT/CANONICAL` to `SUPPORTED/LEARNED`. Search response
remained interactive at about `4.35 ms`; the recovery screenshot reports an
exact last-search value of `4.345 ms`.

The recovery screenshot is
`/tmp/full-walking-gpu5-recover.png` (`51423` bytes, PNG `1366x851`, SHA-256
`bfd0b75d1b51460860052d435289ee6113ffd2253fb17e1d539cd563d39b5623`).
Its visible overlay records:

- the diagnostic, acceptance-ineligible title;
- family `0`, range `928`, row `206992`, and distance `0.243605`;
- `SUPPORTED`, height `0.000`, and `LEARNED`;
- backend `single-gpu-full-row-fp32:cuda:5`; and
- last search `4.345 ms`.

At report preparation, `/proc/1401033` still identified the process with start
ticks `53278727`, cwd
`/home/ubuntu/worktrees/motion-matching-full-walking-integration`, and this
command line:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m mm_sonic.full_walking_terrain_lmm_viewer view --corpus /home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 --model /home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/g1-full-walking-terrain-lmm/selection-full60-latent32-locomotion-v1 --scene ramp-10-up-down --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml --search-device cuda:5
```

The owning task report records its maximized MuJoCo window on DISPLAY `:1` and
a CUDA context only on physical GPU5. Again, PID `1401033` and its screenshot
predate the pending final fix.

## Reserved post-final-fix replacement validation

**Status: NOT RUN; NO POST-FIX CLAIM.** This section is intentionally reserved
for replacement evidence after the final fix commit exists.

- Final fix commit/hash and exact tree baseline: **pending**.
- Old-PID identity check and orderly replacement of PID `1401033`: **pending**.
- Replacement PID, start ticks, cwd, and full command line: **pending**.
- DISPLAY, visible/maximized window identity, and diagnostic overlay: **pending**.
- Real listener press/release validation for the fixed input paths, including
  family/range change, hard stop, mixed aliases, and reset edge behavior:
  **pending**.
- Replacement latency sample and state transition/recovery: **pending**.
- Replacement GPU5 ownership/isolation snapshot: **pending**.
- Replacement screenshot/log paths, byte sizes, and SHA-256 values: **pending**.

Nothing recorded for PID `1401033` satisfies these post-fix placeholders.

## Explicit limits

- This is a diagnostic benchmark, not formal acceptance evidence. The frozen
  test receipt remains red and the overlay remains diagnostic.
- Only physical GPU5 was exercised. No multi-GPU path was run, validated, or
  claimed.
- The benchmark contains 25 forced queries and three full CPU float64 parity
  samples. Those samples are exact, but they are not a proof of parity for all
  possible queries.
- FP32 full-row ranking can differ from complete CPU float64 ranking outside
  the bounded returned top-128. CPU float64 rescoring covers the returned set;
  the close-set overflow path retains complete CPU brute force.
- The GPU and utilization tables are two instantaneous snapshots, not a
  continuous utilization trace.
- Total startup includes strict corpus/model authentication and is not a pure
  GPU initialization measurement; matcher transfer/JIT startup is reported
  separately.
- The benchmark log and screenshots live under `/tmp` and are not themselves
  durable repository artifacts. This report preserves their exact recorded
  values, sizes, and hashes, but does not embed their bytes.
- The live listener exercise is pre-final-fix only. Fresh replacement
  PID/window/input evidence remains required after the pending fix lands.

## Post-final-fix operational closure

This section supersedes the earlier pending status and the placeholders in the
reserved section above; it does not erase or reinterpret the preserved
pre-final-fix evidence.
The closure remains **diagnostic operational evidence only**. It is not formal
acceptance evidence and makes no multi-GPU claim.

### Corrected code, review, and committed-tree verification

The exact corrected code baseline reviewed and exercised here is
`35767f203b89ee8dfe60338d078c73b2e7dc3e6f` (`fix runtime input and formal CPU
gates`). The evidence-report commit is documentation-only and follows that code
commit.

Two independent terminal reviews of that exact code HEAD returned **PASS with
no findings**. The owning lane's committed-tree verification recorded:

```text
122 passed in 20.50s
Ruff check: passed
Ruff format check: passed
py_compile: exit 0
git diff --check: exit 0
```

The corrected installed-`pynput` Space normalization, mixed physical
`Up`+`W` level-safe release behavior, and edge-triggered `R` behavior were
independently exercised by the reviews/tests. The same review confirmed that
GPU search remains diagnostic-only and that the generic and full formal-smoke
paths require the CPU exact-search backend.

### Identity-checked process replacement and window

The old, pre-fix PID `1401033` was identity-checked and stopped cleanly before
replacement. A direct post-replacement process scan found it absent and found
exactly one full-walking viewer: PID `1449809`.

The corrected viewer identity is:

| Field | Exact value |
| --- | --- |
| PID | `1449809` |
| Start time | `Mon Aug 10 19:42:42 2026 PDT` |
| Start ticks | `53563209` |
| Cwd | `/home/ubuntu/worktrees/motion-matching-full-walking-integration` |
| X11 window | `65011730` |
| `_NET_WM_PID` | `1449809` |
| `WM_CLASS` | `"MuJoCo", "MuJoCo"` |
| `WM_NAME` | `MuJoCo : g1_29dof_simplified` |
| Window state | maximized horizontally/vertically and focused |

The exact command line is:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python -u -m mm_sonic.full_walking_terrain_lmm_viewer view --corpus /home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/g1-full-walking-terrain-lmm/corpus-v1 --model /home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/g1-full-walking-terrain-lmm/selection-full60-latent32-locomotion-v1 --scene ramp-10-up-down --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml --search-device cuda:5
```

Report preparation did not signal, replace, or otherwise disturb PID
`1449809`; it remained the single live full-walking viewer.

### Post-fix physical-GPU5 ownership

The live compute-process inventory contains PID `1449809` only on physical
index 5, UUID `GPU-6a819546-e32a-076b-a85e-0e539d6fd9df`, with `2488` MiB
process memory. It contains no viewer context for PID `1449809` on GPUs 0--4
or 6--7. Other unrelated compute processes on the host are not attributed to
the viewer.

This proves the replacement viewer's single-GPU5 ownership for this run only.
No multi-GPU execution or support is claimed.

### Live real X11/`pynput` arrow-listener validation

With window `65011730` focused, the real X11/`pynput` listener received a held
`Up` arrow for `1.0 s`, followed by release and a `0.75 s` wait. The before and
after screenshots show a responsive family/range transition while retaining
the diagnostic backend and full searchable-row count.

Before input:

- screenshot: `/tmp/full-walking-gpu5-postfix-before.png`;
- size/type: `51545` bytes, PNG `1432x851`;
- SHA-256:
  `7b78b87ace439433980f12213d8be358a191d9e495c42fe9d530c5b1f817f5c2`;
- family `0`, range `112`, row `72076`, distance `0.249249`;
- `SUPPORTED`, height `0.000`, `LEARNED`; and
- last search `4.841 ms`.

After the held-and-released `Up` input:

- screenshot: `/tmp/full-walking-gpu5-postfix-after-up.png`;
- size/type: `67756` bytes, PNG `1432x851`;
- SHA-256:
  `ae2f0e48afbdf13cb5b1b88f481bcf148099d32f9905a963ad53e6175c66f6cc`;
- family `1`, range `2569`, row `1129981`, distance `0.223741`;
- `SUPPORTED`, height `0.000`, `LEARNED`; and
- last search `4.815 ms`.

Both screenshots visibly report full search rows `9741525/9741525`, backend
`single-gpu-full-row-fp32:cuda:5`, and the diagnostic acceptance-ineligible
title. The end-to-end listener input changed family `0`/range `112` to family
`1`/range `2569` without a multi-second search stall.

### Remaining post-fix input limits

The installed-`pynput` Space path, mixed `Up`+`W` release bookkeeping, and
`R` repeat behavior were independently checked in review/tests as stated
above. No gamepad was attached during the live operational exercise, so this
report does **not** claim a live physical-gamepad test of Space overriding an
active gamepad command. The end-to-end X11/window exercise above covers the
physical `Up` arrow sequence only; other fixed input paths retain their
automated/review evidence boundary.

This closes the requested post-fix replacement PID/window/arrow-input gate for
diagnostic operation. It does not change the red frozen test receipt, establish
formal motion acceptance, validate multi-GPU behavior, or provide sim-to-real
evidence.
