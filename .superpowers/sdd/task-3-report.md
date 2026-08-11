# Task 3 report: HybridMatcher single-GPU integration

## Outcome

Integrated the frozen single-GPU FP32 candidate scorer behind the existing
`HybridMatcher` without changing the default CPU cKDTree search. GPU mode skips
the CPU tree, scores the complete returned top-128 again with the existing CPU
float64 score, and uses the existing stable lowest-row tie break.

Base commit: `0a6a26a348ea2abb224d8bba75d49ea5ef894fb4`.

## TDD evidence

Five integration tests were added before production changes. The focused RED
command was:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_runtime.py -k single_gpu
```

Observed: `5 failed, 41 deselected in 2.55s`. Four failures were the expected
unsupported `search_device` keyword; the CPU-default test failed on the missing
`search_backend_identity`.

After the implementation, the same command produced:

```text
5 passed, 41 deselected in 2.42s
```

The RED/GREEN contracts prove:

- GPU construction does not allocate `_tree_values` or construct cKDTree.
- Only canonical `cuda:<nonnegative integer>` device strings are accepted.
- The backend receives the full float32 features, sorted searchable rows,
  per-row range IDs, contacts, transition penalty, and retry-budget exclusion
  capacity.
- Each query forwards only normalized explicit caller exclusions, plus the
  current row's exact range and two-bit contact code.
- Every returned top-128 row is rescored in CPU float64, independently of the
  device close count, and equal scores select the lowest global row.
- A close-candidate count larger than the returned set invokes the unchanged
  complete CPU brute-force authority with the normalized explicit exclusions.
- Default mode retains cKDTree/brute-force parity and never constructs the GPU
  backend.

## Implementation

- Added keyword-only `search_device` immediately after `max_search_rows`.
- Added strict canonical parsing that accepts `cuda:0`, `cuda:5`, and other
  canonical nonnegative indices while rejecting bare, signed, negative,
  leading-zero, non-CUDA, and non-string values.
- Exposed `search_backend_identity` as `cpu-ckdtree-exact` or
  `single-gpu-full-row-fp32:cuda:5`.
- Kept the GPU import lazy inside the selected construction branch.
- Added `_single_gpu_match`, which uses `_normalized_excluded_rows` rather than
  `_match_exclusions`; contact compatibility stays a hard device-side filter.
- Retained the existing `_candidate_score`, `_best`, `SearchResult`, and
  `brute_force_match` behavior as the CPU rescore and ambiguity authority.
- Added `last_search_elapsed_ms` and `warm_search_elapsed_ms`. Both are `None`
  before a search; GPU queries update the former, while the first post-compile
  query fixes the latter for Task 4 reporting. CPU mode leaves both `None`.

## Verification

Complete runtime plus frozen GPU scorer suites:

```text
61 passed in 10.32s
```

Static checks:

```text
ruff check: All checks passed!
ruff format --check: 2 files already formatted
py_compile: exit 0
git diff --check: exit 0
```

## Self-review

- GPU overflow fallback uses `close_candidate_count > len(rows)`; equality is
  not overflow because the complete close set still fits in the returned set.
- The overflow fallback passes only normalized explicit exclusions back to
  `brute_force_match`, which independently reapplies hard contact filtering.
- `candidate_count` comes from the exhaustive backend in normal GPU mode and
  from the complete brute-force result after overflow.
- CPU construction and the existing CPU `match` body remain in their original
  branch, including float64 scoring, stable ties, and capped-search behavior.
- A requested GPU failure propagates; there is no implicit CPU or capped-row
  fallback.
- No viewer, model, corpus, Task 1/2 report, or GPU scorer file was changed.

## Concerns

None for Task 3. Task 4 still owns real full-corpus latency measurement, viewer
wiring, overlay display, and the physical-GPU isolation evidence.

## Review-fix follow-up

Review base: `d2deaa24bb6e76cb1e29dc28b9dde6576f00c04e`.

The first review RED covered malformed GPU responses, a real 129-row response
from a 130-searchable-row corpus, and retry with 77 contact-incompatible rows.
The focused command selected those three tests and observed `3 failed, 46
deselected in 2.73s`: malformed rows reached CPU scoring, 129 rows were
accepted, and retry forwarded 78 exclusions. A later metadata RED added an
in-range but incorrect candidate count plus a negative finite device minimum;
it observed `1 failed, 48 deselected in 2.70s` because both reached scoring.

The review fix now:

- caches exact searchable counts for all four contact codes once during GPU
  construction;
- sends only sorted, unique, explicitly rejected searchable rows during GPU
  native-limit retry and determines contact exhaustion from the cached count;
- never calls `_match_exclusions` or `np.union1d` on that retry path (the
  regression test makes either call fail);
- validates the complete device response before CPU scoring or timing mutation:
  1-D exact integer rows, nonempty top-128 bound, strict sort/uniqueness,
  searchable/nonexcluded/contact-compatible membership, exact nonboolean
  counts, exact contact-filtered candidate count after exclusions, finite
  nonnegative minimum, and finite nonnegative elapsed time; and
- preserves close-set overflow fallback, stable float64 CPU rescoring, warm
  timing semantics, and the unchanged CPU path. Negative zero remains a valid
  device minimum.

Final focused GREEN: `8 passed, 41 deselected in 2.39s`.

Final complete runtime plus GPU scorer GREEN: `64 passed in 10.48s`.

Static verification after formatting: Ruff check passed, Ruff format check
reported both files formatted, `py_compile` exited zero, and `git diff --check`
exited zero.

## Invalid-successor review fix

A final review found that a sequential successor can legitimately cross from
the current contact code to another contact code before its learned decode is
mechanically rejected. The previous GPU retry treated that rejected successor
as a malformed GPU candidate even though it had never come from GPU search.

The new regression starts at contact 00, advances sequentially to contact 01,
forces that successor outside a native joint limit, and requires recovery from
the GPU search with an empty explicit-exclusion vector. Both
`_match_exclusions` and `np.union1d` are patched to fail. Its RED was `1 failed,
49 deselected in 2.59s`, at the obsolete all-rejections-compatible assertion.

GPU retry now normalizes mechanical rejections to searchable rows and then
keeps only current-contact-compatible rows for both the device exclusion vector
and cached-count exhaustion. The unfiltered mechanical rejection list still
owns retry-budget and rejection evidence.

Focused GREEN: `9 passed, 41 deselected in 2.40s`.

Complete runtime plus GPU scorer GREEN: `65 passed in 10.38s`.
