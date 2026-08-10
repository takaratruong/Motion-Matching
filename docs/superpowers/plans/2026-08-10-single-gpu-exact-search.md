# Single-GPU Exact Motion Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search all 9.74M range-safe full-walking rows on exactly physical GPU5 with p95 warm latency at most 100 ms, preserving the existing exact score and diagnostic evidence boundary.

**Architecture:** First remove a redundant CPU sort that currently costs 4.34 seconds per query. Then add a lazy JAX float64 exhaustive scorer isolated behind `HybridMatcher`; the full viewer pins JAX visibility to physical GPU5 before import, skips the CPU cKDTree, and records the backend identity. CPU search remains unchanged by default.

**Tech Stack:** Python 3.11, NumPy, SciPy cKDTree, JAX CUDA, MuJoCo, pytest, Ruff.

## Global Constraints

- Use exactly one GPU: physical CUDA index 5.
- Do not use `pmap`, multi-device sharding, replication, or computation on GPUs 0–4 or 6–7.
- Search every eligible range-safe row; do not cap, sample, or approximate the corpus.
- Preserve float64 squared-L2 scoring, hard contact compatibility, transition penalty `0.1`, exclusions, and lowest-row stable tie-breaking.
- CPU mode remains the default and must preserve existing behavior.
- A requested unavailable GPU fails explicitly; there is no silent CPU or capped-search fallback.
- The frozen model test remains diagnostic-red; acceleration must not change acceptance labels or thresholds.

---

### Task 1: Remove redundant CPU exclusion sorting

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py:684-704`
- Test: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Consumes: `_normalized_excluded_rows(excluded_rows) -> np.ndarray`.
- Produces: `_match_exclusions` with identical sorted/unique output and a fast empty-caller-exclusion path.

- [ ] **Step 1: Write the failing no-union regression**

```python
def test_contact_exclusions_do_not_resort_when_caller_exclusions_are_empty(self):
    matcher = self._matcher_with_contacts()
    with mock.patch("numpy.union1d", side_effect=AssertionError("redundant sort")):
        result = matcher.match(np.zeros(31, np.float64))
    self.assertEqual(result.row, matcher.brute_force_match(np.zeros(31)).row)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_runtime.py \
-k contact_exclusions_do_not_resort
```

Expected: fail from the patched `np.union1d`.

- [ ] **Step 3: Implement the fast path**

```python
incompatible = self.searchable_rows[~compatible]
if not len(excluded):
    return incompatible.astype(np.int64, copy=False)
return np.union1d(excluded, incompatible).astype(np.int64, copy=False)
```

- [ ] **Step 4: Run GREEN and commit**

Run the focused runtime file, Ruff, and `git diff --check`; commit:

```bash
git add sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py
git commit -m "perf: avoid redundant contact exclusion sorting"
```

---

### Task 2: Add a one-device exhaustive JAX scorer

**Files:**
- Create: `sonic/python/mm_sonic/hybrid_terrain_lmm_gpu_search.py`
- Create: `tests/python/test_hybrid_terrain_lmm_gpu_search.py`

**Interfaces:**
- Produces: `configure_single_gpu_visibility(spec: str) -> tuple[str, int]`.
- Produces: `SingleGpuExactSearch(features, searchable_rows, row_ranges, contacts, *, physical_device_index, transition_penalty, exclusion_budget=32)`.
- Produces: `match_candidates(query, *, current_range, active_contact_code, excluded_rows) -> GpuSearchCandidates`.
- `GpuSearchCandidates` contains sorted global candidate rows, full compatible candidate count, device minimum score, close-candidate count, and elapsed milliseconds.

- [ ] **Step 1: Write RED contract tests**

Tests must assert:

```python
assert configure_single_gpu_visibility("cuda:5") == ("cuda:5", 5)
```

and reject malformed devices, an already-conflicting `CUDA_VISIBLE_DEVICES`, missing GPU/JAX, non-finite queries, oversized exclusions, and empty compatible sets. A fake JAX seam must record that every `device_put` target is the single selected device and that no multi-device API is touched.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_gpu_search.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement visibility isolation before importing JAX**

```python
def configure_single_gpu_visibility(spec: str) -> tuple[str, int]:
    match = re.fullmatch(r"cuda:([0-9]+)", str(spec))
    if match is None:
        raise ValueError("search device must be cuda:<physical-index>")
    index = int(match.group(1))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, str(index)):
        raise RuntimeError("CUDA visibility conflicts with the requested search GPU")
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before single-GPU isolation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    return f"cuda:{index}", index
```

- [ ] **Step 4: Implement the exhaustive scorer**

Lazily import JAX only inside construction. Require exactly one visible GPU and use its single `Device` for every `device_put`. Store float64 searchable features, sorted global rows, range IDs, and contact codes on that device.

The jitted kernel computes:

```python
scores = jnp.sum(jnp.square(features - query[None, :]), axis=1)
scores += transition_penalty * (range_ids != current_range)
scores = jnp.where(contact_codes == active_contact_code, scores, jnp.inf)
padded = jnp.concatenate((scores, jnp.asarray((jnp.inf,), dtype=jnp.float64)))
safe_exclusions = jnp.where(excluded_positions >= 0, excluded_positions, len(scores))
padded = padded.at[safe_exclusions].set(jnp.inf)
scores = padded[:-1]
values, positions = jax.lax.top_k(-scores, min(128, len(scores)))
minimum = -values[0]
tolerance = jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.abs(minimum)) * 512
close_count = jnp.count_nonzero(scores <= minimum + tolerance)
```

Synchronize returned arrays before timing completes. Return the fixed top-candidate set; callers fail closed or use complete CPU brute force if `close_count` exceeds the returned set.

- [ ] **Step 5: Prove parity on GPU5 in a subprocess**

Use `CUDA_VISIBLE_DEVICES=5` and randomized synthetic corpora. Compare the CPU float64 brute-force winner to the GPU candidate set after CPU rescoring for every contact code, current-range penalty, explicit exclusion, exact duplicate tie, and seeded random query. Assert only one JAX GPU is visible.

- [ ] **Step 6: Run static checks and commit**

Commit:

```bash
git add sonic/python/mm_sonic/hybrid_terrain_lmm_gpu_search.py \
  tests/python/test_hybrid_terrain_lmm_gpu_search.py
git commit -m "feat: add single-gpu exact motion scorer"
```

---

### Task 3: Integrate exact GPU candidates into HybridMatcher

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`
- Modify: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Add keyword-only `search_device: str | None = None` to the existing `HybridMatcher.__init__` signature after `max_search_rows`.
- Expose `search_backend_identity` as `cpu-ckdtree-exact` or `single-gpu-exact:cuda:5`.
- When GPU is selected, skip `_tree_values` and cKDTree construction.

- [ ] **Step 1: Write RED integration tests**

Use an injected/fake backend factory to prove GPU mode skips cKDTree construction, receives only normalized explicit exclusions, uses the current row's contact code and range, CPU-rescores returned candidates with `_candidate_score`, preserves stable row ties, and invokes complete brute force if the close set overflows.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_runtime.py -k single_gpu
```

Expected: fail because `search_device` is unsupported.

- [ ] **Step 3: Wire construction without changing CPU mode**

```python
if search_device is None:
    self.search_backend_identity = "cpu-ckdtree-exact"
    self._tree_values = np.asarray(self.features[searchable], dtype=np.float64)
    self._tree = cKDTree(self._tree_values, compact_nodes=True, balanced_tree=True)
else:
    from .hybrid_terrain_lmm_gpu_search import SingleGpuExactSearch
    self.search_backend_identity = f"single-gpu-exact:{search_device}"
    self._gpu_search = SingleGpuExactSearch(
        self.features,
        self.searchable_rows,
        self.row_ranges,
        self.artifacts.contacts,
        physical_device_index=int(search_device.removeprefix("cuda:")),
        transition_penalty=self.transition_penalty,
        exclusion_budget=self.candidate_retry_budget,
    )
```

Factor GPU matching into `_single_gpu_match`; normalize only caller exclusions, request candidates, CPU-rescore them, and return the existing `SearchResult` type. Keep `brute_force_match` unchanged as the ambiguity fallback and audit authority.

- [ ] **Step 4: Run runtime GREEN and commit**

Run the complete runtime test file plus GPU scorer tests, Ruff, format, compile, and diff checks. Commit:

```bash
git add sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py
git commit -m "feat: use exact single-gpu motion search"
```

---

### Task 4: Expose GPU5 in the full viewer and benchmark the real corpus

**Files:**
- Modify: `sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py`
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`
- Modify: `tests/python/test_full_walking_terrain_lmm_viewer.py`
- Modify: `tests/python/test_hybrid_terrain_lmm_viewer.py`
- Create: `.superpowers/sdd/full-walking-single-gpu-search-report.md`

**Interfaces:**
- Add `view --search-device cuda:5`.
- Call `configure_single_gpu_visibility` immediately after argument parsing and before any JAX import.
- Pass `search_device` to `HybridMatcher`.
- Include `search_backend_identity` and last/warm search milliseconds in interactive identity and overlay.

- [ ] **Step 1: Write RED parser/wiring/label tests**

Assert `--search-device cuda:5` reaches matcher construction, conflicting visibility fails, CPU invocation remains unchanged, and the body displays `single-gpu-exact:cuda:5` without changing the diagnostic title.

- [ ] **Step 2: Run RED, implement minimal wiring, and run GREEN**

Run both full and hybrid viewer test files. Preserve smoke/formal CPU defaults.

- [ ] **Step 3: Run the real single-GPU benchmark**

With the current CPU viewer stopped, launch a bounded benchmark in the same environment with physical GPU5 isolated. Load the exact corpus/model, build the matcher with `search_device="cuda:5"`, warm it, then time at least 20 representative command/search queries. Record median, p95, maximum, selected rows, CPU brute-force parity samples, startup time, and before/after per-GPU memory/utilization. Require p95 at most 100 ms and no process memory growth on GPUs other than physical GPU5.

- [ ] **Step 4: Run full verification and commit**

Run runtime, GPU scorer, hybrid viewer, and full viewer suites; Ruff check/format, py_compile, and diff-check. Commit:

```bash
git add sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
  tests/python/test_full_walking_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  .superpowers/sdd/full-walking-single-gpu-search-report.md
git commit -m "feat: drive full walking viewer with GPU5 search"
```

- [ ] **Step 5: Relaunch the real diagnostic ramp viewer**

Run the existing full-corpus/model/ramp command with `--search-device cuda:5`. Confirm the MuJoCo window opens, the overlay remains diagnostic, arrows respond without a multi-second stall, and `nvidia-smi` shows search memory only on GPU5.
