# G1 Runtime and Evidence Hardening Result

Date: 2026-08-10 (America/Los_Angeles)

## Scope

This result covers the hybrid terrain LMM runtime, MuJoCo viewer/evidence gate,
and combined-corpus viewer wrapper. It does not claim that a strict combined
model has completed training, and it does not substitute a synthetic fixture
for the final full-corpus MuJoCo run.

## Review claims reproduced before changes

Focused tests and direct code inspection reproduced each runtime review issue:

- the CLI silently defaulted to a 500,000-row stratified search view;
- the runtime transition penalty defaulted to `0.0`;
- capped views and full views both displayed `EXACT SEARCH`;
- a generated hills scene could produce an accepted MuJoCo receipt;
- MuJoCo acceptance did not require at least 1,000 frames, terrain variance,
  tree/brute parity, manifest-row retrieval, or full-search scope;
- receipts were overwritten with a non-atomic `Path.write_text` call;
- interactive receipts omitted cache/model/scene/XML/search identity.

The new regression tests were observed failing for these missing behaviors
before the production changes were applied.

A later independent review exposed additional evidence-boundary defects. The
focused RED run reproduced all of them: the viewer rejected the strict v2
combined receipt, runtime-only smoke could self-declare acceptance, formal
smoke did not distinguish a selection model from an all-row refit, neither the
exact `0.1` penalty nor exact feature retrieval was gated, generated terrain
could still display `EXACT SEARCH`, Space did not override an active gamepad,
authority identity was captured only once, and the receipt inode was fsynced
before its final read-only mode was applied.

## Delivered contract

- Formal receipts use schema `hybrid-terrain-lmm-mujoco-smoke/v2` and bind
  acceptance contract
  `strict-combined-frozen-authorities-scripted-runtime/v2`. Earlier v1
  receipts are historical diagnostics and cannot satisfy the hardened gate.
- Full range-safe rows are the default search view. An effective `--search-rows`
  reduction is labeled `diagnostic-stratified-cap` and is acceptance-ineligible.
- Receipts bind total safe rows, searched rows, total/searched family counts,
  search scope, exact search-view SHA-256, and the `0.1` transition penalty.
- Search scoring is squared normalized feature distance plus a `0.1` range
  transition penalty. Exact cKDTree results retain a chunked float64 brute-force
  oracle for audits and ambiguous lower bounds.
- Formal MuJoCo acceptance requires at least 1,000 requested and completed
  forwards, the strict combined-corpus authority, full search, nonzero terrain
  variance, terrain/range diversity, exact manifest-row feature retrieval,
  tree/brute parity, the transition penalty exactly equal to `0.1`, learned
  decodes, a canonical-verified all-row refit with loader-verified selection
  provenance, and zero fallback, clamp, crash, nonfinite, or native-limit
  failures. A primary-only cache remains diagnostic.
- The runtime-only Python smoke is always labeled
  `runtime-only-diagnostic-not-acceptance`; only the MuJoCo-forward gate can
  produce formal accepted evidence.
- The accepted default scene is `ramp-10-up-down`. Scene acceptance follows the
  authenticated corpus source manifest to its scene-index descriptor, indexed
  scene descriptor, `scene.json`, and bound `terrain.bin`. The chain is rechecked
  before and after the run. Strict combined caches authenticate the exact
  `g1-hybrid-terrain-lmm-combined-receipt/v2-strict` receipt and combined
  manifest before resolving the exact
  `g1-hybrid-terrain-lmm-corpus/v2-strict` primary-cache authority. The exact
  authority inventory must also contain and authenticate the strict PFNN
  supplement manifest; an incomplete combined receipt is rejected.
- Generated hills, generated flat terrain, and arbitrary G1HF files remain
  available as finite-domain diagnostics but cannot produce accepted evidence.
- Runtime identity includes cache and model manifests, source manifest, scene
  index, indexed scene, terrain bytes, G1 XML, all referenced G1 mesh assets,
  recursively included MJCF files and their assets, search view, counts, scope,
  transition penalty, generator provenance, and all-row evaluation identity.
  The complete identity is captured before and after the run; missing, stale,
  or changed authority rejects acceptance.
- Receipt files use canonical JSON and an fsynced temporary inode followed by an
  exclusive atomic hard-link publication. Existing evidence is never replaced;
  `fchmod(0444)` precedes the final file fsync, and the parent directory is
  fsynced after publication and temporary-name removal.
- Interactive keyboard/evdev behavior is preserved. Interactive receipts now
  bind full identity and explicitly state
  `interactive-diagnostic-not-acceptance`. A level-triggered Space hard stop
  overrides a non-neutral active gamepad. Indexed scene authentication is
  rechecked while rendering and latches the overlay to diagnostic on failure.

## Rejected-ramp native-limit trace and remediation

The immutable rejected ramp receipt recorded one canonical clamp of
`0.0012159074667957848` rad. A read-only deterministic replay with the receipt's
exact cache, model, indexed scene, XML, full-search hash, and `0.1` transition
penalty located it at zero-based frame `547` (forward `548`):

- command `(speed=1.0, steering=-0.65)`;
- post-integration native root XY
  `(-0.2659250220082839, -4.6985855135025485)` and heading
  `-1.135185307179581` rad;
- periodic exact-search winner row `168`, range `0`, family `flat`, distance
  `2.0853677035560225`;
- learned `left_ankle_roll_joint` angle `-0.26366169147149027` versus native
  lower limit `-0.2618`, an excess of `0.00186169147149029` rad;
- the old canonical fallback for row `168` produced
  `-0.26301590746679576`, whose `0.0012159074667957848`-rad excess exactly
  matches the immutable receipt.

The runtime now treats only a learned `_NativeJointLimitError` as a candidate
feasibility rejection. It excludes that row from the same query's exact
squared-normalized-plus-transition-penalty ordering and commits the exact best
remaining learned candidate. Search and successor paths share the transaction;
rejected attempts do not increment learned, fallback, or clamp counters. The
retry budget is explicitly bounded at 32 rejected candidates. If it is
exhausted, the original candidate follows the pre-existing canonical fail-safe;
native tolerances are never relaxed and no clamp is relabeled as learned.

Runtime state, overlay text, diagnostic receipts, interactive receipts, and
formal MuJoCo receipts expose cumulative candidate rejection count, first
rejected row, and maximum rejections in one step. These diagnostics do not
reject formal evidence by themselves; formal acceptance still requires zero
fallbacks, zero clamps, zero native-limit violations, and every other existing
gate.

## Native-limit claim boundary

A read-only offline diagnostic later decoded all 3,973,057 stored row-aligned
combined-corpus states. It found 6,127 learned and 2,706 canonical rows outside
a native limit at the runtime tolerance. It was not published as a formal
receipt and does not cover arbitrary live-terrain substitutions. This is why
formal acceptance remains scoped to the scripted runtime rather than claiming
a full-dataset native-limit proof. Every formal receipt therefore states:

- `native_limit_audit_scope = scripted-runtime-mujoco-forward-only`;
- `full_dataset_native_limit_audit_performed = false`;
- the exact number of completed MuJoCo forwards and native-limit violations;
- a separate `all_row_canonical_evaluation_identity` binding the final model's
  evaluation artifact plus selection-model digests, while retaining that
  evaluation receipt's explicit `native_joint_limits_evaluated = false` status.

## Verification evidence

The final read-only child review returned `PASS` with no Critical or Important
findings after the strict-authority, generator-snapshot, nested-MJCF, and stale
overlay remediations. Its two non-blocking notes are operational: the full
3.97M-row float64 search matrix plus cKDTree needs substantial host memory, and
the compatibility `TypeError` retry in `_load_generator` can make an internal
loader error less direct to diagnose.

Focused verification command:

```bash
PYTHONPATH=sonic/python:. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_combined_viewer.py
```

Observed result after final authority/schema remediation: `64 passed`.

Static verification command:

```bash
ruff check \
  sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_combined_viewer.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_combined_viewer.py
```

Observed result: `All checks passed!`.

Compilation verification command:

```bash
PYTHONPATH=sonic/python:. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
  sonic/python/mm_sonic/hybrid_terrain_lmm_combined_viewer.py \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_combined_viewer.py
```

Observed result: exit status `0`.

Fresh full-search v2 receipts are accepted for the authenticated ramp and stair
scenes and are listed with exact paths and hashes in
`2026-08-10-g1-hybrid-terrain-lmm-result.md`. A capped or generated-scene smoke
cannot satisfy this contract and must not be promoted as final evidence.
