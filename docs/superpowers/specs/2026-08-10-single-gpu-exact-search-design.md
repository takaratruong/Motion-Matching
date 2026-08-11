# Single-GPU full-row FP32 diagnostic search

## Goal

Make the 9,758,524-row full-walking diagnostic viewer respond interactively without removing or sampling corpus rows. Search runs in float32 on exactly one explicitly selected NVIDIA GPU; all other GPUs remain unused by the viewer.

## User contract

- `--search-device cuda:5` selects physical GPU index 5.
- The backend uses only that one device. It must not use JAX `pmap`, multi-device sharding, or replicate arrays to any other GPU.
- All range-safe, eligible corpus rows remain searchable.
- Every eligible row is scored in float32. This is diagnostic retrieval and is not the formal float64 CPU search contract.
- Contact compatibility, the frozen `0.1` range-transition penalty, candidate exclusions, and lowest-row deterministic tie-breaking remain unchanged.
- A bounded top candidate set is rescored in float64 on CPU before pose commitment.
- The current formal float64 CPU cKDTree path remains the default when no search device is requested.
- A requested but unavailable GPU fails explicitly; it never silently caps rows or falls back to CPU.
- The viewer remains labeled diagnostic because the frozen test receipt is red; GPU acceleration does not change model acceptance.

## Architecture

Add a lazy optional JAX backend dedicated to exhaustive single-query search. It receives the runtime's sorted `searchable_rows`, float32 normalized features, row range IDs, and two-bit contact codes. It copies those immutable tables to one selected JAX `Device` without widening them.

Each query exhaustively evaluates the direct squared-difference score with float32 subtraction, squaring, and accumulation:

`sum((x - q)²)`

It then adds the transition penalty outside the current range, sets incompatible-contact and explicitly excluded rows to infinity, and returns a bounded top candidate set plus the stable lowest global row inside the float32 near-minimum window. The runtime uses the existing float64 CPU `_candidate_score` and stable row tie-break over the returned candidates before commitment. This is deliberately diagnostic rather than a claim of complete CPU-float64 winner parity.

JAX compilation and device transfer happen during matcher construction, before the MuJoCo window opens. The jitted kernel receives the resident feature/range/contact arrays as explicit arguments instead of capturing them as multi-gigabyte compiled constants. Search calls synchronize before returning so measured latency reflects completed device work.

## Integration boundaries

`HybridMatcher` accepts an optional GPU-search backend. CPU construction and formal evidence remain unchanged by default. The full-walking viewer adds `--search-device`; its matcher factory passes the option through. Runtime identity and overlay record `single-gpu-full-row-fp32:cuda:5` versus `cpu-ckdtree-exact` so diagnostic evidence cannot conflate the backends.

The backend is isolated in a new module so importing the normal runtime does not import or initialize JAX. The module validates the requested device index and uses an explicit `jax.Device` for every `device_put` and compiled call.

## Failure handling

- Missing JAX, unavailable CUDA, invalid index, allocation failure, non-finite query, no compatible candidate, or CPU/GPU parity failure raises a specific error.
- Explicit candidate exclusions remain bounded by the existing runtime retry budget.
- No backend failure changes the corpus, model, thresholds, or selected-row authority.
- CPU mode continues to behave byte-for-byte as before when the option is absent.

## Verification

1. RED tests prove the runtime lacks the requested backend and viewer option.
2. Synthetic tests prove full-row FP32 scoring, contact masks, transition penalties, exclusions, stable ties, and float64 CPU rescoring of returned candidates.
3. Tests prove one explicit fake/real device receives every array and no multi-device API is called.
4. Existing CPU tree/brute parity and viewer suites remain green.
5. A real GPU5 benchmark loads the full corpus and reports warm search median and p95; target p95 is at most 100 ms.
6. `nvidia-smi` confirms only GPU5 gains viewer/search memory.
7. The real ramp viewer is restarted with `--search-device cuda:5`; arrow-command response is inspected while its diagnostic label remains visible.
