# Single-GPU exact motion search

## Goal

Make the 9,758,524-row full-walking diagnostic viewer respond interactively without removing, sampling, or approximating corpus rows. Search runs on exactly one explicitly selected NVIDIA GPU; all other GPUs remain unused by the viewer.

## User contract

- `--search-device cuda:5` selects physical GPU index 5.
- The backend uses only that one device. It must not use JAX `pmap`, multi-device sharding, or replicate arrays to any other GPU.
- All range-safe, eligible corpus rows remain searchable.
- Search remains exact under the existing float64 score contract.
- Contact compatibility, the frozen `0.1` range-transition penalty, candidate exclusions, and lowest-row deterministic tie-breaking remain unchanged.
- The current CPU cKDTree path remains the default when no search device is requested.
- A requested but unavailable GPU fails explicitly; it never silently caps rows or falls back to CPU.
- The viewer remains labeled diagnostic because the frozen test receipt is red; GPU acceleration does not change model acceptance.

## Architecture

Add a lazy optional JAX backend dedicated to exhaustive single-query search. It receives the runtime's sorted `searchable_rows`, float32 normalized features, row range IDs, and two-bit contact codes. It copies those immutable tables to one selected JAX `Device` and converts the feature table to float64 there.

Each query exhaustively evaluates the same direct squared-difference score in float64:

`sum((x - q)²)`

It then adds the transition penalty outside the current range, sets incompatible-contact and explicitly excluded rows to infinity, and obtains the minimum in sorted searchable-row order. It returns every candidate within a conservative float64 roundoff window of the device minimum; the runtime uses the existing CPU `_candidate_score` and stable row tie-break over that bounded set before commitment. Randomized and adversarial parity tests compare the final result to the existing complete CPU brute-force result.

JAX compilation and device transfer happen during matcher construction, before the MuJoCo window opens. Search calls synchronize before returning so measured latency reflects completed device work.

## Integration boundaries

`HybridMatcher` accepts an optional exact-search backend. CPU construction and formal evidence remain unchanged by default. The full-walking viewer adds `--search-device`; its matcher factory passes the option through. Runtime identity and overlay record `single-gpu-exact:cuda:5` versus `cpu-ckdtree-exact` so diagnostic evidence cannot conflate the backends.

The backend is isolated in a new module so importing the normal runtime does not import or initialize JAX. The module validates the requested device index and uses an explicit `jax.Device` for every `device_put` and compiled call.

## Failure handling

- Missing JAX, unavailable CUDA, invalid index, allocation failure, non-finite query, no compatible candidate, or CPU/GPU parity failure raises a specific error.
- Explicit candidate exclusions remain bounded by the existing runtime retry budget.
- No backend failure changes the corpus, model, thresholds, or selected-row authority.
- CPU mode continues to behave byte-for-byte as before when the option is absent.

## Verification

1. RED tests prove the runtime lacks the requested backend and viewer option.
2. Synthetic randomized tests compare GPU results against existing float64 brute force for contacts, transition penalties, exclusions, and exact ties.
3. Tests prove one explicit fake/real device receives every array and no multi-device API is called.
4. Existing CPU tree/brute parity and viewer suites remain green.
5. A real GPU5 benchmark loads the exact full corpus and reports warm search median and p95; target p95 is at most 100 ms.
6. `nvidia-smi` confirms only GPU5 gains viewer/search memory.
7. The real ramp viewer is restarted with `--search-device cuda:5`; arrow-command response is inspected while its diagnostic label remains visible.
