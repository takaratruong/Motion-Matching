# Task 4 report — coupled walk/pickup training

Status: DONE

## Execution

- Base: `a34ca758e22ee4fcb5c6c84ad3660b91a7ca3fe8`
- GPU: one `NVIDIA L40S`, launched with `CUDA_VISIBLE_DEVICES=1` (PyTorch device `cuda:0`).
- Real dataset rebuild: 13 seconds.
- Real training: 17 seconds per deterministic run; the final 17-second run published the explicit `last.pt` and `best.pt` aliases after their contract was added. The earlier real run also took 17 seconds.
- Defaults used: A/B/C = 1/1/1 epochs, batch size 64, frozen validation subset 4.

## Artifacts

All artifacts are intentionally untracked under `build/g1-overlap/`.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `dataset.npz` | 129100866 | `8f88f85b17c904bbf538682e54fe037c8a76ec5e7ee45c1dbcd26455c7142754` |
| `checkpoint.pt` | 52549300 | `ca1d113b15fe3688039cc42e94b59eaefba656d5ef55e51d0e74c9875198c60a` |
| `last.pt` | 52549300 | `ca1d113b15fe3688039cc42e94b59eaefba656d5ef55e51d0e74c9875198c60a` |
| `best.pt` | 52549300 | `ca1d113b15fe3688039cc42e94b59eaefba656d5ef55e51d0e74c9875198c60a` |
| `checkpoint.training.jsonl` | 2870 | `6f7c1457a4dc0392cd1384d9358c2558ca943fa6a231b203b48889b05ee6c116` |
| `checkpoint.validation.json` | 1029 | `c13fd48bc7e7ad863a326f8b81f1fd1e1a90e4e0d13837fea69966b3e2fdb39f` |

Dataset audit: 2,020 interaction rows, byte-identical 20-frame overlaps, train-only normalization, dynamic-Hips variance nonzero, canonical signature `6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7`.

## Validation quality and sampler gate

Frozen validation rows: 4.

| DDIM steps | attach proxy @8 | median grasp position | median grasp orientation | no-stop proxy |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.0 | 13.846933841705322 m | 138.6712875366211° | 1.0 |
| 50 | 0.0 | 26.12924289703369 m | 138.37097930908203° | 1.0 |

Selected production sampler: 20 steps. This is the honest deterministic gate result: both percentage proxies match, 20-step median position is lower, and its orientation is within 2°. No validation rows or metrics were altered.

## Verification

`tests.python.test_overlap_checkpoint`, `tests.python.test_overlap_diffusion`, and `tests.python.test_overlap_dataset`: 37 tests passed in 3.143 seconds. The final checkpoint strictly loaded on CPU and CUDA with the 195/25/9, 256-wide/8-block/8-head model schema.

## Deviations and environment notes

- The prescribed funnel interpreter has CUDA PyTorch but no MuJoCo. I ran its exact Python binary with `PYTHONPATH=/home/ubuntu/miniconda3/lib/python3.10/site-packages`, which supplies the already-installed MuJoCo package; no dependencies were modified.
- Native local offsets vary by at most 1.18 mm after existing resampling/FK conversion. Canonical offsets therefore use the native median and reject variation over 2 mm; this is consistent with the repository's existing 1 mm conversion FK contract while still rejecting animated child translations.
