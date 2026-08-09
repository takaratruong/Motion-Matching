# Task 1 report: 60 Hz G1 LMM flat data bundle

## Outcome

Implemented the receipt-bound, flat-only `g1-lmm-flat-data/v1` vertical slice.
It publishes exactly three immutable files: `database.bin`, `features.bin`, and
`manifest.json`. The real released-PFNN input produces one accepted 4,086-frame
range at 60 Hz with the exact source map `0, 2, ..., 8170` and zero alpha.

No GRAIL, slope, stair, scene, or runtime subsystem was added to this bundle.
The pre-existing `g1-terrain-artifacts/v2` path remains supported and its
integration tests pass.

## TDD evidence

Initial RED command:

```text
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_resample tests.python.test_database_builder tests.python.test_build_cli tests.python.test_artifacts
```

Expected failures were observed for missing `resample_map`, missing
`resources.g1_terrain_builder.features`, missing feature binary codecs, and
unrecognized `--output-fps`, `--flat-only`, `--retarget-npz`, and
`--retarget-receipt` arguments.

During scoped self-review, a second RED test exposed incorrectly combined
left/right foot normalization groups:

```text
AssertionError: np.float32(0.015587269) == np.float32(0.015587269)
```

The exporter was corrected to match the C++ group boundaries: each foot
position and each foot velocity is normalized as its own 3D group. The focused
test then passed.

Fresh final GREEN command (same mandated command as RED):

```text
Ran 84 tests in 117.935s
OK
```

Additional compatibility command:

```text
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v tests.python.test_kinematics tests.python.test_schema
```

Result: `Ran 11 tests ... OK`.

`git diff --check` and Python byte-compilation of all changed builder modules
also passed.

## Implemented contracts

- `resample_map` returns int32 left/right indices and float32 alpha, including
  equal-index/zero-alpha exact samples.
- The real Takara source has 34,863 frames at 50 Hz and maps to exactly 41,835
  rows at 60 Hz.
- The released-PFNN loader verifies the accepted receipt schema/status,
  receipt-declared NPZ SHA-256, 120 Hz rate, 8,171 frames, joint-name receipt,
  quaternion order, array shapes, and engine identity before creating qpos.
- Simulation-root filters are rate-derived and bind 31/61 frames at 60 Hz;
  contacts bind a 7-frame median; forward terrain sampling binds 121 rows.
- The feature exporter reproduces the C++ ordering: ankle positions, ankle
  velocities, hip velocity, 20/40/60 root positions, 20/40/60 facings, then
  four authenticated zero flat-terrain deltas.
- `features.bin` uses the Orange Duck little-endian layout: normalized float32
  matrix, float32 offset vector, and strictly positive finite float32 scale
  vector.
- Publication is staged, validated before and under the parent lock, fsynced,
  and atomically replaced. The validator rejects stale files, bad schemas,
  rate/horizon/filter/skeleton/range/source-map changes, binary corruption,
  nonzero flat terrain rows, and artifact size/hash mismatches.

## Real bundle evidence

Build command:

```text
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python resources/build_g1_terrain_database.py --output-fps 60 --flat-only --retarget-npz sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.npz --retarget-receipt sonic/runs/native-g1-pfnn/sample-retarget/LocomotionFlat01_000-120hz.receipt.json --output sonic/runs/g1-lmm-flat-60hz/data
```

Result:

```text
VALID g1-lmm-flat-data/v1 frames=4086 clips=1 bones=31 features=31 scenes=0 source_rows=0
BUILT g1-lmm-flat-data/v1 frames=4086 clips=1 scenes=0 output=sonic/runs/g1-lmm-flat-60hz/data
```

Artifact evidence:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `database.bin` | 6,594,988 | `0803c212e7692a98430a1fee56cf70d814834cffd1cb747cc61c2699cab0906b` |
| `features.bin` | 506,928 | `767caf71bdb667a6f496ffe8f805ef4e82f64e3b9d1f9fd88188a5bef68141c5` |
| `manifest.json` | 172,506 | `f1b757a1aedc86cacb88eddc35a014176dc0225050bd721eb8ec10b839386bc1` |

Direct reload confirmed database shape `[4086,31,3]`, contacts `[4086,2]`,
features `[4086,31]`, one range `[0,4086)`, 31 skeleton names, the canonical
skeleton signature `6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7`,
finite data, positive scales, exact zero feature values at indices 27-30, and
manifest artifact digests/sizes equal to the published files.

## Files changed

- `resources/g1_terrain_builder/resample.py`
- `resources/g1_terrain_builder/schema.py`
- `resources/g1_terrain_builder/kinematics.py`
- `resources/g1_terrain_builder/sources.py`
- `resources/g1_terrain_builder/features.py`
- `resources/g1_terrain_builder/artifacts.py`
- `resources/build_g1_terrain_database.py`
- `resources/validate_g1_terrain_database.py`
- `tests/python/test_resample.py`
- `tests/python/test_database_builder.py`
- `tests/python/test_build_cli.py`
- `tests/python/test_artifacts.py`

## Concerns

An independent reviewer dispatch was attempted but the four-agent thread limit
was full. The parent agent will run the independent Task 1 review after this
commit. No known implementation or bundle blocker remains.

---

## Review correction: continuity-safe flat bundle v2

Review rejected the original v1 publication. The correction supersedes the
v1 result above and publishes only `g1-lmm-flat-data/v2`.

### Correction RED -> GREEN evidence

The correction began with new failing tests for the native/local continuity
union, the exact real v2 receipt, explicit v1 rejection, and publication-time
source/receipt/validation re-authentication. RED failures included the missing
`split_continuity_ranges` and `_authenticate_flat_manifest_inputs` interfaces,
the real candidate still reporting v1/4,086 rows, and the v1 validator path not
emitting the required v2 rejection.

The first independently sliced implementation exposed a second real RED:

```text
maximum_admitted_local_rotation_step_rad:
0.26562131888230156 != 0.23719475193867512
```

This was a Savitzky-Golay `interp` boundary overshoot in the Hips local
rotation. The final fragment path uses endpoint-clamped `nearest` root filters,
so each fragment is filtered without samples from across a rejected edge and
the independently recomputed all-31-bone maximum is admitted.

Fresh full scoped GREEN:

```text
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_resample tests.python.test_database_builder \
  tests.python.test_build_cli tests.python.test_artifacts

Ran 88 tests in 126.750s
OK
```

Python byte-compilation and `git diff --check` also passed. The Task 2 runtime
consumer independently reported its strict v2 parser, both artifact/digest
checks, continuity checks, 600-frame ordinary replay, and v1 plus four v2
tamper rejections all green.

### Corrected contracts

- The 60 Hz continuity gate is the union of native 29-DoF absolute steps and
  all-31-bone local quaternion geodesic steps strictly greater than 0.25 rad.
- Counts independently recompute to 31 native rejections, 32 local rejections,
  and 32 union rejections. Twenty short fragments totaling 233 frames are
  dropped; 13 ranges totaling 3,853 frames are published. The longest range is
  747 frames.
- Every retained source fragment is independently converted, root-filtered,
  differentiated, contact-filtered, and trajectory-clamped. No temporal
  preprocessing crosses a rejected edge.
- The continuity receipt is exactly `g1-lmm-continuity/v1`, including the
  little-endian `[N,4]` range digest and concatenated left/right/alpha source-map
  digest.
- Publication and validation re-hash and reload the actual absolute NPZ and
  receipt paths. The accepted receipt schema/status, hashes, exact source/map
  fields, exact complete validation receipt, and FK maximum `<=1e-5 m` are
  independently enforced.
- The validator explicitly rejects v1, reconstructs each fragment from the
  authenticated source, compares positions, rotations, derivatives, contacts,
  and features, and recomputes the continuity plan, all-bone in-range maximum,
  counts, and digests.

### Corrected real bundle evidence

The canonical build command is unchanged except that it now publishes v2. Its
atomic candidate validator and a fresh standalone validator both reported:

```text
VALID g1-lmm-flat-data/v2 frames=3853 clips=1 bones=31 features=31 scenes=0 source_rows=8171
BUILT g1-lmm-flat-data/v2 frames=3853 clips=1 scenes=0 output=sonic/runs/g1-lmm-flat-60hz/data
```

Corrected artifact evidence:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `database.bin` | 6,219,022 | `b8c7a8b3c831974af2f986414b549986242a5518edbf40e0ea10f795cca3834c` |
| `features.bin` | 478,036 | `64f145efadee0a08a93a58865a2771c384c092353df9317f0b494a2f8bfef26e` |
| `manifest.json` | 166,719 | `05c8ac4c04771fb7f8380d75d64da812f7a299e5994734f15faccd107a105f1f` |

Continuity evidence:

```text
maximum_admitted_native_step_rad = 0.2371947467327118
maximum_admitted_local_rotation_step_rad = 0.23719477124427774
range_digest_sha256 = dc8ecdb4052fea1e70321cc4100c70df25f43cc8b01ee8dd2e406b9d47732e52
source_map_digest_sha256 = f114924ab2f2d7bbea1f2ec4e051dda0bb7388ce67c523ae9ecd4f181cf73c5b
fk_max_error_m = 3.141193603136488e-07
```

No GRAIL, slopes, stairs, scene sidecars, or other subsystem was added. There
are no known Task 1 correction concerns.

Correction commit: `ae7faf64c4a147e0fa81fea66eeee10f623b51e1`.

---

## Review correction: canonical G1 skeleton authentication

An independent cross-review found that the flat builder required only 31
bones, while the flat validator authenticated a caller-selected XML, manifest,
and database only against one another. A reordered or reparented 31-bone XML
could therefore publish a self-consistent noncanonical bundle.

The correction moves the immutable canonical G1 names, parent topology, and
signature into the shared skeleton schema. The builder now requires that exact
contract before continuity processing and for every retained fragment. The
flat validator independently requires the exact manifest names, parents, and
`6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7`
signature, exact database parents, and the same exact skeleton from the
full-source and per-fragment XML reconstructions.

### RED -> GREEN evidence

The two focused negatives use both a name reorder and a valid-looking changed
parent topology. Before production edits, the builder continued into
resampling for both cases and the validator had no canonical receipt gate:

```text
Ran 2 tests in 0.004s
FAILED (failures=2, errors=2)
```

After the shared exact contract and both call-site gates were implemented:

```text
Ran 2 tests in 0.002s
OK
```

Fresh full Task 1 verification:

```text
PYTHONPATH=. /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest -v \
  tests.python.test_resample tests.python.test_database_builder \
  tests.python.test_build_cli tests.python.test_artifacts

Ran 90 tests in 126.490s
OK
```

The 11 schema/kinematics compatibility tests, Python byte-compilation, and
scoped `git diff --check` also passed.

### Real bundle identity

The unchanged canonical build command ran its atomic candidate validator, and
a separate validator invocation also reported:

```text
VALID g1-lmm-flat-data/v2 frames=3853 clips=1 bones=31 features=31 scenes=0 source_rows=8171
BUILT g1-lmm-flat-data/v2 frames=3853 clips=1 scenes=0 output=sonic/runs/g1-lmm-flat-60hz/data
```

The canonical input already used the required skeleton, so all three published
files remain byte-identical:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `database.bin` | 6,219,022 | `b8c7a8b3c831974af2f986414b549986242a5518edbf40e0ea10f795cca3834c` |
| `features.bin` | 478,036 | `64f145efadee0a08a93a58865a2771c384c092353df9317f0b494a2f8bfef26e` |
| `manifest.json` | 166,719 | `05c8ac4c04771fb7f8380d75d64da812f7a299e5994734f15faccd107a105f1f` |

No bundle, subsystem, or terrain scope changed, and no known concern remains.

---

## Stale-retarget root-cause correction: walk-only flat bundle v3

The v2 bundle was built from a stale retarget. Its legacy 14-field receipt
omitted the PFNN scale, source interval, warmup, and grounding mode; used an
obsolete prepared BVH and toe aliases; and recorded a `-0.4619996 m` grounding
offset. The generic loader accepted that document and relabeled every selected
interval from frame zero. The stale output produced effectively unusable
bilateral labels (`[4, 0]`) under the prior terrain-contact path.

The correction requires the exact 18-field current retarget receipt before
kinematics, preserves the receipt's absolute `start_frame`, and independently
re-authenticates the source, receipt, interval, continuity, contacts, skeleton,
and binaries. The canonical canary is the independently approved pure-walk
interval `[7659,8171)` with 120 warmup frames and flat grounding. The rejected
full and transition-containing retargets are not used.

The v3 contact channel now reproduces the bundled Orange Duck database rule,
separately from the generic terrain contact implementation: global LeftToe and
RightToe speed strictly below `0.15 m/s`, followed independently by
`scipy.ndimage.median_filter(size=6, mode="nearest")`. It has no height gate.
The generic terrain contact algorithm remains unchanged.

### RED -> GREEN evidence

Focused REDs proved that the stale receipt reached kinematics, `start_frame`
was discarded, contract-field mutations were accepted, v2 remained canonical,
and no bilateral contact gate or Orange Duck contact implementation existed.
The focused final command covered receipt/source authentication, nonzero source
interval publication, stale rejection before kinematics, canonical skeleton,
v1/v2 rejection, independent continuity/contact recomputation, the real v3
assembly, and the transactional v3 manifest path:

```text
Ran 18 tests in 0.710s
OK
```

Both dedicated Orange Duck contact tests pass, and the pre-existing generic
terrain contact tests also passed unchanged. Python byte-compilation and
`git diff --check` passed.

### Canonical input and output

```text
source interval = [7659, 8171) at 120 Hz
source rows = 512
source SHA-256 = bbdeb79760950480582ae937e54b913c376caa49f344896a8958476b82f3317f
receipt SHA-256 = 2d0e93f485bab9c54773c66cf07c14d25837f5f4e50f5c007f9f8e2c5c20520f
original BVH SHA-256 = 4a01768df71c6f7b5bbb71312c1e94eae489af32b21c3e300fd1c8d1ffc24bc5
prepared BVH SHA-256 = d6dbbac84e68d419d27aff0356b5a8245522f39694d94a6fc83e300ab6feaf8d
PFNN scale = 5.6444
grounding = flat
grounding offset = 0.06142798715901732 m
```

The transactional candidate validator and a separate direct invocation both
reported:

```text
VALID g1-lmm-flat-data/v3 frames=256 clips=1 bones=31 features=31 scenes=0 source_rows=512
BUILT g1-lmm-flat-data/v3 frames=256 clips=1 scenes=0 output=sonic/runs/g1-lmm-flat-60hz/data-v3
```

The published source map spans absolute source frames 7659 through 8169. The
single `[0,256)` range has zero native, local, or union continuity rejections,
zero dropped fragments/frames, native maximum `0.12608182430267334 rad`, and
local maximum `0.12608181972804253 rad`. FK error is
`1.711800573789496e-7 m`.

Orange Duck contact evidence is 116 left / 117 right frames, 3 / 4 runs,
longest runs 49 / 45 frames, and 6 alternating run-order transitions. Pattern
counts are 28 airborne, 223 single-support, and 5 double-support frames.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `database.bin` | 413,368 | `13b368759c22ff3427d937a86fd9399cd6e80646e5f288e80da01bd988daaaad` |
| `features.bin` | 32,008 | `7c35809e1dd5ea14bd56b0f607cb9da22b3464ebbece01a895050372530b40df` |
| `manifest.json` | 17,857 | `db01f5bfb7641333b3e40ea4a5d1eb655131c7bc72681d2738fde2dbf9298709` |

A second transactional build in a fresh temporary directory was byte-identical
for all three files. The historical v2 directory was not overwritten.

## Review correction: canonical G1 XML trust and v3-only publication

The v3 database is now bound to the exact XML bytes used for G1 kinematics.
The manifest carries the portable descriptor below; it intentionally contains
no host path and accepts a byte-identical copy at any caller-selected location:

```json
{
  "kinematics_model": {
    "asset": "g1_29dof.xml",
    "sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
    "size_bytes": 26914
  }
}
```

The builder reads, bounds, and authenticates the selected XML bytes before
constructing kinematics. Those already-authenticated bytes are passed to
`G1Kinematics.from_xml_bytes`; the MuJoCo parser never reopens the XML path.
The independent validator follows the same byte-consuming seam while retaining
its own descriptor/hash constants. A mutation-after-authentication test swaps
the selected path immediately as parsing begins and proves both paths still
consume the authenticated bytes. Mesh payloads are supplied separately through
MuJoCo's asset map and are outside this XML receipt.

The alternate canonical-topology XML with SHA-256 `456eb9a0...` is therefore
rejected before kinematics. The automated negative uses a portable temporary
whitespace-only XML mutation rather than depending on another host worktree.
Missing descriptors and altered asset, hash, integer size, or size type are
also rejected.

Direct flat publication and its repeated authentication gates now accept only
`g1-lmm-flat-data/v3`. A v3 candidate relabeled as v2 is rejected before even
a permissive candidate callback runs. The generic
`g1-terrain-artifacts/v2` GRAIL/non-flat publisher remains unchanged.

The focused REDs first showed the alternate XML completing the builder, all
descriptor mutations passing flat publication authentication, the independent
validator reaching payload loading without a descriptor, and the relabeled v2
candidate being published. A second RED proved both producer and validator
hashed a pathname and later reopened it. After the correction, the focused
trust and mutation gates pass. The complete Task1 modified-area module run
before the final byte-consuming hardening reported:

```text
Ran 98 tests in 119.401s
OK
```

The final byte-consuming implementation also passes all five kinematics tests.
The transactional canonical build, its candidate validator, and a subsequent
standalone validator each reported the same accepted 256-frame, 512-source-row
v3 bundle. A second build to a fresh directory compared byte-identically for
all three files. Python byte-compilation and `git diff --check` also passed.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `database.bin` | 413,368 | `13b368759c22ff3427d937a86fd9399cd6e80646e5f288e80da01bd988daaaad` |
| `features.bin` | 32,008 | `7c35809e1dd5ea14bd56b0f607cb9da22b3464ebbece01a895050372530b40df` |
| `manifest.json` | 18,021 | `5b5c48ccbb1dbabdbf87842d8b033c15b307199d72a8d90e4e39208ba5382db1` |

### Clean-checkout test portability correction

The XML trust unit tests no longer read the host-default XML, external mesh
directory, ignored retarget files, or ignored published v3 tree. Builder tests
use a temporary minimal XML, a mocked authenticated source, a temporary
descriptor derived from those exact bytes, and an in-memory asset map. Validator
tests exercise a narrow `_load_flat_kinematics` seam with the same fully
temporary inputs before any motion payload is read. Real canonical integration
validation remains separate from these unit fixtures.

As a RED, running the five original trust tests from `/tmp` produced five
errors: two missing the ignored retarget NPZ and three missing the ignored v3
manifest. The corrected five-test command passes from `/tmp` in `0.004s` with
only the repository on `PYTHONPATH`. The complete `test_build_cli` module then
reported:

```text
Ran 31 tests in 113.951s
OK
```

The unchanged canonical bundle independently validates with the same hashes
listed above. Python byte-compilation and `git diff --check` pass.
