# Smart Pickup authored-data record

Date: 2026-07-18

Status: Task 5 certification and baked-scene record

Baked scene commit: `f330bde8412da6b85828edc04bde65aadd23b883`

Baked scene tree: `b209905b55fab46b2dc1318a9c55d7cde792f598`

## Authority boundary

This record identifies the data used to author the three demo slots and the
different motion provenance selected by unrestricted runtime preview. Hexadecimal
IEEE-754 binary32 values are the exact scalar authority. Task 4 report decimals
are copied as serialized, frozen scene decimals mirror their C++ literals, and
derived decimal values are C++ `max_digits10` displays of the resulting
binary32 state.

Source provenance is authoring evidence only. Runtime slot IDs do not restrict the global motion matcher and are not PickRequest authority.

The runtime scene contains only ordered slot IDs and object-local scalar values.
It contains no source clip ordinal, global frame, candidate subset, or matcher
allowlist. The preview authority record for every candidate was
`mode="global_unrestricted"`, `candidate_subset_authority=false`, and
`runtime_clip_allowlist_applied=false`.

## Full-pack identity

The Task 0 pack is
`build/smart-pickup/full-pack`, built from dataset
`nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL` with schema `1`, seed
`20260714`, and no diagnostic limit.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `interaction_database.bin` | 914,832,959 | `4d3b65f73e9a207988aaaebded36b988f811ec068988c7e829732701d9d2da1b` |
| `interaction_features.bin` | 145,195,636 | `3b492ca7e5ed12aade5750ff925c689f4acf56f28edc31a4a5f341e434adf145` |
| `manifest.json` | 366,324 | `c9024d12b05c59a492f9f06a6fe2614cb1e814ece6e35e9e96e5be7210d800e4` |
| `evaluation_split.json` | 10,971 | `2edd979e1396550d81523eee6ce999eddf5400e9e3efc90c194064939d39c20d` |
| `validation_report.json` | 214,025 | `23f19607cee5c9d76cbc99f3725ccbe05c16dda58bed8ecd6aeae0dc01e89023` |

The manifest/report counts are:

| Field | Exact value |
|---|---:|
| Target rate | 25 Hz |
| Skeleton bones | 31 |
| Feature scalars | 71 |
| Source clips | 2,991 |
| `included_clips` builder/report field | 2,108 |
| Rejected clips | 883 |
| Runtime manifest clips | 2,045 |
| Runtime frames | 511,250 |
| Database object IDs | 633 |
| Held-out object IDs | 20 |
| Diagnostic limit | `null` |

The 883 rejected clips comprise 138 `ambiguous_active_hand`, 365
`contact_lost_before_hold`, 2 `fk_rotation_error`, 47
`joint_limit_violation`, and 331 `no_distinct_lift_phase` rejections.

## Task 4 authoring report

The exact report is
`build/smart-pickup/beer10-slot-candidates.json`: 19,468 bytes, SHA-256
`6e93f87e455513c887f4deeb2900980f0c13ca8f1f560ffa89c3ce1de715f7e1`.
Static gates accepted 101 candidates. Greedy `0.10 m AND 10 degrees`
deduplication removed 82 and retained 19. The static gate rejections were:

| Gate | Rejected |
|---|---:|
| `contact_position` | 1,265 |
| `contact_orientation` | 510 |
| `root_table` | 167 |
| `hand_table` | 0 |
| `hand_object` | 2 |
| `hand_mismatch` | 0 |

These values describe offline authoring compatibility and static path checks;
they do not certify a runtime match. The unrestricted preview below supplies
that separate evidence.

## Target source provenance and events

The target source is manifest clip ordinal `278`, range `[69500,69750)`, with
right hand encoded as `1`. Its stable authoring key, derived by the Task 4 key
rule, is:

```text
("nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL", 1,
 "pickup_table__beer_10__001", 101, 1)
```

| Event | Local frame |
|---|---:|
| First Reach / entry | 101 |
| Object alignment (Contact minus one) | 125 |
| First Contact | 126 |
| First Lift | 131 |
| First Hold | 140 |
| Stop, exclusive | 250 |

The source object/table data at authoring alignment are:

- table position `(0, 0.3607575, 0.0871601105)` and rotation
  `(w,x,y,z)=(1,0,0,0)`;
- table size `(2, 0.0399999991, 0.600000024)`;
- object position `(0.00394439697, 0.503655553, -0.142831802)` and rotation
  `(w,x,y,z)=(-0.0669774637, 0.67008847, 0.0799996108, -0.734911919)`;
- object dimensions `(0.0645366386, 0.0645366609, 0.240097240)`;
- right-hand grasp position in object
  `(0.0930671170, -0.119263843, 0.0375832170)` and rotation
  `(w,x,y,z)=(0.308746904, 0.0609171167, -0.155041456, 0.936443567)`; and
- object-local approach `(-0.997760296, 0, 0.0668911785)` with clearance
  radius `0.04 m`.

The baked scene preserves the source-relative object/table transform and moves
the table to world Z `3`. It uses target handle `{1,1}`, object profile `1`,
Free/unowned state, and one right-hand affordance `{id=1}`. Its exact compiled
binary32 scalars are:

| Scalar group and component order | Frozen decimal literals | Binary32 bits |
|---|---|---|
| Table position `(x,y,z)` | `(0, 0.360757500, 3)` | `(0x00000000, 0x3eb8b535, 0x40400000)` |
| Table rotation `(w,x,y,z)` | `(1, 0, 0, 0)` | `(0x3f800000, 0x00000000, 0x00000000, 0x00000000)` |
| Table size `(x,y,z)` | `(2, 0.0399999991, 0.600000024)` | `(0x40000000, 0x3d23d70a, 0x3f19999a)` |
| Object position `(x,y,z)` | `(0.00394439697, 0.503655553, 2.77000808716)` | `(0x3b814000, 0x3f00ef92, 0x403147d0)` |
| Object rotation `(w,x,y,z)` | `(-0.0669774629, 0.670088462, 0.0799996098, -0.734911910)` | `(0xbd892b7b, 0x3f2b8aeb, 0x3da3d6d6, 0xbf3c2330)` |
| Object dimensions `(x,y,z)` | `(0.0645366386, 0.0645366609, 0.240097240)` | `(0x3d842bc9, 0x3d842bcc, 0x3e75dc0d)` |
| Grasp position in object `(x,y,z)` | `(0.0930671170, -0.119263843, 0.0375832170)` | `(0x3dbe99f9, 0xbdf4409a, 0x3d19f0dc)` |
| Grasp rotation in object `(w,x,y,z)` | `(0.308746904, 0.0609171167, -0.155041456, 0.936443567)` | `(0x3e9e1413, 0x3d79843a, 0xbe1ec330, 0x3f6fbac4)` |
| Approach in object `(x,y,z)` | `(-0.997760296, 0, 0.0668911785)` | `(0xbf7f6d38, 0x00000000, 0x3d88fe3e)` |
| Clearance radius | `0.04` | `0x3d23d70a` |

Object bounds are zero-centered with half extents equal to one half of the
compiled object dimensions. The destination is derived from this target and
translates the support exactly `+1.20F` in world Z.

## Selected authoring sources

The first three runtime-ready candidates in retained Task 4 stable order became
slot IDs 1 through 3. Every source has active hand `1`; all stop frames are
exclusive.

| Slot | Stable authoring key | Entry | Align | Contact | Lift | Hold | Stop |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `("nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",1,"pickup_table__alcohol_10__005",125,1)` | 125 | 149 | 150 | 158 | 230 | 250 |
| 2 | `("nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",1,"pickup_table__alcohol_13__005",118,1)` | 118 | 142 | 143 | 149 | 152 | 250 |
| 3 | `("nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",1,"pickup_table__apple_1__000",90,1)` | 90 | 114 | 115 | 139 | 210 | 250 |

The corresponding authoring manifest rows are clip ordinal/range `6 / [1500,
1750)`, `19 / [4750,5000)`, and `94 / [23500,23750)`. Those ordinals are
pack-local diagnostics and are not stored as runtime authority.

## Object-planar slot constants

The decimal column is the Task 4 nine-significant-digit report representation.
The bit column is the exact compiled binary32 authority.

| Slot | `root_x_object_m` | X bits | `root_z_object_m` | Z bits | `root_yaw_object_radians` | Yaw bits |
|---:|---:|---|---:|---|---:|---|
| 1 | -0.396769345 | `0xbecb255a` | -0.0414382927 | `0xbd29bb33` | 1.38969707 | `0x3fb1e198` |
| 2 | -0.402728528 | `0xbece326f` | -0.244846597 | `0xbe7ab911` | 1.71573567 | `0x3fdb9d3a` |
| 3 | -0.467868507 | `0xbeef8c76` | -0.194983453 | `0xbe47a9be` | 1.29209125 | `0x3fa5633f` |

## Exact demo-world mapping

The mapping was rerun in C++ against `make_smart_pickup_demo_target()` using
the production arithmetic: rotate world +Z by the compiled object quaternion,
normalize XZ in double, cast the planar basis to binary32, perform binary32
slot products/additions, and normalize yaw with
`atan2(sin(unwrapped),cos(unwrapped))`.

| Derived planar state | Binary32 value | Bits |
|---|---:|---|
| Object world X | 0.00394439697 | `0x3b814000` |
| Object world Z | 2.77000809 | `0x403147d0` |
| Projected object forward X | -0.995628297 | `0xbf7ee17f` |
| Projected object forward Z | 0.0891630054 | `0x3db69b18` |
| Normalized object forward X | -0.996013939 | `0xbf7efac5` |
| Normalized object forward Z | 0.0891975462 | `0x3db6ad34` |
| Object right X | 0.0891975462 | `0x3db6ad34` |
| Object right Z | 0.996013939 | `0x3f7efac5` |
| Object planar yaw | -1.48148012 | `0xbfbda124` |

The exact mapped planar `PickEntryRoot` values passed to unrestricted preview
are:

| Slot | World X | X bits | World Z | Z bits | World yaw radians | Yaw bits |
|---:|---:|---|---:|---|---:|---|
| 1 | 0.00982666388 | `0x3c210004` | 2.37112403 | `0x4017c07f` | -0.0917830467 | `0xbdbbf8c0` |
| 2 | 0.211892635 | `0x3e58fa62` | 2.34704518 | `0x401635fd` | 0.234255552 | `0x3e6fe0b0` |
| 3 | 0.156417906 | `0x3e202c04` | 2.28661227 | `0x401257db` | -0.189388871 | `0xbe41ef28` |

`PickEntryRoot` is intentionally planar and contains only world X, world Z,
and world yaw. Live flat-root Y is preserved separately by slot navigation.

## Unrestricted runtime preview provenance

For each authored source row, the preview searched all 186 validated stationary
flat snapshots and selected the lowest total cost, breaking an exact cost tie
by the lower flat frame. Costs below are binary32 values serialized with
`max_digits10`; fingerprints and frame indices are exact integers.

| Slot | Stationary flat frame | Snapshot fingerprint | Globally selected matcher source | Clip | Entry local / global | Contact local / global | Total cost |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 185 | 5012902868381772545 | `pickup_table__avocado_1__003` | 153 | 104 / 38354 | 129 / 38379 | 0.353508234 |
| 2 | 9 | 9190637701547627343 | `pickup_table__pear_3__001` | 1674 | 91 / 418591 | 116 / 418616 | 0.357398659 |
| 3 | 184 | 10910211689610565823 | `pickup_table__pear_18__003` | 1659 | 111 / 414861 | 136 / 414886 | 0.38119027 |

The selected matcher manifest ranges are respectively `[38250,38500)`,
`[418500,418750)`, and `[414750,415000)`. The authoring sequences and globally
selected matcher sequences differ for every slot. This is direct evidence that
the slot source did not constrain runtime motion selection.

## Full preview-probe run

The current binary was run at commit
`f330bde8412da6b85828edc04bde65aadd23b883` on x86-64 with GCC
`13.3.0`, completing at `2026-07-18 17:10:02 UTC`. The invocation was:

```text
./interaction_smart_pickup_preview_probe \
  resources/database.bin \
  build/smart-pickup/full-pack \
  build/smart-pickup/beer10-slot-candidates.json
```

The complete canonical stdout contained 214,595 bytes and 20 newline-terminated
JSON records: 19 `candidate_preview` records followed by one `selected_slots`
record. Its SHA-256 was
`3b8b1375a587c9f5b2e04db4578e8e855e0f6099c656d21d2ecbab5aaab68110`,
matching the prior byte-identical repeated Task 5 runs.

| Runtime observation | Exact value |
|---|---:|
| Flat database frames / bones | 22,296 / 23 |
| Interaction frames / clips / features | 511,250 / 2,045 / 71 |
| Retained candidates | 19 |
| Stationary snapshots per candidate | 186 |
| Preview cases | 3,534 |
| Preview workers | 16 |
| Runtime-ready candidates | 19 |
| Target registry registrations | 1 |
| First stationary frame / fingerprint | 0 / 4990634721315354365 |
| Last stationary frame / fingerprint | 210 / 10235195792794356143 |

Measured resource use for this rerun was 18.17 seconds elapsed, 275.38 seconds
user CPU, 1.61 seconds system CPU, and 1,065,988 KiB maximum RSS. These timing
and resource measurements describe this host run and are not deterministic
acceptance authority; the canonical stdout digest and content are.

The compiled-scene probe emitted one newline-terminated record with SHA-256
`0c1547eec6ad38c17bc68bb68f7baf88eebdeb57b86778cad86145d921572981`.
After extracting and canonically encoding each record's `{target,slots}`
structure, its bytes are identical to the preview probe's final
`selected_slots` target and rows.
