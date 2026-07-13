# G1 Exact-Surface Scene Artifact Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the approved Task 5 penetration with deterministic diagnostics, then build and atomically publish one validated 1,770-clip G1 motion pack plus exact-surface multiscene terrain packs.

**Architecture:** Extend the existing Python artifact builder around one canonical vertical-triangle surface provider, rasterize every runtime scene to a G1HF/v2 fixed-diagonal grid, and export the visible OBJ from that same grid. Keep motion, support metadata, and scene data independently versioned, but stage and validate the complete directory before one atomic rename; the C++ work in this plan is limited to the baseline logger and the shared G1HF/v1-v2 loader/sampler that later runtime work consumes.

**Tech Stack:** Python 3.11 at `/home/ubuntu/miniconda3/envs/diffsim/bin/python`; NumPy 2.4; SciPy 1.17; MuJoCo 3.9; USD `pxr`; standard-library `unittest`; C++17; existing Holden headers; Raylib/Raygui from `/home/ubuntu/apps`.

## Global Constraints

- Run Python with `/home/ubuntu/miniconda3/envs/diffsim/bin/python`; add no dependency and use standard-library `unittest`.
- Keep Daniel Holden's matcher, full-pose inertialization, and the fixed 25 Hz update authoritative.
- Keep exactly 31 matching dimensions: the existing 27 pose/trajectory features followed by terrain changes at `[0.25, 0.50, 0.75, 1.00]` metres.
- Complete and run the baseline logger before changing terrain sampling, support data, or runtime behavior.
- Load one immutable motion pack and select independent scene packs; never duplicate `database.bin`, `terrain_features.bin`, or `terrain_support.bin` inside a scene.
- Use one surface for motion features, contacts, support rows, runtime queries, clearance, IK targets, and visible terrain; remove the current `0.14 m` maximum-point dilation.
- New scenes use G1HF/v2 with the fixed diagonal from the minimum-X/minimum-Z node to the maximum-X/maximum-Z node; G1HF/v1 remains readable only for migration.
- To remain deterministic under the production `-ffast-math` build, G1HF/v2
  header values, heights, and derived binary32 node coordinates must be finite
  normal-or-zero values; `cell_size` must be positive normal. V2 subnormals are
  rejected transactionally, and cross-language query parity covers only
  normal-or-zero binary32 coordinates. The historical v1 path is unchanged.
- G1SP/v1 is little-endian, has a 16-byte header, dimension 3, and columns `source_root_height_m`, `source_left_toe_height_m`, and `source_right_toe_height_m`.
- G1WM/v1 is little-endian, has a 16-byte header, matches the scene heightfield's `nx` and `nz`, and stores one row-major `uint8` per cell: `0=blocked`, `1=certified`, `2=stress`.
- Scene heightfield cell size is nominally `0.02 m`; the authoritative serialized
  value is binary32 `0.019999999552965164` (`0x3ca3d70a`). Terrain-feature
  distances are exactly `[0.25, 0.50, 0.75, 1.00]`.
- A positive `landing_hold_seconds` requires at least four route waypoints;
  waypoint index `2` is the exact landing-hold point and a later waypoint
  resumes motion. Zero-hold routes have no special waypoint.
- Surface parity tolerances are `1e-6 m` at grid nodes, `1e-4 m` for deterministic within-cell probes, `0.005 m` against GRAIL source tops away from discontinuities, and at most one `0.02 m` cell of edge movement.
- Preserve the current 459,682-frame/1,770-clip corpus contract: one Takara clip plus all 1,769 sorted GRAIL clips, with `skipped_clips=0`.
- Preserve `resources/database.bin`, `resources/features.bin`, all pre-existing user files, and the last good `resources/g1_terrain/` directory. Never build in place, delete the prior pack first, or stage unrelated dirty files.
- Generated packs, diagnostic CSVs, videos, and validation scratch remain ignored and uncommitted.
- A publish command writes a sibling staging directory, validates every staged byte and JSON contract, fsyncs files/directories, and then performs one rollback-safe directory rename.
- Every JSON reader rejects duplicate keys and non-finite numbers; every binary reader rejects bad magic/version/dimensions, zero/overflowing sizes, truncation, trailing bytes, and non-finite floats before committing destination state.

---

## File and Interface Map

- `motion_match_log.h`: Runtime Task 6-compatible deterministic CSV writer and the immutable Gate A row schema.
- `resources/check_g1_runtime_log.py`: CSV parsing, transition/sequential invariants, first-positive-query diagnosis, and raw-versus-blended penetration classification.
- `tests/python/test_runtime_log.py`: synthetic logger/checker contract tests.
- `tests/cpp/test_motion_match_log.cpp`: strict standalone writer and
  open/write/close fault-injection regression.
- `controller.cpp`: deterministic `MM_TEST_MODE`, `MM_TEST_FRAMES`, `MM_LOG`, and pre-behavior-change stage snapshots only.
- `resources/g1_terrain_builder/schema.py`: `HoldenClip` and `ArtifactSet` gain aligned `(frames, 3)` source-support rows.
- `resources/g1_terrain_builder/terrain.py`: exact transformed source triangles, vertical top intersection, G1HF/v2 grids, fixed-diagonal queries/normals, and grid-derived OBJ export.
- `resources/g1_terrain_builder/database.py`: per-clip source support sampling and combination without changing `database.bin`.
- `resources/g1_terrain_builder/artifacts.py`: G1TF/v1, G1SP/v1, G1WM/v1 serializers, SHA-256 helpers, staged validation, fsync, and rollback-safe atomic publication.
- `resources/g1_terrain_builder/scenes.py`: locked scene/index schemas, deterministic GRAIL selection, procedural surfaces, route/walkability construction, and scene emission.
- `resources/build_g1_terrain_database.py`: one-clip diagnostic and full-corpus orchestration into an unpublished staging candidate.
- `resources/validate_g1_terrain_database.py`: independent motion-pack, scene-pack, route, hash, bounds, source-row, and surface-parity validation.
- `terrain_runtime.h`: migration-only G1HF/v1 bilinear sampling plus authoritative G1HF/v2 fixed-diagonal triangle sampling.
- `tests/cpp/test_terrain_runtime.cpp`: strict v1/v2 loader tests and byte-for-byte Python/C++ surface fixtures.
- `tests/python/test_schema.py`: support-row shape/finite validation.
- `tests/python/test_terrain.py`: exact GRAIL and Python G1HF/v2/OBJ parity tests.
- `tests/python/test_database_builder.py`: root/left/right support sampling and clip-boundary isolation.
- `tests/python/test_artifacts.py`: G1SP/G1WM corruption and transactional complete-pack publication tests.
- `tests/python/test_scenes.py`: all required scene geometry, ordering, routes, bounds, walkability, and parity tests.
- `tests/python/test_build_cli.py`: diagnostic build, complete catalog, hash, validator corruption, and no-publication-on-failure tests.

### Locked cross-plan artifact contracts

The runtime consumer plan must use these names verbatim:

~~~text
resources/g1_terrain/
  database.bin
  terrain_features.bin
  terrain_support.bin
  manifest.json
  validation.json
  scenes/
    index.json
    <scene-id>/
      scene.json
      terrain.bin
      terrain.obj
      walkability.bin
~~~

Binary headers are exact little-endian layouts:

~~~text
G1TF/v1: <4sIII> = magic, version=1, frames, dimensions=4; float32[frames][4]
G1SP/v1: <4sIII> = magic, version=1, frames, dimensions=3; float32[frames][3]
G1HF/v2: <4sIII4f> = magic, version=2, nx, nz, origin_x, origin_z,
           cell_size, exterior_height; float32[nz][nx]
G1WM/v1: <4sIII> = magic, version=1, nx, nz; uint8[nz][nx]
~~~

For G1HF/v2, “finite float32” below always means normal-or-zero binary32;
positive `cell_size` additionally excludes both signed zero and subnormals.
This is a versioned v2 rule, not a retroactive v1 parser change.

The motion manifest schema is `g1-terrain-artifacts/v2`, the scene-index schema
is `g1-terrain-scene-index/v1`, each scene uses `g1-terrain-scene/v1`, and
surface semantics use `g1-terrain-surface/v1`. Tasks below define every JSON
field and signature input; runtime work must consume those names verbatim.

---

### Task 1: Add the pre-change Gate A deterministic logger and checker

**Files:**
- Create: `motion_match_log.h`
- Create: `resources/check_g1_runtime_log.py`
- Create: `tests/python/test_runtime_log.py`
- Create: `tests/cpp/test_motion_match_log.cpp`
- Modify: `controller.cpp:1-140,1578-1660,1940-1980,1990-2520,3038-3063`

**Interfaces:**
- Preserves Runtime Task 6 environment names: `MM_LOG`, `MM_TEST_MODE`, `MM_TEST_FRAMES`, and `MM_TERRAIN_WEIGHT`.
- `MM_TEST_MODE` accepts exactly `sequential`, `flat`, or `terrain`; `terrain` uses weight 4 unless `MM_TERRAIN_WEIGHT` is present.
- Produces: `motion_match_log::open(const char*, char*, int) -> bool`.
- Produces: `motion_match_log::write(const motion_match_log_row&, char*, int) -> bool`.
- Produces: `motion_match_log::close(char*, int) -> bool`.
- Produces: `check_rows(rows) -> dict`,
  `compare_control(treatment, control) -> tuple[float,float]`,
  `check_gate_a_contract(rows, expected_frames=375) -> dict`, and
  `diagnose_gate_a(rows) -> dict`.
- The CSV keeps every Runtime Task 6 column and adds immutable query points, stage heights/clearances, and adjustment/clamp displacement. Later runtime plans append columns; they do not rename these.
- Row timing is immutable: `query_database_frame/query_range` identify the
  pre-search incumbent; `selected_database_frame/source_range` identify the
  search decision whose selected cost/error are logged; and
  `database_frame/range` identify the range-safe post-advance pose used by the
  raw, inertialized, and rendered diagnostics.
- Bounded test modes and deterministic CSV logging are desktop-only. A
  `PLATFORM_WEB` build rejects every positive frame limit, every non-live test
  mode, and (outside the legacy `MM_DISCRETE` stream) `MM_LOG`, because the
  Emscripten main loop cannot reach this task's synchronous close/cleanup path.
- Costs and raw terrain error are non-negative sums of squares. Search
  materializes normalized float32 query values independently from the logged
  incumbent calculation, so a searched near-tie may differ by at most four
  float32 ULPs; unsearched selected/incumbent costs remain exactly equal.
- Treatment and weight-zero control runs keep identical scripted metadata but
  may follow different root trajectories. Each run therefore computes its own
  active-terrain aggregate using `max(abs(terrain0..3)) > 0.05`; both active
  sets must be non-empty, rather than reusing treatment row indices in control.

- [ ] **Step 1: Write failing checker tests for invariants and penetration classification**

Create `tests/python/test_runtime_log.py`:

~~~python
import tempfile
import struct
import unittest

from resources.check_g1_runtime_log import (
    CSV_COLUMNS,
    check_gate_a_contract,
    check_rows,
    compare_control,
    diagnose_gate_a,
    read_rows,
)


def row(frame, database_frame, **changes):
    values = {
        "frame": str(frame),
        "fixed_dt": "0.04",
        "scene_id": "grail-curb-default",
        "mode": "terrain",
        "route": "curb-forward",
        "query_bits_hex": "00000000" * 31,
        "query_database_frame": str(database_frame),
        "query_range": "0",
        "selected_database_frame": str(database_frame),
        "database_frame": str(database_frame),
        "range": "0",
        "source_range": "0",
        "searched": "0",
        "transitioned": "0",
        "incumbent_cost": "1.0",
        "selected_cost": "1.0",
        "selected_terrain_error": "0.0",
        "effective_terrain_weight": "4.0",
        "terrain0": "0.0", "terrain1": "0.0",
        "terrain2": "0.0", "terrain3": "0.0",
        "raw_selected_min_clearance": "0.03",
        "inertialized_min_clearance": "0.03",
        "rendered_min_clearance": "0.03",
        "raw_selected_hips_y": "0.8",
        "inertialized_hips_y": "0.8",
        "rendered_hips_y": "0.8",
        "hips_inertial_offset_y": "0.0",
        "runtime_root_surface_height": "0.0",
        "runtime_left_toe_surface_height": "0.0",
        "runtime_right_toe_surface_height": "0.0",
        "adjustment_xz": "0.0", "adjustment_y": "0.0",
        "clamp_xz": "0.0", "clamp_y": "0.0",
        "matching_enabled": "1", "adjustment_enabled": "1",
        "clamping_enabled": "1", "support_retargeting_enabled": "0",
        "ik_enabled": "0",
    }
    for sample in range(4):
        for axis in "xyz":
            values[f"terrain_point{sample}_{axis}"] = "0.0"
    for stage in ("raw_selected", "inertialized", "rendered"):
        for joint in ("hips", "left_toe", "right_toe"):
            values[f"{stage}_{joint}_clearance"] = "0.03"
    values.update({key: str(value) for key, value in changes.items()})
    if "query_bits_hex" not in changes:
        values["query_bits_hex"] = "00000000" * 27 + "".join(
            struct.pack(">f", float(values[f"terrain{sample}"])).hex()
            for sample in range(4))
    return values


class RuntimeLogTests(unittest.TestCase):
    def test_rejects_nonsequential_advance_without_transition(self):
        rows = [
            row(0, 10),
            row(
                1, 10, query_database_frame=10,
                selected_database_frame=10),
        ]
        with self.assertRaisesRegex(ValueError, "nonsequential"):
            check_rows(rows)

    def test_rejects_transition_that_does_not_beat_incumbent(self):
        rows = [row(
            0, 20, query_database_frame=10, selected_database_frame=20,
            transitioned=1, searched=1,
            incumbent_cost=1.0, selected_cost=1.5,
        )]
        with self.assertRaisesRegex(ValueError, "beat incumbent"):
            check_rows(rows)

    def test_rejects_incomplete_query_bit_snapshot(self):
        with self.assertRaisesRegex(ValueError, "31 float bit patterns"):
            check_rows([row(0, 10, query_bits_hex="0" * 247)])

    def test_rejects_nonfinite_or_terrain_inconsistent_query_bits(self):
        with self.assertRaisesRegex(ValueError, "non-finite query"):
            check_rows([row(
                0, 10,
                query_bits_hex="00000000" * 27 + "7fc00000" +
                "00000000" * 3)])
        with self.assertRaisesRegex(ValueError, "terrain query bits"):
            check_rows([row(
                0, 10, terrain0=0.25,
                query_bits_hex="00000000" * 31)])

    def test_gate_a_classifies_only_blended_pose_penetration(self):
        rows = [
            row(0, 10),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10, terrain0=0.12,
                raw_selected_min_clearance=0.02,
                inertialized_min_clearance=-0.01,
                rendered_min_clearance=-0.02,
                hips_inertial_offset_y=-0.04,
                adjustment_y=-0.01,
            ),
        ]
        report = diagnose_gate_a(rows)
        self.assertEqual(report["first_positive_query_frame"], 1)
        self.assertEqual(report["penetration_class"], "blended-rendered")
        self.assertEqual(report["first_penetration_frame"], 1)

    def test_gate_a_classifies_raw_selected_penetration(self):
        rows = [row(
            0, 10, terrain1=0.08,
            raw_selected_min_clearance=-0.003,
            inertialized_min_clearance=0.01,
            rendered_min_clearance=0.01,
        )]
        self.assertEqual(
            diagnose_gate_a(rows)["penetration_class"], "raw-selected")

    def test_gate_a_uses_earliest_penetration_with_raw_same_frame_priority(self):
        rows = [
            row(0, 10, terrain0=.1, rendered_min_clearance=-.001),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                raw_selected_min_clearance=-.002),
        ]
        report = diagnose_gate_a(rows)
        self.assertEqual(report["first_penetration_frame"], 0)
        self.assertEqual(report["penetration_class"], "blended-rendered")
        rows[0]["raw_selected_min_clearance"] = "-.003"
        self.assertEqual(
            diagnose_gate_a(rows)["penetration_class"], "raw-selected")

    def test_gate_a_enforces_the_exact_run_contract(self):
        valid = row(0, 10, terrain0=.1)
        self.assertEqual(check_gate_a_contract(
            [valid], expected_frames=1)["frames"], 1)
        for name, bad in (
            ("scene_id", "other"), ("mode", "live"),
            ("route", "manual"), ("effective_terrain_weight", "0"),
            ("matching_enabled", "0"), ("adjustment_enabled", "0"),
            ("clamping_enabled", "0"),
            ("support_retargeting_enabled", "1"), ("ik_enabled", "1"),
            ("fixed_dt", "0.041"),
        ):
            changed = dict(valid)
            changed[name] = bad
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "Gate A"):
                check_gate_a_contract([changed], expected_frames=1)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            check_gate_a_contract([valid], expected_frames=2)

    def test_terrain_treatment_must_improve_raw_terrain_error(self):
        treatment = [row(0, 10, terrain0=0.1, selected_terrain_error=0.5)]
        control = [row(
            0, 10, terrain0=0.1, effective_terrain_weight=0,
            selected_terrain_error=2.0,
        )]
        self.assertEqual(compare_control(treatment, control), (0.5, 2.0))

    def test_control_uses_its_own_active_terrain_rows(self):
        treatment = [
            row(0, 10, terrain0=-.1, selected_terrain_error=.5),
            row(1, 11, query_database_frame=10,
                selected_database_frame=10, selected_terrain_error=9),
        ]
        control = [
            row(0, 10, effective_terrain_weight=0,
                selected_terrain_error=.25),
            row(1, 11, query_database_frame=10,
                selected_database_frame=10, terrain1=.1,
                effective_terrain_weight=0, selected_terrain_error=2),
        ]
        self.assertEqual(compare_control(treatment, control), (.5, 2))

    def test_control_comparison_rejects_wrong_weight_or_script(self):
        treatment = [row(0, 10, terrain0=.1, selected_terrain_error=.5)]
        control = [row(
            0, 10, terrain0=.1, effective_terrain_weight=0,
            selected_terrain_error=2.0)]
        control[0]["route"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "scripted input"):
            compare_control(treatment, control)
        control[0]["route"] = treatment[0]["route"]
        control[0]["effective_terrain_weight"] = "1"
        with self.assertRaisesRegex(ValueError, "weight"):
            compare_control(treatment, control)

    def test_range_change_requires_transition(self):
        rows = [
            row(0, 10, range=0, source_range=0, query_range=0),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                range=1, source_range=1, query_range=0),
        ]
        with self.assertRaisesRegex(ValueError, "range change"):
            check_rows(rows)

    def test_query_frame_must_continue_the_prior_rendered_pose(self):
        rows = [
            row(0, 10, query_database_frame=9, selected_database_frame=9),
            row(1, 8, query_database_frame=7, selected_database_frame=7),
        ]
        with self.assertRaisesRegex(ValueError, "query frame"):
            check_rows(rows)

    def test_detects_eight_frame_snapback_even_at_transitions(self):
        rows = []
        for frame in range(24):
            current = 10 + frame % 8
            restarting = frame > 0 and frame % 8 == 0
            query_frame = 17 if restarting else current - 1
            selected_frame = 9 if restarting else query_frame
            rows.append(row(
                frame, current,
                query_database_frame=query_frame,
                selected_database_frame=selected_frame,
                searched=int(restarting), transitioned=int(restarting),
                incumbent_cost=2.0 if restarting else 1.0,
                selected_cost=1.0,
            ))
        with self.assertRaisesRegex(ValueError, "sub-stride period 8"):
            check_rows(rows)

    def test_read_rows_rejects_duplicate_csv_columns(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            stream.write("frame,frame\n0,0\n")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "duplicate CSV column"):
                read_rows(stream.name)

    def test_read_rows_rejects_reordered_immutable_prefix(self):
        fields = list(CSV_COLUMNS)
        fields[0], fields[1] = fields[1], fields[0]
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            stream.write(",".join(fields) + "\n")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "immutable CSV prefix"):
                read_rows(stream.name)


if __name__ == "__main__":
    unittest.main()
~~~

Also cover both searched and unsearched no-transition rows, accepted four-ULP
searched near-ties and rejected five-ULP drift, transitioned near-ties, negative
cost/error fields, non-positive `fixed_dt`, invalid negative database/range
indices, terrain weights outside `[0,10]`, negative XZ displacement magnitudes,
rows wider than their CSV header, and the requirement that both treatment and
control have an independently active terrain set. Every integer field must fit
the writer's signed C++ `int`, and every non-integer numeric field must be
finite and representable as the writer's float32 type.

- [ ] **Step 2: Run the tests and verify the missing checker failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
~~~

Expected: `ERROR` with `ModuleNotFoundError: No module named 'resources.check_g1_runtime_log'`.

- [ ] **Step 3: Implement the Runtime Task 6-compatible checker and Gate A report**

Create `resources/check_g1_runtime_log.py`:

~~~python
#!/usr/bin/env python3
import argparse
import csv
import math
import struct


CSV_COLUMNS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex",
    "query_database_frame", "query_range", "selected_database_frame",
    "database_frame", "range", "source_range", "searched", "transitioned",
    "incumbent_cost",
    "selected_cost", "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "raw_selected_hips_y", "inertialized_hips_y", "rendered_hips_y",
    "hips_inertial_offset_y", "runtime_root_surface_height",
    "runtime_left_toe_surface_height", "runtime_right_toe_surface_height",
    "raw_selected_hips_clearance", "raw_selected_left_toe_clearance",
    "raw_selected_right_toe_clearance", "raw_selected_min_clearance",
    "inertialized_hips_clearance", "inertialized_left_toe_clearance",
    "inertialized_right_toe_clearance", "inertialized_min_clearance",
    "rendered_hips_clearance", "rendered_left_toe_clearance",
    "rendered_right_toe_clearance", "rendered_min_clearance",
    "adjustment_xz", "adjustment_y", "clamp_xz", "clamp_y",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "ik_enabled",
)
REQUIRED_COLUMNS = set(CSV_COLUMNS)
TEXT_COLUMNS = {"scene_id", "mode", "route", "query_bits_hex"}
INTEGER_COLUMNS = {
    "frame", "query_database_frame", "query_range", "selected_database_frame",
    "database_frame", "range", "source_range", "searched", "transitioned",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "ik_enabled",
}
FLAG_COLUMNS = {
    "searched", "transitioned", "matching_enabled", "adjustment_enabled",
    "clamping_enabled", "support_retargeting_enabled", "ik_enabled",
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate CSV column")
        if fields[:len(CSV_COLUMNS)] != list(CSV_COLUMNS):
            raise ValueError("immutable CSV prefix was renamed or reordered")
        missing = sorted(REQUIRED_COLUMNS - set(fields))
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")
        rows = []
        for index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"row {index}: CSV data is wider than header")
            rows.append(row)
        return rows


def _finite(row, name, index):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {index}: invalid {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"row {index}: non-finite {name}")
    return value


def _integer(row, name, index):
    value = _finite(row, name, index)
    if value != int(value):
        raise ValueError(f"row {index}: non-integer {name}")
    integer = int(value)
    if integer < -(2 ** 31) or integer > 2 ** 31 - 1:
        raise ValueError(f"row {index}: {name} is outside signed int32")
    return integer


def _float32_bits(value, index, label):
    try:
        return int.from_bytes(struct.pack(">f", value), "big")
    except (OverflowError, struct.error) as error:
        raise ValueError(f"row {index}: {label} is not float32") from error


def _float32(row, name, index):
    value = _finite(row, name, index)
    _float32_bits(value, index, name)
    return value


def _float32_ulp_distance(left, right, index):
    left_bits = _float32_bits(left, index, "cost")
    right_bits = _float32_bits(right, index, "cost")
    return abs(left_bits - right_bits)


def _check_query_snapshot(row, index):
    snapshot = row["query_bits_hex"]
    values = []
    for dimension in range(31):
        bits = snapshot[dimension * 8:(dimension + 1) * 8]
        value = struct.unpack(">f", bytes.fromhex(bits))[0]
        if not math.isfinite(value):
            raise ValueError(
                f"row {index}: non-finite query dimension {dimension}")
        values.append(value)
    for sample in range(4):
        terrain = _float32(row, f"terrain{sample}", index)
        terrain_bits = struct.pack(">f", terrain).hex()
        query_bits = snapshot[(27 + sample) * 8:(28 + sample) * 8]
        if query_bits != terrain_bits:
            raise ValueError(
                f"row {index}: terrain query bits disagree at sample {sample}")
    return values


def check_substride(rows, minimum_period=13):
    values = [_integer(row, "database_frame", i) for i, row in enumerate(rows)]
    for period in range(1, minimum_period):
        width = 3 * period
        for start in range(0, len(values) - width + 1):
            a = values[start:start + period]
            if a == values[start + period:start + 2 * period] == \
                    values[start + 2 * period:start + 3 * period]:
                raise ValueError(
                    f"row {start}: repeated sub-stride period {period}")


def check_rows(rows):
    if not rows:
        raise ValueError("runtime log is empty")
    previous = None
    previous_range = None
    for index, row in enumerate(rows):
        frame = _integer(row, "frame", index)
        query_frame = _integer(row, "query_database_frame", index)
        query_range = _integer(row, "query_range", index)
        selected_frame = _integer(row, "selected_database_frame", index)
        current = _integer(row, "database_frame", index)
        current_range = _integer(row, "range", index)
        source_range = _integer(row, "source_range", index)
        transitioned = _integer(row, "transitioned", index)
        searched = _integer(row, "searched", index)
        for name, value in (
                ("query_database_frame", query_frame),
                ("selected_database_frame", selected_frame),
                ("database_frame", current),
                ("query_range", query_range),
                ("range", current_range),
                ("source_range", source_range)):
            if value < 0:
                raise ValueError(f"row {index}: {name} must be nonnegative")
        fixed_dt = _float32(row, "fixed_dt", index)
        if fixed_dt <= 0.0:
            raise ValueError(f"row {index}: fixed_dt must be positive")
        terrain_weight = _float32(
            row, "effective_terrain_weight", index)
        if not 0.0 <= terrain_weight <= 10.0:
            raise ValueError(
                f"row {index}: effective_terrain_weight must be in [0, 10]")
        for name in ("adjustment_xz", "clamp_xz"):
            if _float32(row, name, index) < 0.0:
                raise ValueError(f"row {index}: {name} must be nonnegative")
        if frame != index:
            raise ValueError(f"row {index}: frame sequence is {frame}")
        if transitioned not in (0, 1) or searched not in (0, 1):
            raise ValueError(f"row {index}: flags must be 0 or 1")
        if previous is not None and query_frame != previous:
            raise ValueError(
                f"row {index}: query frame {query_frame} does not match "
                f"prior pose frame {previous}")
        if previous_range is not None and query_range != previous_range:
            raise ValueError(
                f"row {index}: query range {query_range} does not match "
                f"prior pose range {previous_range}")
        if transitioned and not searched:
            raise ValueError(f"row {index}: transition without search")
        if transitioned and selected_frame == query_frame:
            raise ValueError(
                f"row {index}: transitioned with unchanged selected frame")
        if not transitioned and selected_frame != query_frame:
            raise ValueError(
                f"row {index}: selected frame changed without transition")
        if current not in (selected_frame, selected_frame + 1):
            raise ValueError(
                f"row {index}: post-advance frame is inconsistent with selected frame")
        if current_range != source_range:
            raise ValueError(
                f"row {index}: pose range differs from selected source range")
        if not transitioned and query_range != source_range:
            raise ValueError(
                f"row {index}: source range changed without transition")
        incumbent = _finite(row, "incumbent_cost", index)
        selected = _finite(row, "selected_cost", index)
        selected_terrain_error = _finite(
            row, "selected_terrain_error", index)
        for value in (incumbent, selected, selected_terrain_error):
            _float32_bits(value, index, "cost")
        for name, value in (
                ("incumbent_cost", incumbent),
                ("selected_cost", selected),
                ("selected_terrain_error", selected_terrain_error)):
            if value < 0.0:
                raise ValueError(f"row {index}: negative {name}")
        # Search and incumbent costs are independently normalized and
        # materialized as float32, so near-ties can round a few ULPs apart.
        if transitioned and not selected < incumbent:
            cost_ulps = _float32_ulp_distance(selected, incumbent, index)
            if cost_ulps > 4:
                raise ValueError(
                    f"row {index}: transition did not beat incumbent cost "
                    f"and differs by {cost_ulps} float32 ULPs")
        if not transitioned and not searched and selected != incumbent:
            raise ValueError(
                f"row {index}: unsearched no-transition selected cost "
                "differs from incumbent cost")
        if not transitioned and searched:
            cost_ulps = _float32_ulp_distance(selected, incumbent, index)
            if cost_ulps > 4:
                raise ValueError(
                    f"row {index}: searched no-transition selected cost "
                    f"differs by {cost_ulps} float32 ULPs")
        for name in CSV_COLUMNS:
            if name in TEXT_COLUMNS:
                if not row.get(name):
                    raise ValueError(f"row {index}: empty {name}")
                if name == "query_bits_hex" and (
                        len(row[name]) != 31 * 8 or
                        any(character not in "0123456789abcdef"
                            for character in row[name])):
                    raise ValueError(
                        f"row {index}: query_bits_hex is not 31 float bit patterns")
            elif name in INTEGER_COLUMNS:
                _integer(row, name, index)
            else:
                _float32(row, name, index)
        for name in FLAG_COLUMNS:
            if _integer(row, name, index) not in (0, 1):
                raise ValueError(f"row {index}: {name} must be 0 or 1")
        _check_query_snapshot(row, index)
        if previous is not None and not transitioned and current != previous + 1:
            raise ValueError(
                f"row {index}: nonsequential advance {previous}->{current}")
        if (previous_range is not None and not transitioned and
                current_range != previous_range):
            raise ValueError(
                f"row {index}: range change without transition")
        previous = current
        previous_range = current_range
    check_substride(rows)
    return {
        "frames": len(rows),
        "transitions": sum(int(row["transitioned"]) for row in rows),
    }


def compare_control(treatment, control):
    check_rows(treatment)
    check_rows(control)
    if len(treatment) != len(control):
        raise ValueError("control and treatment lengths differ")
    metadata = ("frame", "fixed_dt", "scene_id", "mode", "route")
    for index, (treatment_row, control_row) in enumerate(
            zip(treatment, control)):
        if any(treatment_row[name] != control_row[name] for name in metadata):
            raise ValueError(
                f"row {index}: control and treatment script metadata differ")
        if _finite(treatment_row, "effective_terrain_weight", index) != 4.0:
            raise ValueError(f"row {index}: treatment weight must be 4")
        if _finite(control_row, "effective_terrain_weight", index) != 0.0:
            raise ValueError(f"row {index}: control weight must be 0")

    def active_errors(rows, label):
        errors = [
            _finite(row, "selected_terrain_error", index)
            for index, row in enumerate(rows)
            if max(abs(_finite(row, f"terrain{sample}", index))
                   for sample in range(4)) > 0.05
        ]
        if not errors:
            raise ValueError(f"{label} terrain query never became active")
        return errors

    treatment_errors = active_errors(treatment, "treatment")
    control_errors = active_errors(control, "control")
    treatment_error = sum(treatment_errors) / len(treatment_errors)
    control_error = sum(control_errors) / len(control_errors)
    if not treatment_error < control_error:
        raise ValueError(
            "terrain treatment did not improve error: "
            f"{treatment_error} >= {control_error}")
    return treatment_error, control_error


def check_gate_a_contract(rows, expected_frames=375):
    summary = check_rows(rows)
    if len(rows) != expected_frames:
        raise ValueError(
            f"Gate A requires exactly {expected_frames} rows, got {len(rows)}")
    expected_dt = struct.pack(">f", 0.04)
    expected_text = {
        "scene_id": "grail-curb-default",
        "mode": "terrain",
        "route": "curb-forward",
    }
    expected_flags = {
        "matching_enabled": (1, "matching"),
        "adjustment_enabled": (1, "adjustment"),
        "clamping_enabled": (1, "clamping"),
        "support_retargeting_enabled": (0, "support retargeting"),
        "ik_enabled": (0, "IK"),
    }
    for index, row in enumerate(rows):
        fixed_dt = _finite(row, "fixed_dt", index)
        if struct.pack(">f", fixed_dt) != expected_dt:
            raise ValueError(f"row {index}: Gate A fixed_dt must be float32 0.04")
        for name, expected in expected_text.items():
            if row[name] != expected:
                raise ValueError(
                    f"row {index}: Gate A {name.replace('_id', '')} must be {expected}")
        if _finite(row, "effective_terrain_weight", index) != 4.0:
            raise ValueError(f"row {index}: Gate A weight must be 4")
        for name, (expected, label) in expected_flags.items():
            if _integer(row, name, index) != expected:
                raise ValueError(
                    f"row {index}: Gate A {label} must be {expected}")
    return summary


def diagnose_gate_a(rows):
    check_rows(rows)
    positive = next((
        index for index, row in enumerate(rows)
        if max(float(row[f"terrain{sample}"]) for sample in range(4)) > 1e-6
    ), None)
    if positive is None:
        raise ValueError("terrain query never became positive")
    first_penetration = None
    classification = "no-penetration"
    for index, row in enumerate(rows):
        raw = float(row["raw_selected_min_clearance"]) < 0.0
        blended = (
            float(row["inertialized_min_clearance"]) < 0.0
            or float(row["rendered_min_clearance"]) < 0.0)
        if raw or blended:
            first_penetration = index
            classification = "raw-selected" if raw else "blended-rendered"
            break
    diagnostic = rows[first_penetration if first_penetration is not None else positive]
    return {
        "first_positive_query_frame": int(rows[positive]["frame"]),
        "first_penetration_frame": (
            None if first_penetration is None
            else int(rows[first_penetration]["frame"])),
        "penetration_class": classification,
        "selected_range": int(diagnostic["range"]),
        "hips_inertial_offset_y": float(diagnostic["hips_inertial_offset_y"]),
        "adjustment_xz": float(diagnostic["adjustment_xz"]),
        "adjustment_y": float(diagnostic["adjustment_y"]),
        "clamp_xz": float(diagnostic["clamp_xz"]),
        "clamp_y": float(diagnostic["clamp_y"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--compare-control")
    parser.add_argument("--gate-a", action="store_true")
    args = parser.parse_args(argv)
    rows = read_rows(args.log)
    summary = check_rows(rows)
    if args.compare_control:
        treatment, control = compare_control(
            rows, read_rows(args.compare_control))
        print(
            "VALID terrain-comparison "
            f"treatment={treatment:.9g} control={control:.9g}")
    if args.gate_a:
        check_gate_a_contract(rows)
        report = diagnose_gate_a(rows)
        print(
            "VALID gate-a "
            f"first_positive={report['first_positive_query_frame']} "
            f"first_penetration={report['first_penetration_frame']} "
            f"classification={report['penetration_class']} "
            f"range={report['selected_range']} "
            f"hips_offset_y={report['hips_inertial_offset_y']:.9g} "
            f"adjust_y={report['adjustment_y']:.9g} "
            f"clamp_y={report['clamp_y']:.9g}")
    print(
        f"VALID runtime-log frames={summary['frames']} "
        f"transitions={summary['transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 4: Run the checker tests and verify they pass**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
~~~

Expected: all logger/checker contract tests pass and report `OK`.

- [ ] **Step 5: Add the exact CSV row and writer contract**

Create `motion_match_log.h`:

~~~cpp
#pragma once

#include <stdlib.h>
#include "array.h"
#include "vec.h"
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static inline bool motion_match_query_is_finite_31d(
    const slice1d<float> query)
{
    if (query.size != 31) return false;
    for (int i = 0; i < 31; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, &query(i), sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            return false;
    }
    return true;
}

static inline bool motion_match_query_bits_hex(
    char* output, int capacity, const slice1d<float> query)
{
    if (output == NULL || capacity < 31 * 8 + 1 ||
        !motion_match_query_is_finite_31d(query))
        return false;
    for (int i = 0; i < 31; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, &query(i), sizeof(bits));
        snprintf(output + i * 8, 9, "%08x", (unsigned)bits);
    }
    output[31 * 8] = '\0';
    return true;
}

struct motion_match_pose_diagnostic
{
    float hips_y = 0.0f;
    float hips_clearance = 0.0f;
    float left_toe_clearance = 0.0f;
    float right_toe_clearance = 0.0f;
    // Minimum over Hips plus both knees, ankles, and toes (seven probes).
    float minimum_clearance = 0.0f;
};

struct motion_match_log_row
{
    int frame = 0;
    float fixed_dt = 0.04f;
    const char* scene_id = "grail-curb-default";
    const char* mode = "live";
    const char* route = "manual";
    const char* query_bits_hex = "";
    int query_database_frame = 0;
    int query_range = 0;
    int selected_database_frame = 0;
    int database_frame = 0;
    int range = 0;
    int source_range = 0;
    bool searched = false;
    bool transitioned = false;
    float incumbent_cost = 0.0f;
    float selected_cost = 0.0f;
    float selected_terrain_error = 0.0f;
    float effective_terrain_weight = 0.0f;
    float terrain[4] = {};
    vec3 terrain_points[4] = {};
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic rendered;
    float hips_inertial_offset_y = 0.0f;
    float runtime_root_surface_height = 0.0f;
    float runtime_left_toe_surface_height = 0.0f;
    float runtime_right_toe_surface_height = 0.0f;
    float adjustment_xz = 0.0f;
    float adjustment_y = 0.0f;
    float clamp_xz = 0.0f;
    float clamp_y = 0.0f;
    bool matching_enabled = true;
    bool adjustment_enabled = true;
    bool clamping_enabled = true;
    bool support_retargeting_enabled = false;
    bool ik_enabled = false;
};

struct motion_match_log
{
    FILE* file = NULL;
    const char* path = NULL;

    bool io_error(
        char* error, int error_capacity,
        const char* action, const int saved_errno) const
    {
        if (error != NULL && error_capacity > 0) {
            snprintf(
                error, (size_t)error_capacity, "%s: cannot %s motion log (%s)",
                path != NULL ? path : "<disabled>", action,
                strerror(saved_errno != 0 ? saved_errno : EIO));
        }
        return false;
    }

    bool open(const char* log_path, char* error, int error_capacity)
    {
        if (log_path == NULL) return true;
        path = log_path;
        file = fopen(path, "w");
        if (file == NULL) {
            return io_error(error, error_capacity, "open", errno);
        }
        const bool header_ok = fprintf(file,
            "frame,fixed_dt,scene_id,mode,route,query_bits_hex,"
            "query_database_frame,query_range,selected_database_frame,"
            "database_frame,range,source_range,searched,transitioned,"
            "incumbent_cost,selected_cost,selected_terrain_error,"
            "effective_terrain_weight,terrain0,terrain1,terrain2,terrain3,"
            "terrain_point0_x,terrain_point0_y,terrain_point0_z,"
            "terrain_point1_x,terrain_point1_y,terrain_point1_z,"
            "terrain_point2_x,terrain_point2_y,terrain_point2_z,"
            "terrain_point3_x,terrain_point3_y,terrain_point3_z,"
            "raw_selected_hips_y,inertialized_hips_y,rendered_hips_y,"
            "hips_inertial_offset_y,runtime_root_surface_height,"
            "runtime_left_toe_surface_height,"
            "runtime_right_toe_surface_height,"
            "raw_selected_hips_clearance,raw_selected_left_toe_clearance,"
            "raw_selected_right_toe_clearance,raw_selected_min_clearance,"
            "inertialized_hips_clearance,inertialized_left_toe_clearance,"
            "inertialized_right_toe_clearance,inertialized_min_clearance,"
            "rendered_hips_clearance,rendered_left_toe_clearance,"
            "rendered_right_toe_clearance,rendered_min_clearance,"
            "adjustment_xz,adjustment_y,clamp_xz,clamp_y,matching_enabled,"
            "adjustment_enabled,clamping_enabled,support_retargeting_enabled,"
            "ik_enabled\n") >= 0;
        if (!header_ok || fflush(file) != 0) {
            const int saved_errno = errno;
            fclose(file);
            file = NULL;
            return io_error(error, error_capacity, "initialize", saved_errno);
        }
        return true;
    }

    bool write(
        const motion_match_log_row& r,
        char* error, const int error_capacity)
    {
        if (file == NULL) return true;
        bool ok = fprintf(file,
            "%d,%.9g,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
            r.frame, r.fixed_dt, r.scene_id, r.mode, r.route,
            r.query_bits_hex,
            r.query_database_frame, r.query_range, r.selected_database_frame,
            r.database_frame, r.range, r.source_range,
            (int)r.searched, (int)r.transitioned,
            r.incumbent_cost, r.selected_cost, r.selected_terrain_error,
            r.effective_terrain_weight,
            r.terrain[0], r.terrain[1], r.terrain[2], r.terrain[3]) >= 0;
        for (int i = 0; ok && i < 4; ++i) {
            ok = fprintf(file, ",%.9g,%.9g,%.9g",
                    r.terrain_points[i].x,
                    r.terrain_points[i].y,
                    r.terrain_points[i].z) >= 0;
        }
        if (ok) ok = fprintf(file,
            ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
            "%d,%d,%d,%d,%d\n",
            r.raw_selected.hips_y, r.inertialized.hips_y,
            r.rendered.hips_y, r.hips_inertial_offset_y,
            r.runtime_root_surface_height,
            r.runtime_left_toe_surface_height,
            r.runtime_right_toe_surface_height,
            r.raw_selected.hips_clearance,
            r.raw_selected.left_toe_clearance,
            r.raw_selected.right_toe_clearance,
            r.raw_selected.minimum_clearance,
            r.inertialized.hips_clearance,
            r.inertialized.left_toe_clearance,
            r.inertialized.right_toe_clearance,
            r.inertialized.minimum_clearance,
            r.rendered.hips_clearance,
            r.rendered.left_toe_clearance,
            r.rendered.right_toe_clearance,
            r.rendered.minimum_clearance,
            r.adjustment_xz, r.adjustment_y, r.clamp_xz, r.clamp_y,
            (int)r.matching_enabled, (int)r.adjustment_enabled,
            (int)r.clamping_enabled, (int)r.support_retargeting_enabled,
            (int)r.ik_enabled) >= 0;
        if (ok) ok = fflush(file) == 0;
        return ok ? true : io_error(error, error_capacity, "write", errno);
    }

    bool close(char* error, const int error_capacity)
    {
        if (file == NULL) return true;
        int saved_errno = 0;
        if (fflush(file) != 0) saved_errno = errno;
        if (fclose(file) != 0 && saved_errno == 0) saved_errno = errno;
        file = NULL;
        if (saved_errno != 0) {
            const bool result = io_error(
                error, error_capacity, "close", saved_errno);
            path = NULL;
            return result;
        }
        path = NULL;
        return true;
    }
};
~~~

Also create `tests/cpp/test_motion_match_log.cpp` with
`#include "motion_match_log.h"` deliberately first. The test must use temporary
files plus `/dev/full`/`freopen` fault injection to prove four cases: header
initialization failure makes `open` return false, a row flush failure makes
`write` return false, a buffered final flush makes `close` return false, and a
normal open/write/close succeeds. Every failure diagnostic names its operation,
every path leaves `log.file == NULL` after close, and every temporary file is
removed. When given one output-path argument, retain a semantically valid
one-row success CSV so the Python checker can exercise the native
writer-to-reader boundary.
The native test must construct finite, positive-infinity, and NaN values by
copying exact uint32 bit patterns and exercise both the cheap 31D predicate and
hex serializer. Compile/run this test once under strict flags and again under
the production `-O3 -ffast-math` flags; standard-library finiteness predicates
are not a valid implementation under that build.

- [ ] **Step 6: Add deterministic modes and non-mutating pose snapshots to the controller**

Include `motion_match_log.h` immediately after `terrain_runtime.h`, then add these helpers above `main`:

~~~cpp
enum g1_test_mode { G1_TestLive, G1_TestSequential, G1_TestFlat, G1_TestTerrain };

struct g1_test_config
{
    g1_test_mode mode = G1_TestLive;
    const char* name = "live";
    const char* route = "manual";
    int frame_limit = 0;
};

static bool g1_parse_test_config(
    g1_test_config& out, char* error, const int error_capacity)
{
    const char* mode = getenv("MM_TEST_MODE");
    if (mode != NULL) {
        if (strcmp(mode, "sequential") == 0) {
            out.mode = G1_TestSequential; out.name = mode;
            out.route = "curb-forward";
        } else if (strcmp(mode, "flat") == 0) {
            out.mode = G1_TestFlat; out.name = mode;
            out.route = "curb-forward";
        } else if (strcmp(mode, "terrain") == 0) {
            out.mode = G1_TestTerrain; out.name = mode;
            out.route = "curb-forward";
        } else {
            return g1_error(error, error_capacity,
                "MM_TEST_MODE must be sequential, flat, or terrain, got '%s'",
                mode);
        }
    }
    if (!g1_parse_test_frames(out.frame_limit, error, error_capacity))
        return false;
    if (out.mode != G1_TestLive && out.frame_limit <= 0)
        return g1_error(error, error_capacity,
            "MM_TEST_FRAMES must be positive in deterministic test mode");
    return true;
}

static motion_match_pose_diagnostic g1_pose_diagnostic(
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents,
    const heightfield& terrain)
{
    array1d<vec3> positions(local_positions.size);
    array1d<quat> rotations(local_rotations.size);
    forward_kinematics_full(
        positions, rotations, local_positions, local_rotations, parents);
    motion_match_pose_diagnostic out;
    out.hips_y = positions(G1_Hips).y;
    out.hips_clearance = positions(G1_Hips).y - heightfield_sample(
        terrain, positions(G1_Hips).x, positions(G1_Hips).z);
    out.left_toe_clearance = positions(G1_LeftToe).y - heightfield_sample(
        terrain, positions(G1_LeftToe).x, positions(G1_LeftToe).z);
    out.right_toe_clearance = positions(G1_RightToe).y - heightfield_sample(
        terrain, positions(G1_RightToe).x, positions(G1_RightToe).z);
    out.minimum_clearance = out.hips_clearance;
    const int probes[] = {
        G1_LeftKnee, G1_RightKnee, G1_LeftAnkle, G1_RightAnkle,
        G1_LeftToe, G1_RightToe
    };
    for (int i = 0; i < 6; ++i) {
        const vec3 p = positions(probes[i]);
        out.minimum_clearance = minf(
            out.minimum_clearance,
            p.y - heightfield_sample(terrain, p.x, p.z));
    }
    return out;
}

static float g1_xz_length(const vec3 value)
{
    return sqrtf(value.x * value.x + value.z * value.z);
}
~~~

The immutable `*_min_clearance` fields are explicitly a seven-joint Gate A
probe diagnostic: hips plus left/right knee, ankle, and toe. They are not a
claim of full-body or link-envelope clearance; the later IK/clearance plan adds
the authoritative collision envelope.

Replace the independent test-frame parsing in `main` with:

~~~cpp
g1_test_config test_config;
if (!g1_parse_test_config(
        test_config, artifact_error, (int)sizeof(artifact_error))) {
    fprintf(stderr, "G1 terrain option error: %s\n", artifact_error);
    return 2;
}
if (test_config.mode == G1_TestFlat ||
    test_config.mode == G1_TestSequential) {
    feature_weight_terrain = 0.0f;
} else if (test_config.mode == G1_TestTerrain &&
           getenv("MM_TERRAIN_WEIGHT") == NULL) {
    feature_weight_terrain = 4.0f;
}
~~~

Under `PLATFORM_WEB`, reject `test_config.frame_limit > 0` or any non-live
mode before opening a window. Outside `MM_DISCRETE`, also reject a non-null
`MM_LOG`. These paths require the bounded desktop loop to observe
`controller_exit_requested` and reach synchronous writer cleanup.

In an `MM_DISCRETE` build, reject every non-live test mode with a controlled
option error before opening a window. This preserves that build's legacy
`MM_LOG` text stream and `_Exit` behavior rather than opening the deterministic
CSV writer on the same path.

After the two stick reads, make only test-mode input deterministic:

~~~cpp
if (test_config.mode != G1_TestLive) {
    gamepadstick_left = vec3(0.0f, 0.0f, 0.9f);
    gamepadstick_right = vec3();
}
~~~

After `desired_strafe_update()`, force `desired_strafe = false`,
`desired_gait = 0`, and `desired_gait_velocity = 0.0f` only when
`test_config.mode != G1_TestLive`. Disable every behavior-mutating Raygui
control while deterministic test mode is active, restoring GUI state after the
control block; visualization may continue, but the mouse must not be able to
change weights, synchronization, adjustment, or clamping. Open the log before
entering the main loop. If learned motion matching is ever enabled, reject a
non-null deterministic log path: its generated poses have no truthful database
frame selection, and this CSV contract is deliberately database-frame-only.
Track `applied_feature_weight_terrain` separately from the GUI slider:
initialize it only after the initial feature build validates, update it only
after a GUI rebuild validates, and log that applied value as
`effective_terrain_weight`. Moving a live slider without rebuilding must never
relabel the active matcher.

The first prediction update must also initialize the malloc-backed prior-state
array before `trajectory_desired_rotations_predict` reads it. Immediately after
the first `desired_velocity = desired_velocity_curr`, while
`rendered_frames == 0`, call
`trajectory_desired_velocities.set(desired_velocity)`. This removes the existing
read-before-write without changing later prediction order: rotations continue
to use the preceding update's desired-velocity trajectory, and update zero uses
the current desired velocity as its only valid predecessor. This initialization
is deliberately unconditional: it corrects pre-existing undefined behavior in
live mode as well as deterministic tests; all later prediction ordering and
defined live matching behavior remain unchanged.

~~~cpp
motion_match_log deterministic_log;
#ifndef MM_DISCRETE
const char* deterministic_log_path = getenv("MM_LOG");
#else
const char* deterministic_log_path = NULL;
#endif
if (!deterministic_log.open(
        deterministic_log_path,
        artifact_error, (int)sizeof(artifact_error))) {
    fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
    UnloadModel(terrain_model);
    CloseWindow();
    return 2;
}
const bool logging_enabled = deterministic_log.file != NULL;
~~~

Before terrain snapshot/query construction, reject a non-queryable heightfield,
invalid/non-finite trajectory input, or non-finite snapshot with a controlled
runtime error and normal cleanup. Never let
`terrain_centerline_snapshot_compute` turn invalid state into four apparently
valid zero samples. Run `motion_match_query_is_finite_31d(query)` in every mode
before search. Only call the `snprintf`-based hex serializer when
`logging_enabled` is true.

At query construction, preserve the pre-search frame/range and initialize the
Runtime Task 6 cost fields:

~~~cpp
const int query_database_frame = frame_index;
const int query_range = g1_active_range(db, query_database_frame);
int selected_database_frame = query_database_frame;
const bool matching_enabled = test_config.mode != G1_TestSequential;
const bool search_requested = matching_enabled &&
    (force_search || search_timer <= 0.0f || end_of_anim);
float incumbent_cost = 0.0f;
float selected_cost = 0.0f;
float selected_terrain_error = 0.0f;
if (logging_enabled) {
    incumbent_cost = end_of_anim
        ? FLT_MAX : database_frame_cost(db, frame_index, query);
    selected_cost = incumbent_cost;
    selected_terrain_error = database_raw_terrain_error(
        db, frame_index, query);
}
bool transitioned = false;
~~~

Guard the existing search block with `if (search_requested)`. Immediately after
`database_search`, assign `selected_database_frame = best_index`,
and, only when logging, assign `selected_cost = best_cost` and
`selected_terrain_error = database_raw_terrain_error(db, best_index, query)`;
set `transitioned = true` only inside the existing `best_index != frame_index`
branch. Replace raw `frame_index++` with the range-safe advance:

~~~cpp
frame_index = database_trajectory_index_clamp(db, frame_index, 1);
~~~

Before entering deterministic sequential mode, prove that its requested update
count fits strictly before the initial animation range's terminal clamp. Reject
an overrun with exit code `2`; a repeated terminal frame must never be emitted
as a successful sequential test.

Immediately after `inertialize_pose_update`, capture the two pre-adjustment
stages without mutating the live pose, but only inside `if (logging_enabled)`:

~~~cpp
array1d<vec3> raw_selected_positions(curr_bone_positions);
array1d<quat> raw_selected_rotations(curr_bone_rotations);
raw_selected_positions(0) = bone_positions(0);
raw_selected_rotations(0) = bone_rotations(0);
const motion_match_pose_diagnostic raw_selected_diagnostic =
    g1_pose_diagnostic(
        raw_selected_positions, raw_selected_rotations,
        db.bone_parents, runtime_terrain);
const motion_match_pose_diagnostic inertialized_diagnostic =
    g1_pose_diagnostic(
        bone_positions, bone_rotations, db.bone_parents, runtime_terrain);
const vec3 root_before_adjustment = bone_positions(0);
~~~

After the adjustment block and before clamping, capture
`root_after_adjustment = bone_positions(0)` only when logging. After clamping,
perform the rendered diagnostic/FK and write one row inside the same logging
guard:

~~~cpp
const vec3 root_after_clamp = bone_positions(0);
const motion_match_pose_diagnostic rendered_diagnostic = g1_pose_diagnostic(
    bone_positions, bone_rotations, db.bone_parents, runtime_terrain);
array1d<vec3> rendered_global(db.nbones());
array1d<quat> rendered_global_rotations(db.nbones());
forward_kinematics_full(
    rendered_global, rendered_global_rotations,
    bone_positions, bone_rotations, db.bone_parents);
char query_bits_hex[31 * 8 + 1] = {};
const bool query_bits_ok = motion_match_query_bits_hex(
    query_bits_hex, sizeof(query_bits_hex), query);
if (!query_bits_ok) {
    fprintf(stderr, "G1 runtime query error: non-finite or non-31D query\n");
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}

motion_match_log_row log_row;
log_row.frame = rendered_frames;
log_row.fixed_dt = dt;
log_row.mode = test_config.name;
log_row.route = test_config.route;
log_row.query_bits_hex = query_bits_hex;
log_row.query_database_frame = query_database_frame;
log_row.query_range = query_range;
log_row.selected_database_frame = selected_database_frame;
log_row.database_frame = frame_index;
log_row.range = g1_active_range(db, frame_index);
log_row.source_range = g1_active_range(db, selected_database_frame);
log_row.searched = search_requested;
log_row.transitioned = transitioned;
log_row.incumbent_cost = incumbent_cost;
log_row.selected_cost = selected_cost;
log_row.selected_terrain_error = selected_terrain_error;
log_row.effective_terrain_weight = applied_feature_weight_terrain;
for (int i = 0; i < 4; ++i) {
    log_row.terrain[i] = terrain_query_snapshot.values[i];
    log_row.terrain_points[i] = terrain_query_snapshot.points[i];
}
log_row.raw_selected = raw_selected_diagnostic;
log_row.inertialized = inertialized_diagnostic;
log_row.rendered = rendered_diagnostic;
log_row.hips_inertial_offset_y =
    inertialized_diagnostic.hips_y - raw_selected_diagnostic.hips_y;
log_row.runtime_root_surface_height = heightfield_sample(
    runtime_terrain, bone_positions(0).x, bone_positions(0).z);
log_row.runtime_left_toe_surface_height = heightfield_sample(
    runtime_terrain, rendered_global(G1_LeftToe).x,
    rendered_global(G1_LeftToe).z);
log_row.runtime_right_toe_surface_height = heightfield_sample(
    runtime_terrain, rendered_global(G1_RightToe).x,
    rendered_global(G1_RightToe).z);
const vec3 adjustment_delta = root_after_adjustment - root_before_adjustment;
const vec3 clamp_delta = root_after_clamp - root_after_adjustment;
log_row.adjustment_xz = g1_xz_length(adjustment_delta);
log_row.adjustment_y = adjustment_delta.y;
log_row.clamp_xz = g1_xz_length(clamp_delta);
log_row.clamp_y = clamp_delta.y;
log_row.matching_enabled = matching_enabled;
log_row.adjustment_enabled = adjustment_enabled;
log_row.clamping_enabled = clamping_enabled;
log_row.support_retargeting_enabled = false;
log_row.ik_enabled = ik_enabled;
if (!deterministic_log.write(
        log_row, artifact_error, (int)sizeof(artifact_error))) {
    fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}
~~~

The invalid-query branch emits no row. A failed write invalidates the evidence,
emits no subsequent rows, and routes immediately through the same post-window
cleanup path with exit code `2`.

Use `test_config.frame_limit` for normal-loop termination. If
`WindowShouldClose()` ends a deterministic run before exactly that many updates,
report a controlled error and exit `2`. Call
`deterministic_log.close(
    artifact_error, (int)sizeof(artifact_error))` exactly once
on the common cleanup path; a flush or close failure also reports the logger
error and changes an otherwise-successful exit to `2`. With `MM_LOG` unset,
query hex serialization, pose diagnostics, their array allocations/FK passes,
and row construction do not run. The live path (`MM_TEST_MODE` unset) retains
keyboard input, search, 3D adjustment, clamping, and IK-off; the sole intentional
matching-path correction is the update-zero desired-velocity initialization
that removes the pre-existing read-before-write described above.

- [ ] **Step 7: Build and run the logger/checker test cycle**

Run:

~~~bash
set -euo pipefail
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_motion_match_log.cpp -o /tmp/test_motion_match_log_strict
rm -f /tmp/g1_writer_strict.csv
/tmp/test_motion_match_log_strict /tmp/g1_writer_strict.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_writer_strict.csv
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG -I. \
  tests/cpp/test_motion_match_log.cpp -o /tmp/test_motion_match_log_fast
rm -f /tmp/g1_writer_fast.csv
/tmp/test_motion_match_log_fast /tmp/g1_writer_fast.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_writer_fast.csv
cmp /tmp/g1_writer_strict.csv /tmp/g1_writer_fast.csv
g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
  -Wno-error=sign-compare -Wno-error=pedantic \
  -Wno-error=unused-parameter \
  -Wno-error=missing-field-initializers \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_gate_a_warning \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 \
  2> /tmp/g1_task1_controller_warnings.txt
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import re
from pathlib import Path
text = Path("/tmp/g1_task1_controller_warnings.txt").read_text()
categories = set(re.findall(r"\[-W([^]]+)\]", text))
expected = {
    "sign-compare", "pedantic", "unused-parameter",
    "missing-field-initializers",
}
assert categories == expected, (sorted(expected - categories),
                                sorted(categories - expected))
assert "motion_match_log.h:" not in text
print("VALID controller warning baseline", sorted(categories))
PY

g++ -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_gate_a \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

rm -f /tmp/g1_task1_smoke_a.csv /tmp/g1_task1_smoke_b.csv
for output in /tmp/g1_task1_smoke_a.csv /tmp/g1_task1_smoke_b.csv; do
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=25 \
    MM_LOG="$output" /tmp/controller_g1_gate_a
  test "$(wc -l < "$output")" -eq 26
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$output"
done
cmp /tmp/g1_task1_smoke_a.csv /tmp/g1_task1_smoke_b.csv

DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=2 \
  /tmp/controller_g1_gate_a

set +e
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=2 \
  MM_LOG=/dev/full /tmp/controller_g1_gate_a \
  >/tmp/g1_task1_dev_full.stdout 2>/tmp/g1_task1_dev_full.stderr
status=$?
set -e
test "$status" -eq 2
grep -q 'G1 runtime log error:' /tmp/g1_task1_dev_full.stderr

set +e
env -u MM_TEST_FRAMES DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain /tmp/controller_g1_gate_a \
  >/tmp/g1_task1_missing_limit.stdout \
  2>/tmp/g1_task1_missing_limit.stderr
missing_limit_status=$?
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=sequential MM_TEST_FRAMES=1000000000 \
  /tmp/controller_g1_gate_a \
  >/tmp/g1_task1_overrun.stdout 2>/tmp/g1_task1_overrun.stderr
overrun_status=$?
set -e
test "$missing_limit_status" -eq 2
test "$overrun_status" -eq 2
grep -q 'MM_TEST_FRAMES must be set to a positive integer' \
  /tmp/g1_task1_missing_limit.stderr
grep -q 'G1 sequential test overrun' /tmp/g1_task1_overrun.stderr

g++ -O3 -ffast-math -march=native -DNDEBUG -DMM_DISCRETE \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_gate_a_discrete \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
rm -f /tmp/g1_task1_discrete_conflict.log
set +e
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TEST_FRAMES=2 \
  MM_LOG=/tmp/g1_task1_discrete_conflict.log \
  /tmp/controller_g1_gate_a_discrete \
  >/tmp/g1_task1_discrete.stdout 2>/tmp/g1_task1_discrete.stderr
discrete_status=$?
set -e
test "$discrete_status" -eq 2
test ! -e /tmp/g1_task1_discrete_conflict.log
grep -q 'MM_TEST_MODE is unavailable in MM_DISCRETE builds' \
  /tmp/g1_task1_discrete.stderr
~~~

Expected: all Python tests pass; both standalone-writer builds produce a
checker-valid, byte-identical row; and the strict writer is warning-clean under
`-Werror -pedantic` while the production-flags writer still rejects Inf/NaN.
The controller warning-audit build exits `0` with
only Holden's pre-existing `sign-compare`, GNU compound-literal `pedantic`,
unused-parameter, and aggregate-initializer categories; no diagnostic names the
new logger header. The exact production command also builds and is the binary
used for all runtime evidence. Then both 25-update controller runs exit 0,
contain exactly 26 lines, pass the base checker, and are byte-identical; a
two-update no-log run exits normally; and `/dev/full` exits 2 through normal
cleanup with one logger diagnostic. Missing limits, sequential overrun, and a
non-live `MM_DISCRETE` conflict each fail closed with exit 2 before creating a
conflicting log. Do not run a terrain-sampling rebuild in this task and do not
stop or replace any pre-existing live visualizer process.

- [ ] **Step 8: Commit only the baseline diagnostic boundary**

~~~bash
git add motion_match_log.h resources/check_g1_runtime_log.py \
  tests/python/test_runtime_log.py tests/cpp/test_motion_match_log.cpp \
  controller.cpp
git diff --cached --check
git commit -m "test: capture G1 terrain penetration stages"
~~~

Expected: the commit contains only the five named files; existing modified
`resources/database.bin`, `resources/features.bin`, executables, logs, and media
remain unstaged.

### Task 2: Reproduce and freeze Gate A before changing surface behavior

**Files:**
- Generated, never committed: `/tmp/g1_gate_a_before.csv`
- Generated, never committed: `/tmp/g1_gate_a_csv.sha256`
- Generated, never committed: `/tmp/g1_gate_a_before.sha256`
- Generated, never committed: `/tmp/g1_gate_a_commit.txt`
- Generated, never committed: `/tmp/g1_gate_a_report.txt`
- Generated, never committed: `/tmp/g1_gate_a_status_before.txt`
- Generated, never committed: `/tmp/g1_gate_a_status_after.txt`
- Generated, never committed: `/tmp/g1_gate_a_evidence.sha256`

**Interfaces:**
- Consumes the pre-existing G1HF/v1 `resources/g1_terrain/` pack and the exact Task 1 controller.
- Produces immutable local evidence for comparing later support/runtime behavior; it does not alter or republish artifacts.

- [ ] **Step 1: Record the exact baseline revision and artifact hashes**

Run:

~~~bash
set -euo pipefail
for path in /tmp/g1_gate_a_before.csv /tmp/g1_gate_a_before.sha256 \
  /tmp/g1_gate_a_csv.sha256 \
  /tmp/g1_gate_a_commit.txt /tmp/g1_gate_a_report.txt \
  /tmp/g1_gate_a_status_before.txt /tmp/g1_gate_a_status_after.txt \
  /tmp/g1_gate_a_evidence.sha256
do
  test ! -e "$path"
done
git rev-parse HEAD > /tmp/g1_gate_a_commit.txt
{
  printf 'HEAD '
  cat /tmp/g1_gate_a_commit.txt
  printf 'STATUS\n'
  git status --short
  printf 'TRACKED_DIFF_SHA256 '
  git diff --binary --no-ext-diff HEAD | sha256sum
} > /tmp/g1_gate_a_status_before.txt
sha256sum /tmp/controller_g1_gate_a \
  resources/g1_terrain/database.bin \
  resources/g1_terrain/terrain_features.bin \
  resources/g1_terrain/terrain.bin \
  resources/g1_terrain/terrain.obj \
  resources/g1_terrain/manifest.json \
  resources/g1_terrain/validation.json \
  > /tmp/g1_gate_a_before.sha256
~~~

Expected: every evidence path was absent, all commands exit 0, and the exact
pre-task HEAD, dirty-status shape, tracked binary diff hash, producer-executable
hash, and six input-artifact hashes are saved before any run. Refuse to
overwrite prior evidence; choose a separately reviewed evidence basename if a
rerun is needed.

- [ ] **Step 2: Run the deterministic terrain-weight-four curb approach**

Run:

~~~bash
set -euo pipefail
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=375 \
  MM_LOG=/tmp/g1_gate_a_before.csv \
  /tmp/controller_g1_gate_a
test "$(wc -l < /tmp/g1_gate_a_before.csv)" -eq 376
sha256sum --check /tmp/g1_gate_a_before.sha256
sha256sum /tmp/g1_gate_a_before.csv > /tmp/g1_gate_a_csv.sha256
chmod a-w /tmp/g1_gate_a_before.csv /tmp/g1_gate_a_csv.sha256
sha256sum --check /tmp/g1_gate_a_csv.sha256
~~~

Expected: normal cleanup and exit 0 after exactly 375 updates; the CSV contains
376 lines including its header. The existing `resources/g1_terrain/` bytes are
unchanged from `/tmp/g1_gate_a_before.sha256`.

- [ ] **Step 3: Run the Gate A classifier and retain the observed answer**

Run:

~~~bash
set -euo pipefail
sha256sum --check /tmp/g1_gate_a_csv.sha256
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_gate_a_before.csv --gate-a \
  | tee /tmp/g1_gate_a_report.txt
test "$(grep -c '^VALID gate-a ' /tmp/g1_gate_a_report.txt)" -eq 1
grep -Eq 'classification=(raw-selected|blended-rendered|no-penetration)' \
  /tmp/g1_gate_a_report.txt
sha256sum --check /tmp/g1_gate_a_csv.sha256
~~~

Expected: exit 0 with a `VALID gate-a` line containing a finite
`first_positive`, selected `range`, `hips_offset_y`, `adjust_y`, and `clamp_y`,
and exactly one evidence classification: `raw-selected`, `blended-rendered`, or
`no-penetration`. The classification is intentionally measured rather than
preselected; if the run never produces a positive query, this step fails and
the artifact work must not begin.

- [ ] **Step 4: Verify this evidence-only task changed no tracked file**

Run:

~~~bash
set -euo pipefail
sha256sum --check /tmp/g1_gate_a_before.sha256
test "$(git rev-parse HEAD)" = "$(cat /tmp/g1_gate_a_commit.txt)"
{
  printf 'HEAD '
  git rev-parse HEAD
  printf 'STATUS\n'
  git status --short
  printf 'TRACKED_DIFF_SHA256 '
  git diff --binary --no-ext-diff HEAD | sha256sum
} > /tmp/g1_gate_a_status_after.txt
diff -u /tmp/g1_gate_a_status_before.txt /tmp/g1_gate_a_status_after.txt
sha256sum /tmp/controller_g1_gate_a \
  /tmp/g1_gate_a_before.csv \
  /tmp/g1_gate_a_before.sha256 \
  /tmp/g1_gate_a_csv.sha256 \
  /tmp/g1_gate_a_commit.txt \
  /tmp/g1_gate_a_report.txt \
  /tmp/g1_gate_a_status_before.txt \
  /tmp/g1_gate_a_status_after.txt \
  > /tmp/g1_gate_a_evidence.sha256
sha256sum --check /tmp/g1_gate_a_evidence.sha256
chmod a-w /tmp/controller_g1_gate_a \
  /tmp/g1_gate_a_before.csv \
  /tmp/g1_gate_a_before.sha256 \
  /tmp/g1_gate_a_csv.sha256 \
  /tmp/g1_gate_a_commit.txt \
  /tmp/g1_gate_a_report.txt \
  /tmp/g1_gate_a_status_before.txt \
  /tmp/g1_gate_a_status_after.txt \
  /tmp/g1_gate_a_evidence.sha256
for path in /tmp/controller_g1_gate_a \
  /tmp/g1_gate_a_before.csv \
  /tmp/g1_gate_a_before.sha256 \
  /tmp/g1_gate_a_csv.sha256 \
  /tmp/g1_gate_a_commit.txt \
  /tmp/g1_gate_a_report.txt \
  /tmp/g1_gate_a_status_before.txt \
  /tmp/g1_gate_a_status_after.txt \
  /tmp/g1_gate_a_evidence.sha256
do
  case "$(stat -c '%A' "$path")" in
    *w*) exit 1 ;;
  esac
done
sha256sum --check /tmp/g1_gate_a_evidence.sha256
~~~

Expected: every checksum reports `OK`; repository status contains only the
exact saved pre-task changes and no Gate A output; HEAD and the byte content of
every tracked dirty path are unchanged. The final manifest binds the producer
executable, CSV, report, commit record, input-hash record, and before/after
repository-state records, and every evidence file is made read-only. The report
persistently records the measured classification, and there is no commit for
this evidence-only task.

### Task 3: Replace radius dilation with exact vertical-triangle GRAIL queries

**Files:**
- Modify: `resources/g1_terrain_builder/terrain.py:1-303`
- Modify: `tests/python/test_terrain.py:1-314`

**Interfaces:**
- Produces: `triangulate_faces(vertices, face_counts, face_indices) -> ndarray[int32]` with deterministic fan triangulation.
- Produces: `VerticalTriangleSurface.height(x: float, z: float) -> float`, choosing the highest exact vertical intersection and returning `exterior_height` outside projected triangles.
- Produces: `GrailTerrain.from_base(base: str) -> GrailTerrain` using transformed USD vertices directly, with no dense point cloud, KD-tree, or query radius.
- Produces: `surface_semantics() -> dict` and `surface_semantics_signature() -> str`.

- [ ] **Step 1: Replace dilation tests with exact vertical-intersection tests**

In `tests/python/test_terrain.py`, remove `_densify_faces` from the import list
and replace the three densification/radius-constructor tests with:

~~~python
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    StepTerrain,
    VerticalTriangleSurface,
    build_facing_centerline,
    export_heightfield,
    sample_terrain_features,
    surface_semantics,
    surface_semantics_signature,
    triangulate_faces,
)

    def test_vertical_triangle_query_interpolates_exact_top(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
        ])
        surface = VerticalTriangleSurface(
            vertices, np.array([[0, 1, 2]], np.int32), exterior_height=-2.0)
        self.assertAlmostEqual(surface.height(0.25, 0.25), 0.375, places=12)
        self.assertEqual(surface.height(0.75, 0.75), -2.0)

    def test_vertical_query_uses_highest_overlapping_triangle(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 0, 1],
            [0, 0.4, 0], [1, 0.4, 0], [0, 0.4, 1],
        ], np.float64)
        surface = VerticalTriangleSurface(
            vertices, np.array([[0, 1, 2], [3, 4, 5]], np.int32))
        self.assertAlmostEqual(surface.height(0.2, 0.2), 0.4, places=12)

    def test_exact_query_does_not_dilate_a_top_by_fourteen_centimetres(self):
        vertices = np.array([
            [0.0, 0.2, 0.0], [1.0, 0.2, 0.0],
            [1.0, 0.2, 1.0], [0.0, 0.2, 1.0],
        ])
        triangles = triangulate_faces(
            vertices, np.array([4], np.int32),
            np.array([0, 1, 2, 3], np.int32))
        surface = VerticalTriangleSurface(vertices, triangles)
        self.assertEqual(surface.height(-0.001, 0.5), 0.0)
        self.assertEqual(surface.height(1.001, 0.5), 0.0)
        self.assertAlmostEqual(surface.height(0.001, 0.5), 0.2, places=12)

    def test_polygon_triangulation_is_a_stable_first_vertex_fan(self):
        vertices = np.zeros((5, 3), np.float64)
        triangles = triangulate_faces(
            vertices, np.array([5], np.int32),
            np.array([4, 2, 0, 1, 3], np.int32))
        np.testing.assert_array_equal(
            triangles, [[4, 2, 0], [4, 0, 1], [4, 1, 3]])

    def test_surface_signature_covers_every_query_semantic(self):
        semantics = surface_semantics()
        self.assertEqual(semantics, {
            "schema": "g1-terrain-surface/v1",
            "coordinate_signature":
                "holden-y-up-right-handed-forward-plus-z",
            "source_query": "vertical-triangle-top",
            "polygon_triangulation": "fan-from-first-index",
            "overlap_height_policy": "maximum-y",
            "projected_boundary_policy": "closed",
            "triangle_winding_policy": "orientation-independent",
            "degenerate_projected_triangle_policy": "ignore",
            "projected_area_measure": "absolute-two-times-area",
            "bbox_tolerance_m": 1e-12,
            "projected_area_epsilon_m2": 1e-12,
            "barycentric_tolerance": 1e-10,
            "heightfield_schema": "G1HF/v2",
            "heightfield_interpolation": "fixed-diagonal-triangles",
            "heightfield_diagonal":
                "min-x-min-z_to_max-x-max-z",
            "cell_size_m": 0.02,
            "exterior_height_m": 0.0,
        })
        self.assertRegex(surface_semantics_signature(), r"^[0-9a-f]{64}$")
~~~

Update synthetic `GrailTerrain(...)` construction in the remaining tests to
the exact three-argument form below; delete all `query_points` and `radius`
fixtures and assert the true transformed vertex bounds because dilation no
longer expands them:

~~~python
terrain = GrailTerrain(
    render_vertices=vertices,
    face_counts=face_counts,
    face_indices=face_indices,
)
self.assertEqual(terrain.xz_bounds(), (
    float(vertices[:, 0].min()), float(vertices[:, 0].max()),
    float(vertices[:, 2].min()), float(vertices[:, 2].max()),
))
~~~

Also restore negative and `index == len(vertices)` topology regressions and add
tests that reject float topology and `uint64` values outside int32 before any
cast. Lock both windings, closed edge/vertex ownership, projected-degenerate
rejection, maximum-height overlap selection, and defensive copies: mutating
caller arrays must not change queries, bounds, footprints, or diagnostic OBJ
bytes, and all retained geometry/topology arrays must be read-only.

- [ ] **Step 2: Run the focused tests and verify the old provider fails**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain.TerrainTests.test_vertical_triangle_query_interpolates_exact_top \
  tests.python.test_terrain.TerrainTests.test_exact_query_does_not_dilate_a_top_by_fourteen_centimetres \
  tests.python.test_terrain.TerrainTests.test_surface_signature_covers_every_query_semantic -v
~~~

Expected: `ERROR` because `VerticalTriangleSurface`, `triangulate_faces`, and
the surface-semantics functions do not exist.

- [ ] **Step 3: Implement deterministic triangulation and exact vertical top queries**

Remove `scipy.spatial.cKDTree`, `_densify_faces`, `_points`, `_tree`, and
`_radius` from `terrain.py`. Use `_checked_int32_topology` in the USD loader as
well as every public topology boundary so validation always precedes casting.
Add:

~~~python
import hashlib
import json

BBOX_TOLERANCE_M = 1e-12
PROJECTED_AREA_EPSILON_M2 = 1e-12
BARYCENTRIC_TOLERANCE = 1e-10

SURFACE_SEMANTICS = {
    "schema": "g1-terrain-surface/v1",
    "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
    "source_query": "vertical-triangle-top",
    "polygon_triangulation": "fan-from-first-index",
    "overlap_height_policy": "maximum-y",
    "projected_boundary_policy": "closed",
    "triangle_winding_policy": "orientation-independent",
    "degenerate_projected_triangle_policy": "ignore",
    "projected_area_measure": "absolute-two-times-area",
    "bbox_tolerance_m": BBOX_TOLERANCE_M,
    "projected_area_epsilon_m2": PROJECTED_AREA_EPSILON_M2,
    "barycentric_tolerance": BARYCENTRIC_TOLERANCE,
    "heightfield_schema": "G1HF/v2",
    "heightfield_interpolation": "fixed-diagonal-triangles",
    "heightfield_diagonal": "min-x-min-z_to_max-x-max-z",
    "cell_size_m": 0.02,
    "exterior_height_m": 0.0,
}


def surface_semantics() -> dict:
    return json.loads(json.dumps(SURFACE_SEMANTICS, sort_keys=True))


def surface_semantics_signature() -> str:
    payload = json.dumps(
        SURFACE_SEMANTICS, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checked_int32_topology(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if array.size:
        limits = np.iinfo(np.int32)
        if int(array.min()) < limits.min or int(array.max()) > limits.max:
            raise ValueError(f"{name} must fit int32")
    return np.array(array, np.int32, copy=True)


def triangulate_faces(
    vertices: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    counts = _checked_int32_topology(face_counts, "face counts")
    indices = _checked_int32_topology(face_indices, "face indices")
    _validate_mesh_topology(vertices, counts, indices)
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor:cursor + int(count)]
        cursor += int(count)
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, int(count) - 1)
        )
    return np.asarray(triangles, np.int32)


class VerticalTriangleSurface:
    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        exterior_height: float = 0.0,
    ):
        self.vertices = np.array(vertices, np.float64, copy=True)
        self.triangles = _checked_int32_topology(
            triangles, "surface triangle indices")
        if self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,) \
                or not len(self.vertices):
            raise ValueError("surface vertices must have non-empty shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,) \
                or not len(self.triangles):
            raise ValueError("surface triangles must have non-empty shape (M, 3)")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices)):
            raise ValueError("surface triangle index is outside the vertex array")
        if not np.isfinite(self.vertices).all() or not np.isfinite(exterior_height):
            raise ValueError("surface vertices and exterior height must be finite")
        self.exterior_height = float(exterior_height)
        self._triangle_vertices = self.vertices[self.triangles]
        projected = self._triangle_vertices[:, :, (0, 2)]
        self._minimum_xz = projected.min(axis=1)
        self._maximum_xz = projected.max(axis=1)
        for array in (
            self.vertices, self.triangles, self._triangle_vertices,
            self._minimum_xz, self._maximum_xz,
        ):
            array.setflags(write=False)

    def height(self, x: float, z: float) -> float:
        x, z = float(x), float(z)
        if not np.isfinite(x) or not np.isfinite(z):
            raise ValueError("terrain query coordinates must be finite")
        candidates = np.flatnonzero(
            (self._minimum_xz[:, 0] - BBOX_TOLERANCE_M <= x)
            & (x <= self._maximum_xz[:, 0] + BBOX_TOLERANCE_M)
            & (self._minimum_xz[:, 1] - BBOX_TOLERANCE_M <= z)
            & (z <= self._maximum_xz[:, 1] + BBOX_TOLERANCE_M)
        )
        if not len(candidates):
            return self.exterior_height
        triangle = self._triangle_vertices[candidates]
        a = triangle[:, 0]
        b = triangle[:, 1]
        c = triangle[:, 2]
        v0x, v0z = b[:, 0] - a[:, 0], b[:, 2] - a[:, 2]
        v1x, v1z = c[:, 0] - a[:, 0], c[:, 2] - a[:, 2]
        px, pz = x - a[:, 0], z - a[:, 2]
        determinant = v0x * v1z - v0z * v1x
        projected = np.abs(determinant) > PROJECTED_AREA_EPSILON_M2
        u = np.zeros_like(determinant)
        v = np.zeros_like(determinant)
        u[projected] = (
            px[projected] * v1z[projected]
            - pz[projected] * v1x[projected]
        ) / determinant[projected]
        v[projected] = (
            v0x[projected] * pz[projected]
            - v0z[projected] * px[projected]
        ) / determinant[projected]
        w = 1.0 - u - v
        inside = projected \
            & (u >= -BARYCENTRIC_TOLERANCE) \
            & (v >= -BARYCENTRIC_TOLERANCE) \
            & (w >= -BARYCENTRIC_TOLERANCE)
        if not np.any(inside):
            return self.exterior_height
        heights = w[inside] * a[inside, 1] \
            + u[inside] * b[inside, 1] \
            + v[inside] * c[inside, 1]
        return float(np.max(heights))
~~~

Replace `GrailTerrain` with a thin exact-source wrapper:

~~~python
class GrailTerrain(VerticalTriangleSurface):
    def __init__(self, render_vertices, face_counts, face_indices):
        self._vertices = np.array(render_vertices, np.float64, copy=True)
        self._face_counts = _checked_int32_topology(
            face_counts, "face counts")
        self._face_indices = _checked_int32_topology(
            face_indices, "face indices")
        _validate_mesh_topology(
            self._vertices, self._face_counts, self._face_indices)
        super().__init__(
            self._vertices,
            triangulate_faces(
                self._vertices, self._face_counts, self._face_indices),
            exterior_height=0.0,
        )
        self._max_height = float(self._vertices[:, 1].max())
        self._vertices.setflags(write=False)
        self._face_counts.setflags(write=False)
        self._face_indices.setflags(write=False)

    @classmethod
    def from_base(cls, base: str) -> "GrailTerrain":
        vertices, face_counts, face_indices = _load_usd_mesh(base)
        rotation, translation = _object_pose0(base)
        world_vertices = (rotation @ vertices.T).T + translation
        return cls(_mujoco_to_holden(world_vertices), face_counts, face_indices)

    def footprint(self) -> dict:
        top = self._vertices[
            self._vertices[:, 1] > self._max_height - 0.02]
        return {
            "x": (float(top[:, 0].min()), float(top[:, 0].max())),
            "z": (float(top[:, 2].min()), float(top[:, 2].max())),
            "height": self._max_height,
        }

    def xz_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self._vertices[:, 0].min()),
            float(self._vertices[:, 0].max()),
            float(self._vertices[:, 2].min()),
            float(self._vertices[:, 2].max()),
        )
~~~

Keep source OBJ export only as a diagnostic method; runtime scene OBJ emission
switches to the height grid in Task 4.

- [ ] **Step 4: Run all exact-source terrain tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import glob
import os
import numpy as np

from resources.g1_terrain_builder.terrain import (
    GrailTerrain, PROJECTED_AREA_EPSILON_M2, USD_DIR,
)

paths = sorted(glob.glob(os.path.join(USD_DIR, "*.usd")))
triangles = nondegenerate = degenerate = 0
for path in paths:
    terrain = GrailTerrain.from_base(os.path.splitext(os.path.basename(path))[0])
    values = terrain._triangle_vertices
    determinant = (
        (values[:, 1, 0] - values[:, 0, 0])
        * (values[:, 2, 2] - values[:, 0, 2])
        - (values[:, 1, 2] - values[:, 0, 2])
        * (values[:, 2, 0] - values[:, 0, 0])
    )
    triangles += len(values)
    for triangle, area in zip(values, np.abs(determinant)):
        if area <= PROJECTED_AREA_EPSILON_M2:
            degenerate += 1
            continue
        centroid = triangle.mean(axis=0)
        sampled = terrain.height(float(centroid[0]), float(centroid[2]))
        # Near-vertical projected faces amplify centroid arithmetic; the
        # provider must still resolve their plane within one micrometre.
        assert np.isfinite(sampled) and sampled >= float(centroid[1]) - 1e-6
        nondegenerate += 1
assert (len(paths), triangles, nondegenerate, degenerate) \
    == (1769, 71724, 71648, 76)
print("VALID strict-grail-surfaces=1769 triangles=71724")
PY
~~~

Expected: all terrain tests pass; the real default curb's top remains between
`0.1 m` and `0.5 m`, while an XZ query `0.001 m` outside a synthetic top returns
the exterior height rather than the old dilated maximum. The hardened suite is
22 tests, and an independent full-corpus scan must load all 1,769 meshes under
the strict topology rules and print exactly
`VALID strict-grail-surfaces=1769 triangles=71724`.

- [ ] **Step 5: Hold the source-surface change for the v2 grid boundary**

~~~bash
git diff --check -- \
  resources/g1_terrain_builder/terrain.py tests/python/test_terrain.py
~~~

Expected: the two-file change is review-clean but remains uncommitted. Its
locked surface semantics already name the G1HF/v2 scene contract. Task 4 adds
that v2 grid capability while retaining the old top-level v1 exporter for
migration safety; both exact-source and v2-grid halves are committed together.

### Task 4: Add an authoritative G1HF/v2 grid and grid-derived OBJ

**Files:**
- Modify: `resources/g1_terrain_builder/terrain.py`
- Modify: `tests/python/test_terrain.py`

**Migration boundary:**
- Produces: `HeightGrid.height(x, z) -> float` and
  `HeightGrid.normal(x, z) -> ndarray shape (3,)` from one fixed diagonal.
- Produces: `HeightGrid.g1hf_bytes() -> bytes`,
  `HeightGrid.obj_bytes() -> bytes`, and `HeightGrid.metadata() -> dict`.
- Produces: `rasterize_heightfield(terrain, bounds, cell_size=0.02) -> HeightGrid`.
- Produces: `export_heightfield_obj(grid, path) -> None`.
- Preserves the existing `export_heightfield(...)` G1HF/v1 writer, its old
  six-key metadata, the v1 builder/validator, and the published pack until
  Task 11 atomically replaces the top-level terrain with v2 scene directories.
- Keeps `GrailTerrain.export_obj` only as a diagnostic source-mesh export.
  Runtime v2 scene code uses `HeightGrid.obj_bytes()` exclusively.

- [ ] **Step 1: Write the failing v2 precision, ownership, and byte tests**

Import `HeightGrid`, `rasterize_heightfield`, and
`export_heightfield_obj` in `tests/python/test_terrain.py`. Add tests that lock
all of these contracts before production code:

1. A `2 x 2` asymmetric grid checks both fixed-diagonal interpolation branches,
   `tx == tz` ownership, both upward normals, and exterior height/normal.
2. A multi-cell grid checks every physical node at
   `origin + index * cell`, an interior X/Z grid-line tie, the final node, and
   all four inclusive world-space edges.
3. For each edge, `np.nextafter` immediately inward remains inside and
   immediately outward is exterior. Use awkward, non-float32-exact origins and
   cell sizes so normalized-coordinate rounding cannot hide an upper-edge bug.
4. Decode `g1hf_bytes()` with `<4sIII4f` and require the grid properties,
   metadata, query coordinates, and OBJ coordinates to derive from those exact
   decoded float32 header values. The payload is exact C-order little-endian
   float32 with no trailing bytes.
5. Mutating the caller's height array after construction cannot change query,
   binary, OBJ, or metadata bytes. `grid.heights` is owned, C-contiguous,
   little-endian, and read-only; an attempted write raises.
6. An asymmetric `3 x 2` or `2 x 3` OBJ locks Z-major/X-minor vertex order,
   two upward-wound faces per cell (`p00 p11 p10`, then `p00 p01 p11`),
   cell order, `.9g` formatting, and the final newline. Generate an independent
   expected OBJ from the decoded G1HF header to prove both files describe the
   same serialized nodes. Parse awkward real-grid X/Z tokens back to binary32
   and require bit equality with an explicit binary32 round of each promoted-
   double node; decimal closeness is insufficient.
7. Reject non-2D/smaller-than-`2 x 2` grids, non-finite/float32-overflowing
   samples or metadata, complex samples before any lossy cast, cell sizes that
   encode to float32 zero, dimensions or sample counts above the C++ consumer's
   `INT_MAX` limit, and X/Z node sequences that collapse or become non-finite
   after binary32 runtime rounding. Reject subnormal header/payload/node values
   and negative-zero serialization; preserve positive zero and a minimum-normal
   cell when its nodes remain distinguishable. Rasterization must reject
   impossible/collapsed sizes before allocation or any terrain query.
8. Rasterization rounds each minimum origin toward negative infinity in
   float32 when needed, uses the encoded positive float32 cell, and ceil-covers
   each requested maximum. Test both requested minima and maxima are covered.
9. `export_heightfield_obj` validates/serializes before opening the target and
   emits exactly `grid.obj_bytes()`.
10. Rename the existing binary contract test to state explicitly that the
    legacy `export_heightfield` remains G1HF/v1 until Task 11; do not change its
    bytes or metadata in this task.
11. Lock the complete `surface_semantics()` object, including source-node,
    runtime-query, cross-language parity-domain, runtime-node-distinguishability,
    and OBJ-coordinate-quantization policy strings; require its signature to
    change with those fields.
12. Prove the O(1) axis helper with the interior-collapse and crossing-zero
    near-cancellation regressions, a huge mocked dimension with a constant call
    count, and seeded small-fixture property tests. Brute-force every rounded
    node for each helper-accepted small fixture and require normal-or-positive-
    zero, finite, strictly increasing output; conservative rejection is allowed,
    false acceptance is not.

- [ ] **Step 2: Capture the legacy green baseline, then the missing-grid red state**

Before adding the new module-level imports, rename and run only:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain.TerrainTests.test_legacy_export_heightfield_remains_g1hf_v1 -v
~~~

Expected: `OK`. Then add the imports/tests and run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain.TerrainTests.test_height_grid_uses_one_fixed_diagonal_for_height_and_normal \
  tests.python.test_terrain.TerrainTests.test_height_grid_binary_header_is_float32_coordinate_authority \
  tests.python.test_terrain.TerrainTests.test_height_grid_owns_read_only_little_endian_height_bytes -v
~~~

Expected: module import fails because `HeightGrid` and the v2 helpers do not yet
exist. The separately captured legacy-v1 baseline remains the migration oracle.

- [ ] **Step 3: Implement one immutable, serialized-float32 authority**

Add shared constants before `SURFACE_SEMANTICS` and reuse them everywhere:

~~~python
HEIGHTFIELD_HEADER = struct.Struct("<4sIII4f")
HEIGHTFIELD_VERSION = 2
HEIGHTFIELD_DIAGONAL = "min-x-min-z_to_max-x-max-z"
HEIGHTFIELD_INTERPOLATION = "fixed-diagonal-triangles"
HEIGHTFIELD_SCALAR_ENCODING = "ieee754-binary32-little-endian"
HEIGHTFIELD_DOMAIN_POLICY = "inclusive-authoritative-node-rectangle"
HEIGHTFIELD_GRID_LINE_POLICY = "positive-index-cell-except-maximum-edge"
HEIGHTFIELD_DIAGONAL_TIE_POLICY = "tx-greater-or-equal-tz-uses-p00-p10-p11"
HEIGHTFIELD_SOURCE_NODE_ENCODING = \
    "binary32-header-values-promoted-to-binary64-arithmetic"
HEIGHTFIELD_RUNTIME_QUERY_ENCODING = \
    "normal-or-zero-binary32-canonicalized-positive-and-promoted-to-binary64"
HEIGHTFIELD_SCALAR_DOMAIN = "normal-or-zero-binary32"
HEIGHTFIELD_CELL_DOMAIN = "positive-normal-binary32"
HEIGHTFIELD_RUNTIME_NODE_DOMAIN = "normal-or-zero-binary32"
HEIGHTFIELD_RUNTIME_QUERY_DOMAIN = "normal-or-zero-binary32-coordinates"
HEIGHTFIELD_RUNTIME_PARITY_DOMAIN = "normal-or-zero-binary32-coordinates"
HEIGHTFIELD_DENORMAL_POLICY = "reject-nonzero-binary32-subnormals"
HEIGHTFIELD_EVALUATION_PRECISION = \
    "binary64-from-binary32-samples-and-promoted-node-weights"
HEIGHTFIELD_RUNTIME_HEIGHT_OUTPUT = \
    "finite-binary64-interpolation-rounded-to-binary32"
HEIGHTFIELD_NORMAL_EVALUATION = \
    "selected-triangle-binary64-gradient-scale-safe-unit-normalization"
HEIGHTFIELD_RUNTIME_NORMAL_OUTPUT = \
    "unit-normal-components-rounded-to-binary32"
HEIGHTFIELD_RUNTIME_OUTPUT_FTZ_POLICY = \
    "binary32-subnormals-and-signed-zero-canonicalized-to-positive-zero"
HEIGHTFIELD_ZERO_ENCODING = "canonical-positive-zero"
HEIGHTFIELD_RUNTIME_NODE_DISTINGUISHABILITY_POLICY = \
    "normal-or-positive-zero-strictly-increasing-proven-by-endpoints-" \
    "near-zero-candidates-max-binary32-spacing-and-aligned-equality"
HEIGHTFIELD_OBJ_COORDINATE_QUANTIZATION = \
    "binary32-round-of-promoted-origin-plus-index-times-cell"
HEIGHTFIELD_RASTER_BOUNDS_POLICY = \
    "float32-minimum-rounded-down-and-maximum-ceil-covered"
HEIGHTFIELD_OBJ_VERTEX_ORDER = "z-major-x-minor"
HEIGHTFIELD_OBJ_FACE_ORDER = "p00-p11-p10_then_p00-p01-p11"
HEIGHTFIELD_OBJ_FLOAT_FORMAT = ".9g-final-newline"
HEIGHTFIELD_EXTERIOR_NORMAL = (0.0, 1.0, 0.0)
HEIGHTFIELD_MAX_SAMPLES = np.iinfo(np.int32).max
HEIGHTFIELD_MIN_NORMAL = float(np.finfo(np.float32).tiny)
~~~

Extend `SURFACE_SEMANTICS` with those precision, inclusive-domain,
grid-line, diagonal-tie, exterior-normal, raster-bounds, and OBJ byte policies.
The signature must change in the same commit and the exact semantics test must
lock every new field. `cell_size_m: 0.02` remains the nominal scene setting;
each grid's metadata records the authoritative decoded float32 cell.

The exact added semantic keys and values are:

~~~python
"heightfield_version": 2,
"heightfield_scalar_encoding": HEIGHTFIELD_SCALAR_ENCODING,
"heightfield_domain_policy": HEIGHTFIELD_DOMAIN_POLICY,
"heightfield_grid_line_policy": HEIGHTFIELD_GRID_LINE_POLICY,
"heightfield_diagonal_tie_policy": HEIGHTFIELD_DIAGONAL_TIE_POLICY,
"heightfield_exterior_normal": HEIGHTFIELD_EXTERIOR_NORMAL,
"heightfield_source_node_encoding": HEIGHTFIELD_SOURCE_NODE_ENCODING,
"heightfield_runtime_query_encoding": HEIGHTFIELD_RUNTIME_QUERY_ENCODING,
"heightfield_scalar_domain": HEIGHTFIELD_SCALAR_DOMAIN,
"heightfield_cell_domain": HEIGHTFIELD_CELL_DOMAIN,
"heightfield_runtime_node_domain": HEIGHTFIELD_RUNTIME_NODE_DOMAIN,
"heightfield_runtime_query_domain": HEIGHTFIELD_RUNTIME_QUERY_DOMAIN,
"heightfield_runtime_parity_domain": HEIGHTFIELD_RUNTIME_PARITY_DOMAIN,
"heightfield_denormal_policy": HEIGHTFIELD_DENORMAL_POLICY,
"heightfield_evaluation_precision": HEIGHTFIELD_EVALUATION_PRECISION,
"heightfield_runtime_height_output": HEIGHTFIELD_RUNTIME_HEIGHT_OUTPUT,
"heightfield_normal_evaluation": HEIGHTFIELD_NORMAL_EVALUATION,
"heightfield_runtime_normal_output": HEIGHTFIELD_RUNTIME_NORMAL_OUTPUT,
"heightfield_runtime_output_ftz_policy":
    HEIGHTFIELD_RUNTIME_OUTPUT_FTZ_POLICY,
"heightfield_zero_encoding": HEIGHTFIELD_ZERO_ENCODING,
"heightfield_runtime_node_distinguishability_policy":
    HEIGHTFIELD_RUNTIME_NODE_DISTINGUISHABILITY_POLICY,
"heightfield_obj_coordinate_quantization":
    HEIGHTFIELD_OBJ_COORDINATE_QUANTIZATION,
"heightfield_raster_bounds_policy": HEIGHTFIELD_RASTER_BOUNDS_POLICY,
"heightfield_obj_vertex_order": HEIGHTFIELD_OBJ_VERTEX_ORDER,
"heightfield_obj_face_order": HEIGHTFIELD_OBJ_FACE_ORDER,
"heightfield_obj_float_format": HEIGHTFIELD_OBJ_FLOAT_FORMAT,
~~~

Implement `HeightGrid` as a frozen dataclass with `version` fixed internally to
integer `2` (not an init argument). In `__post_init__`, inspect the source
array's shape and enforce the dimension/sample-count limits before any owned
copy or allocation. Then:

- convert header scalars to float32 and classify their raw bits; reject
  overflow/non-finite/subnormal values, require a positive-normal cell, and
  canonicalize every accepted signed zero positive before storing decoded
  Python floats;
- before copying heights, use the named constant-time axis proof to require
  every X/Z node coordinate to round to normal-or-positive-zero binary32 and
  remain strictly increasing. Check endpoints, first/final adjacent pairs,
  clamped candidates nearest zero, maximum inward binary32 spacing, and the
  aligned-equality exception. Reject a collapsed/non-domain endpoint or
  interior step even if promoted-double endpoint coordinates are distinct;
- make an unconditional owned C-order `<f4` copy of heights, validate it, and
  reject non-finite/subnormal samples, canonicalize signed zeros positive, and
  mark it read-only;
- require `nx`, `nz`, and `nx * nz` to fit `INT_MAX` before byte production;
- expose `nx`, `nz`, `max_x`, and `max_z` from authoritative values.

The named O(1) proof is `_validate_runtime_axis(origin, count, cell, axis)` and
must be ported literally in Task 5:

1. Compute promoted-double and explicitly rounded/canonicalized binary32
   coordinates only at indices `0`, `1`, `count-2`, and `count-1`. Require all
   eight values finite, both adjacent source/runtime pairs strictly increasing,
   and all four runtime values normal-or-positive-zero.
2. If the promoted interval strictly crosses zero, compute
   `base = floor(-origin / cell)` and inspect the clamped indices in
   `[base-1, base+2]`; every rounded value must be normal-or-positive-zero.
3. For either binary64 source endpoints or binary32 runtime endpoints, define
   maximum inward spacing as `last - prev(last)` when `first >= 0`,
   `next(first) - first` when `last <= 0`, and the maximum of those two gaps
   when the interval crosses zero. Let `maximum_spacing` be the maximum of the
   source and runtime results.
4. Counts at most three are already completely covered by the adjacent checks.
   Otherwise accept only when `cell > maximum_spacing`, or when
   `cell == maximum_spacing` and the finite quotient `origin / cell` is an
   exact integer. Reject all other cases.

This is deliberately conservative: it may reject an unusual valid grid but may
never accept a collapsed or out-of-domain node sequence. No loop bound may
depend on `count`; the seeded property oracle brute-forces only small accepted
fixtures to prove zero false acceptance.

`_cell(x, z)` first rejects non-finite inputs, then compares against explicit
inclusive world-space `[origin, max]` bounds. Only accepted values are
normalized and clamped. Exact interior node coordinates belong to the
positive-index cell; the maximum edge belongs to the final cell. The diagonal
uses `tx >= tz` for the `p00/p10/p11` triangle. `normal` uses the identical
branch and returns exactly `[0, 1, 0]` outside.

`g1hf_bytes()` packs only already-validated authoritative values. `obj_bytes()`
computes every X/Z node with promoted authoritative header arithmetic, rounds
the result explicitly to binary32, then formats that rounded value with `.9g`.
Parsed OBJ X/Z coordinates must therefore have the identical binary32 bits as
the runtime/render node. It emits read-only float32 heights and uses the locked
order/format/winding. `metadata()` has exactly:

~~~text
schema, version, nx, nz, origin_x, origin_z, cell_size_m,
exterior_height_m, interpolation, diagonal
~~~

- [ ] **Step 4: Implement ceil-cover rasterization and safe OBJ writing**

`rasterize_heightfield` validates finite strict bounds and the float32-encoded
positive-normal cell. Quantize each requested minimum to float32 and, if round-to-
nearest moved it above the request, take one float32 `nextafter` toward negative
infinity. Compute dimensions from those authoritative values so the final node
ceil-covers the requested maximum; reject consumer-invalid dimensions/sample
counts and runtime-collapsed/non-domain binary32 node coordinates before
allocating or querying the source. Query the source at the exact promoted-
double authoritative grid nodes and let `HeightGrid` reject non-finite results.

Python grid queries accept finite binary64 coordinates over the exact promoted-
double node rectangle. C++ parity is intentionally defined only for normal-or-
zero binary32 query coordinates: canonicalize signed zero positive and promote
those values to binary64 before applying the same bounds, grid-line, and
diagonal policies; a subnormal runtime query is exterior/up. An exact Python
maximum with no accepted binary32 representation has no C++ query counterpart;
an accepted inward neighbor is inside and an outward neighbor beyond the exact
rectangle is exterior.

`export_heightfield_obj` requires a `HeightGrid`, computes `payload =
grid.obj_bytes()` before opening the path, then writes that payload once. Do not
modify the legacy v1 `export_heightfield`, the builder, validator, or published
resource directory in this task.

- [ ] **Step 5: Run focused, terrain, and full-repository gates**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
  resources/g1_terrain_builder/terrain.py tests/python/test_terrain.py
git diff --check -- \
  resources/g1_terrain_builder/terrain.py tests/python/test_terrain.py
~~~

Expected: every terrain and repository test passes; the old builder still
publishes/validates G1HF/v1, while the new in-memory v2 binary and its OBJ are
byte-deterministic and share authoritative serialized coordinates.

- [ ] **Step 6: Commit the exact-source and v2-grid capability together**

~~~bash
git add resources/g1_terrain_builder/terrain.py tests/python/test_terrain.py
git diff --cached --check
git commit -m "feat: add exact G1 terrain grids"
~~~

Expected: this is the first committed revision whose signed v2 surface
semantics are fully implemented. It deliberately does not claim the currently
published/top-level terrain is v2; Task 11 performs that atomic migration.

### Task 5: Add migration-safe C++ G1HF/v1-v2 query and normal parity

**Files:**
- Modify: `terrain_runtime.h`
- Modify: `tests/cpp/test_terrain_runtime.cpp`

**Interfaces:**
- `heightfield` gains an appended `uint32_t version` after `heights`, preserving
  every historical v1 member offset; successful loads set it to 1 or 2 and
  every failed load preserves the complete destination, including version.
- Preserves: `heightfield_load(heightfield&, const char*, char*, int) -> bool`.
- Preserves `heightfield_sample(...)` as the literal historical float/bilinear
  G1HF/v1 body. It deliberately does not dispatch or validate; v1 must not
  share the new locator because doing so changes production fast-math codegen.
- Adds checked `heightfield_sample_v2(...)` and checked
  `heightfield_sample_versioned(...)`. Their internal prevalidated v2 helper is
  explicitly named as such and is not a downstream public contract.
- Adds authoritative G1HF/v2 fixed-triangle sampling with float32-decoded
  normal-or-positive-zero metadata promoted to double, double local weights,
  and finite binary32 output rounded from the binary64 interpolation.
- Produces: `heightfield_normal(...) -> vec3` for v2 using the identical
  triangle and scale-safe double normalization. V1, exterior, non-finite, and
  structurally invalid inputs return exactly `(0, 1, 0)`.
- Cross-language parity is defined for normal-or-zero binary32 query X/Z
  values; query signed zero is canonicalized positive and subnormal queries
  return exterior/up. C++ promotes accepted exact values to double and compares
  them against the same double node rectangle as Python. An exact Python
  maximum that is not binary32-representable has no C++ query counterpart; the
  adjacent accepted inward float is inside and an outward-rounded float beyond
  the rectangle is exterior.

- [ ] **Step 1: Lock legacy v1 and write failing v2/parser/parity tests**

Keep existing manual fixtures v1 by default; change `initialize_heightfield` to
take an explicit version argument only where a v2 fixture is intended. Before
production edits, compile/run the existing suite and retain its output as the
v1 baseline.

Convert every runtime `assert(...)` in the test harness to the existing
always-on `check(...)` mechanism, including fixture I/O, loader calls, error
checks, transactions, and samples. Retain `static_assert` only. Add these tests:

1. Load otherwise-identical v1 and v2 asymmetric `2 x 2` grids. V1 remains
   bilinear (`0.1875`); v2 returns the fixed-triangle value (`0.25`).
2. A v1 coordinate-arithmetic regression uses:

~~~text
origin=-36257.83203125f, cell=0.04736527055501938f,
x=-32593.607421875f
~~~

   and requires the current v1 result/float bits. This prevents routing v1
   through the double v2 locator.
3. Generate one asymmetric awkward-metadata `HeightGrid.g1hf_bytes()` oracle
   with Python, paste its exact bytes into a C++ fixture, and compare the entire
   independent C++ byte construction before loading it. Lock decoded header and
   payload bits, nodes, both triangle branches, diagonal equality, normals, and
   exterior results against the Python oracle.
4. A diagonal downcast regression uses:

~~~text
ox=7.857595920562744f, oz=-0.7662742137908936f,
cell=21.012887954711914f,
x=22.33635711669922f, z=13.71248722076416f
~~~

   where double `tx < tz` but float downcast ties; v2 must select the second
   triangle.
5. Test all exact representable edges, `nextafterf` inward/outward probes,
   awkward outward-rounded maxima, interior X/Z grid-line ownership, maximum-
   edge final-cell ownership, and `tx == tz` first-triangle ownership.
6. Require a flat v2 normal to be bit-exact `(0,1,0)` and cover both sloped
   triangles. Add extreme normal heights (`-FLT_MAX`/`+FLT_MAX`), minimum-normal
   and very large positive cells whose node coordinates remain runtime-distinct,
   proving samples/normals stay finite. Reject v2 subnormal header scalars,
   payload heights, rounded nodes, and query coordinates; subnormal queries
   return exterior/up without indexing. Reject serialized negative zero. Add
   collapsed-coordinate v2 headers (large origin with too-small cell, an
   overflowing final node, and first/final-adjacent-distinct but interior
   collapse) transactionally. Matching legacy-v1 fixtures retain their exact
   historical behavior.
7. Run every truncation, trailing-byte, dimension-overflow, metadata, and
   non-finite-payload rejection for both versions. Unknown version 3 is rejected
   transactionally. `expect_heightfield_rejected` and the no-mutation test
   include a sentinel version.
8. Structurally invalid or unknown-version in-memory fields return exterior/up
   without indexing storage.
9. Port the exact Python O(1) axis vectors to C++, including the alignment-
   equality acceptance. A deterministic small-grid property test brute-forces
   every rounded node for each helper-accepted fixture and proves finite,
   normal-or-zero, strictly increasing nodes with zero false acceptance. The
   validator call count is constant and never loops over header dimensions.
10. For the independent Python oracle, cast query X/Z to binary32 before the
    Python call, then round Python height/normal results to binary32 and
    canonicalize any subnormal or signed zero to positive zero. Compare C++
    height and every normal component bit-for-bit with that runtime oracle.

Change the generated-artifact probe CLI from two path arguments to three exact
arguments: `G1TF_PATH G1HF_PATH EXPECTED_G1HF_VERSION`. Accept only decimal
version text `1` or `2`, pass that value into the probe, and require
`field.version == expected_version`. The currently published top-level artifact
is intentionally v1 until Task 11, so Task 5 passes `1`; Task 13 passes `2` for
each scene. In both cases require non-empty feature rows, `nx,nz >= 2`, exact
storage size, and finite samples; remove hard-coded full-pack frame/grid sizes.
The always-on main contract is `argc == 1 || argc == 4`; in the latter case use
exact `strcmp(argv[3], "1")` / `strcmp(argv[3], "2")` branches (no permissive
`atoi` prefix parsing), then call
`probe_generated_artifacts(argv[1], argv[2], expected_version)`.

- [ ] **Step 2: Compile and capture the real v2 red state**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_red
~~~

Expected: compilation fails because `heightfield_normal` and
`heightfield.version` do not exist. If those declarations are added alone, the
new v2 load assertion still fails because the current loader accepts only v1.

- [ ] **Step 3: Extend the strict transactional loader**

Add the version field:

~~~cpp
struct heightfield
{
    int nx = 0;
    int nz = 0;
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    float cell_size = 0.0f;
    float exterior_height = 0.0f;
    array1d<float> heights;
    // Append only: legacy v1 optimizer-visible member offsets are immutable.
    uint32_t version = 0;
};
~~~

Replace the exact `version != 1` branch in `heightfield_load` with:

~~~cpp
if (version != 1 && version != 2) {
    fclose(file);
    return terrain_error(
        error, error_capacity,
        "%s: unsupported G1HF version %u (expected 1 or 2)",
        path, static_cast<unsigned>(version));
}
~~~

Keep the existing dimensions, overflow, exact-length, metadata-finite, payload
read, payload-finite, and `terrain_finish_read` branches for both versions. In
the v2 branch only, classify raw float bits: header/payload values and derived
rounded nodes must be normal or positive zero, while cell must be positive
normal; reject subnormals and negative-zero encodings transactionally. Run the
complete common corruption matrix for both versions plus the v2-only domain
matrix. Assign `loaded.version = version` only in the temporary object and swap
it with the other fields after all reads and validation succeed.

~~~cpp
loaded.version = version;
~~~

For version 2 only, validate before payload allocation that the promoted-double
first/final X/Z nodes are finite and that every rounded node is normal-or-zero
and remains strictly increasing. Port Python's named O(1) helper and its exact
maximum-inward-spacing/aligned-equality rule literally; never loop over
dimensions supplied by an untrusted header. Version 1 must not gain this
rejection, because its parser/arithmetic contract remains historical.

Production `-ffast-math` flushes a hardware double-to-float subnormal result,
so the v2 port must classify each promoted-double node *before* casting. For
`magnitude = abs(source_node)`, IEEE round-to-nearest-even produces positive
zero when `magnitude <= 0x1p-150`, a forbidden nonzero binary32 subnormal when
`0x1p-150 < magnitude < (0x1p-126 - 0x1p-150)`, and a normal value at or above
the upper tie (which rounds to minimum normal). Canonicalize the first case to
`+0.0f`, reject the middle interval, and only then cast the normal case. Add
exact lower/upper-tie and `nextafter` tests under the release flags; an ordinary
unchecked `static_cast<float>` is not an acceptable domain check.

Update `terrain_heightfield_is_queryable` to require `version == 1 ||
version == 2` in addition to its current checks. Never use standard
`isfinite` as a fast-math safety guard; retain the existing bitwise finite
predicate for float inputs and decoded metadata.

- [ ] **Step 4: Preserve the public v1 body literally and add checked v2 APIs**

Keep the current public `heightfield_sample` body at its existing call sites
without changing its float operations, comparison order, edge degeneracy, or
adding a version/validity branch. A unified dispatcher was empirically proven
to alter GCC 13.3 `-O3 -ffast-math` inlining/FMA decisions and the Gate A CSV.
If a legacy helper name is useful, it may be a trivial wrapper around that
literal public body; the controller and centerline remain on the literal v1
entry until the later multiscene plan switches them explicitly.

Add a checked `heightfield_sample_v2` that requires version 2 and a queryable
structure before delegating to an explicitly named prevalidated internal
helper. Add `heightfield_sample_versioned` as the checked artifact/test entry:
it validates once, routes version 1 to the literal legacy sampler, routes
version 2 to the prevalidated v2 helper, and returns exterior for every
unknown/malformed field. Direct malformed-structure tests must cover both
checked public entries and prove neither indexes invalid storage.

~~~cpp
struct heightfield_cell
{
    int x0, z0;
    double tx, tz;
};
~~~

The v2 locator first bitwise-rejects non-finite or subnormal query inputs and
canonicalizes either signed zero to positive zero, then promotes accepted float
metadata/query values to double. It computes explicit double minima/maxima first
and rejects outside queries before normalization. For each accepted axis, clamp
the normalized coordinate, then mirror Python's boundary comparisons so an exactly
representable interior node owns the positive-index cell and the maximum owns
the final cell with fraction 1. Keep both fractions double through `tx >= tz`.
As with interpolation, use named volatile-double products/sums/differences for
node bounds, normalized coordinates, and local fractions so `-ffast-math`
cannot fuse or reassociate the signed Python operation sequence.

Convert all four finite heights to double before subtraction. Evaluate the
fixed-triangle convex formulas in double, cast the finite result to float, and
canonicalize a subnormal or signed-zero result positive. Compare the exact
output bits with the same FTZ-canonicalized Python oracle in every build
configuration. Because production uses `-ffast-math`, materialize each
subtraction, multiplication, and ordered addition in a separately named
`volatile double` intermediate; do not permit contraction or reassociation to
change the signed evaluation order.
For normals, compute the chosen triangle's double gradients, then perform
scale-safe explicit normalization (scale by the largest absolute component
before squaring, and materialize the three ordered squares/additions and square
root input with the same volatile-double discipline), round each component to float, and canonicalize zero
or subnormal components positive. Do not call Holden `normalize`, which divides by
`length + 1e-8` and changes a flat nominal-cell normal.

- [ ] **Step 5: Run standard, strict, production-release, and sanitizer gates**

Run:

~~~bash
! rg -n '(^|[^_[:alnum:]])assert[[:space:]]*\(' \
  tests/cpp/test_terrain_runtime.cpp
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime
/tmp/test_terrain_runtime \
  resources/g1_terrain/terrain_features.bin resources/g1_terrain/terrain.bin 1
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_strict
/tmp/test_terrain_runtime_strict \
  resources/g1_terrain/terrain_features.bin resources/g1_terrain/terrain.bin 1
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG -I. \
  tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_release
/tmp/test_terrain_runtime_release \
  resources/g1_terrain/terrain_features.bin resources/g1_terrain/terrain.bin 1
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_san
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/test_terrain_runtime_san \
  resources/g1_terrain/terrain_features.bin resources/g1_terrain/terrain.bin 1
~~~

Expected: all four binaries exit 0 with no output, compiler warning, sanitizer
report, compiled-away check, or v1 regression. The live published pack loads as
v1; the in-memory oracle loads as v2.

- [ ] **Step 6: Compile the real controller and prove Gate A v1 byte stability**

Build with the exact production flags and run the frozen v1 pack:

~~~bash
g++ -O3 -ffast-math -march=native -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I /home/ubuntu/apps/raylib/src \
  -I /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_task5 \
  -L /home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
sha256sum --check /tmp/g1_gate_a_before.sha256
test ! -e /tmp/g1_gate_a_task5.csv
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=375 \
  MM_LOG=/tmp/g1_gate_a_task5.csv /tmp/controller_g1_task5
test "$(wc -l < /tmp/g1_gate_a_task5.csv)" -eq 376
cmp -s /tmp/g1_gate_a_before.csv /tmp/g1_gate_a_task5.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_gate_a_task5.csv --gate-a
sha256sum --check /tmp/g1_gate_a_before.sha256
~~~

Expected: production compilation and both checks succeed; the new v1 log is
byte-identical to frozen Gate A, proving loader migration did not change the
currently running behavior or artifact bytes. The frozen controller and its
centerline must still call only literal legacy `heightfield_sample`; this gate
does not authorize using that name for a future G1HF/v2 scene.

- [ ] **Step 7: Commit the shared runtime surface primitive**

~~~bash
git add terrain_runtime.h tests/cpp/test_terrain_runtime.cpp
git diff --cached --check
git commit -m "feat: sample G1HF v2 fixed triangles"
~~~

### Task 6: Derive three-column source support rows per clip

**Files:**
- Modify: `resources/g1_terrain_builder/schema.py:32-120`
- Modify: `resources/g1_terrain_builder/database.py:104-329,383-454`
- Modify: `resources/g1_terrain_builder/kinematics.py:221-231`
- Modify: `resources/build_g1_terrain_database.py:55-83`
- Modify: `tests/python/test_schema.py:23-37`
- Modify: `tests/python/test_database_builder.py:53-398`
- Modify: `tests/python/test_kinematics.py:55-88`

**Interfaces:**
- `HoldenClip.terrain_support: ndarray[float32]` has shape `(frames, 3)`.
- `ArtifactSet.terrain_support: ndarray[float32]` has shape `(frames, 3)`; it is not serialized into `database.bin`.
- Produces: `sample_terrain_support(global_positions, terrain, root, left_toe, right_toe) -> ndarray`.
- `finalize_clip` uses one terrain instance for contacts, four feature columns, and all three support columns.

- [ ] **Step 1: Write failing schema and support-sampling tests**

Add to `tests/python/test_schema.py`:

~~~python
    def test_holden_clip_requires_three_finite_support_columns(self):
        clip = HoldenClip.empty(frames=3, bones=2)
        clip.terrain_support = np.zeros((3, 2), np.float32)
        with self.assertRaisesRegex(ValueError, "support shape"):
            clip.validate()
        clip.terrain_support = np.full((3, 3), np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            clip.validate()
~~~

Import `sample_terrain_support` in `tests/python/test_database_builder.py` and
add:

~~~python
    def test_source_support_samples_root_and_named_toes_in_column_order(self):
        positions = np.zeros((2, 4, 3), np.float64)
        positions[0, :, 0] = [0.25, 0.75, 0.25, 0.75]
        positions[1, :, 0] = [0.75, 0.25, 0.75, 0.25]
        support = sample_terrain_support(
            positions, StepTerrain(0.5, 0.29), 0, 2, 3)
        np.testing.assert_array_equal(support, np.array([
            [0.0, 0.0, 0.29],
            [0.29, 0.29, 0.0],
        ], np.float32))

    def test_flat_source_support_is_exact_zero(self):
        positions = np.arange(45, dtype=np.float64).reshape(5, 3, 3)
        support = sample_terrain_support(
            positions, FlatTerrain(), 0, 1, 2)
        np.testing.assert_array_equal(support, np.zeros((5, 3), np.float32))

    def test_combination_preserves_support_rows_at_clip_boundaries(self):
        first = HoldenClip.empty(3, 1)
        second = HoldenClip.empty(2, 1)
        first.terrain_features[:] = [10.0, 11.0, 12.0, 13.0]
        second.terrain_features[:] = [20.0, 21.0, 22.0, 23.0]
        first.terrain_support[:] = [1.0, 2.0, 3.0]
        second.terrain_support[:] = [4.0, 5.0, 6.0]
        artifacts = combine_clips(
            [first, second],
            SkeletonSpec(("Simulation",), np.array([-1], np.int32)))
        np.testing.assert_array_equal(artifacts.terrain_support, [
            [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [4, 5, 6], [4, 5, 6],
        ])
        np.testing.assert_array_equal(artifacts.terrain_features, [
            [10, 11, 12, 13], [10, 11, 12, 13], [10, 11, 12, 13],
            [20, 21, 22, 23], [20, 21, 22, 23],
        ])

    def test_support_rejects_bad_indices_and_nonfinite_heights(self):
        positions = np.zeros((2, 3, 3), np.float64)
        with self.assertRaisesRegex(ValueError, "support bone indices"):
            sample_terrain_support(positions, FlatTerrain(), 0, 1, 3)
        for boolean in (True, np.bool_(False)):
            with self.subTest(boolean=boolean):
                with self.assertRaisesRegex(ValueError, "support bone indices"):
                    sample_terrain_support(
                        positions, FlatTerrain(), boolean, 1, 2)

        class BadTerrain:
            def height(self, x, z):
                return np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            sample_terrain_support(positions, BadTerrain(), 0, 1, 2)

        class Float32OverflowTerrain:
            def height(self, x, z):
                return 1e300

        with self.assertRaisesRegex(ValueError, "float32|finite"):
            sample_terrain_support(
                positions, Float32OverflowTerrain(), 0, 1, 2)
~~~

In the existing `test_convert_source_clip_prepends_simulation_bone`, require
the direct constructor path to initialize sidecar rows before finalization:

~~~python
        self.assertEqual(clip.terrain_support.dtype, np.float32)
        np.testing.assert_array_equal(
            clip.terrain_support, np.zeros((5, 3), np.float32))
~~~

- [ ] **Step 2: Run the focused tests and verify the schema/API failures**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_schema \
  tests.python.test_database_builder.DatabaseBuilderTests.test_source_support_samples_root_and_named_toes_in_column_order \
  tests.python.test_database_builder.DatabaseBuilderTests.test_combination_preserves_support_rows_at_clip_boundaries \
  tests.python.test_kinematics.KinematicsTests.test_convert_source_clip_prepends_simulation_bone -v
~~~

Expected: import failure for `sample_terrain_support` and constructor/attribute
failures because the dataclasses have no support rows.

- [ ] **Step 3: Extend the clip and artifact schemas without changing database.bin**

Add `terrain_support` immediately after `terrain_features` in both dataclasses.
The exact definitions become:

~~~python
@dataclass
class HoldenClip:
    name: str
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray
    terrain_support: np.ndarray
    source_frames: np.ndarray
    terrain_id: str

    @classmethod
    def empty(cls, frames: int, bones: int) -> "HoldenClip":
        return cls(
            "empty",
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, bones, 3), np.float32),
            np.tile(np.array([1, 0, 0, 0], np.float32),
                    (frames, bones, 1)),
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, 2), np.uint8),
            np.zeros((frames, 4), np.float32),
            np.zeros((frames, 3), np.float32),
            np.arange(frames),
            "flat",
        )

    def validate(self) -> None:
        frames, bones, xyz = self.positions.shape
        if xyz != 3 or self.velocities.shape != (frames, bones, 3):
            raise ValueError("position/velocity shape mismatch")
        if self.rotations.shape != (frames, bones, 4):
            raise ValueError("rotation shape mismatch")
        if self.angular_velocities.shape != (frames, bones, 3):
            raise ValueError("angular velocity shape mismatch")
        if self.contacts.shape != (frames, 2):
            raise ValueError("contact shape mismatch")
        if self.terrain_features.shape != (frames, 4):
            raise ValueError("terrain feature shape must be (T, 4)")
        if self.terrain_support.shape != (frames, 3):
            raise ValueError("terrain support shape must be (T, 3)")
        arrays = (
            self.positions, self.velocities, self.rotations,
            self.angular_velocities, self.terrain_features,
            self.terrain_support,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("converted clip contains non-finite values")


@dataclass
class ArtifactSet:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray
    terrain_support: np.ndarray
~~~

Make `ArtifactSet.empty` carry `clip.terrain_support`, and require shape
`(frames, 3)` and finite values in `ArtifactSet.validate` and
`database._validate_artifacts`. Add the concatenation to `combine_clips`:

~~~python
np.concatenate([clip.terrain_support for clip in clips]).astype(
    np.float32, copy=False),
~~~

Because support is a sidecar, keep `write_holden_database` byte-for-byte
unchanged. Initialize `terrain_support` to zeros in `read_holden_database`; the
sidecar loader overwrites it after database loading.

`convert_source_clip` also constructs `HoldenClip` directly. Insert
`np.zeros((len(positions), 3), np.float32)` immediately after its four-column
terrain-feature zeros; do not add a dataclass default that could hide another
positional constructor mismatch.

- [ ] **Step 4: Implement exact root/left/right support sampling**

Add to `resources/g1_terrain_builder/database.py` before `derive_contacts`:

~~~python
def sample_terrain_support(
    global_positions: np.ndarray,
    terrain,
    root: int,
    left_toe: int,
    right_toe: int,
) -> np.ndarray:
    positions = np.asarray(global_positions, np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or not len(positions):
        raise ValueError("support positions must have shape (frames, bones, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("support positions must be finite")
    indices = (root, left_toe, right_toe)
    if any(
        not isinstance(index, Integral)
        or isinstance(index, (bool, np.bool_)) or index < 0
        or index >= positions.shape[1] for index in indices
    ) or len(set(indices)) != 3:
        raise ValueError("support bone indices must be distinct and in range")
    if not callable(getattr(terrain, "height", None)):
        raise TypeError("support terrain must provide height(x, z)")
    support = np.empty((len(positions), 3), np.float32)
    for frame in range(len(positions)):
        for column, bone in enumerate(indices):
            value = float(terrain.height(
                float(positions[frame, bone, 0]),
                float(positions[frame, bone, 2])))
            with np.errstate(over="ignore", invalid="ignore"):
                encoded = np.float32(value)
            if not np.isfinite(value) or not np.isfinite(encoded):
                raise ValueError(
                    "support terrain heights must be finite float32")
            support[frame, column] = encoded
    return support
~~~

- [ ] **Step 5: Wire one provider through contacts, features, and support**

Import `sample_terrain_support` in `build_g1_terrain_database.py` and add this
immediately after `derive_contacts` in `finalize_clip`:

~~~python
clip.terrain_support = sample_terrain_support(
    gp,
    terrain,
    skeleton.names.index("Simulation"),
    skeleton.names.index("LeftToe"),
    skeleton.names.index("RightToe"),
)
~~~

The existing feature loop continues to receive the same `terrain` object. Do
not reconstruct `GrailTerrain` between contacts, support, or feature sampling.

- [ ] **Step 6: Run schema, builder, and exact database-byte tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_schema tests.python.test_database_builder \
  tests.python.test_kinematics -v
~~~

Expected: all tests pass, including the existing exact `database.bin`
little-endian comparison, proving the new support rows did not change the
Holden database format.

- [ ] **Step 7: Commit source support derivation**

~~~bash
git add resources/g1_terrain_builder/schema.py \
  resources/g1_terrain_builder/database.py \
  resources/g1_terrain_builder/kinematics.py \
  resources/build_g1_terrain_database.py \
  tests/python/test_schema.py tests/python/test_database_builder.py \
  tests/python/test_kinematics.py
git diff --cached --check
git commit -m "feat: derive G1 source support rows"
~~~

### Task 7: Serialize strict G1SP/v1 and G1WM/v1 sidecars

**Files:**
- Modify: `resources/g1_terrain_builder/artifacts.py:1-208`
- Modify: `tests/python/test_artifacts.py:1-321`

**Interfaces:**
- Produces: `terrain_sidecar_bytes(values) -> bytes` while preserving the
  existing `write_terrain_sidecar` and `read_terrain_sidecar` names.
- Produces: `support_sidecar_bytes(values) -> bytes`.
- Produces: `write_support_sidecar(path, values) -> None` and `read_support_sidecar(path) -> ndarray[float32]`.
- Produces: `walkability_bytes(values) -> bytes`.
- Produces: `write_walkability(path, values) -> None` and `read_walkability(path) -> ndarray[uint8]`.
- G1SP columns are locked as `SUPPORT_COLUMNS = ("source_root_height_m", "source_left_toe_height_m", "source_right_toe_height_m")`.

- [ ] **Step 1: Write exact-byte and corruption tests for both formats**

Extend the artifact import list and add to `tests/python/test_artifacts.py`:

~~~python
from resources.g1_terrain_builder.artifacts import (
    SUPPORT_COLUMNS,
    publish_artifacts,
    read_support_sidecar,
    read_terrain_sidecar,
    read_walkability,
    support_sidecar_bytes,
    walkability_bytes,
    write_support_sidecar,
    write_terrain_sidecar,
    write_walkability,
)

    def test_support_sidecar_is_exact_g1sp_v1_little_endian(self):
        values = np.arange(15, dtype=np.float32).reshape(5, 3)
        payload = support_sidecar_bytes(values)
        self.assertEqual(
            payload[:16], struct.pack("<4sIII", b"G1SP", 1, 5, 3))
        self.assertEqual(payload[16:], values.astype("<f4").tobytes())
        self.assertEqual(SUPPORT_COLUMNS, (
            "source_root_height_m",
            "source_left_toe_height_m",
            "source_right_toe_height_m",
        ))
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_support.bin")
            write_support_sidecar(path, values)
            np.testing.assert_array_equal(read_support_sidecar(path), values)

    def test_support_sidecar_rejects_schema_payload_and_nonfinite_values(self):
        valid = struct.pack("<4sIII", b"G1SP", 1, 2, 3) \
            + np.zeros((2, 3), "<f4").tobytes()
        corruptions = (
            (valid[:12], "truncated"),
            (b"BAD!" + valid[4:], "schema"),
            (struct.pack("<4sIII", b"G1SP", 2, 2, 3) + valid[16:], "schema"),
            (struct.pack("<4sIII", b"G1SP", 1, 2, 2) + valid[16:], "schema"),
            (valid[:-1], "truncated"),
            (valid + b"x", "trailing"),
            (struct.pack("<4sIII", b"G1SP", 1, 1, 3)
             + np.array([[0, np.nan, 0]], "<f4").tobytes(), "finite"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (payload, message) in enumerate(corruptions):
                path = os.path.join(temporary, f"support-{index}.bin")
                open(path, "wb").write(payload)
                with self.subTest(index=index):
                    with self.assertRaisesRegex(ValueError, message):
                        read_support_sidecar(path)

    def test_walkability_is_exact_g1wm_v1_row_major_bytes(self):
        values = np.array([[0, 1, 2], [2, 1, 0]], np.uint8)
        payload = walkability_bytes(values)
        self.assertEqual(
            payload[:16], struct.pack("<4sIII", b"G1WM", 1, 3, 2))
        self.assertEqual(payload[16:], bytes([0, 1, 2, 2, 1, 0]))
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            write_walkability(path, values)
            np.testing.assert_array_equal(read_walkability(path), values)

    def test_walkability_rejects_bad_classes_grid_size_and_trailing_data(self):
        with self.assertRaisesRegex(ValueError, "classes"):
            walkability_bytes(np.array([[0, 3], [1, 2]], np.uint8))
        with self.assertRaisesRegex(ValueError, "grid"):
            walkability_bytes(np.zeros((1, 2), np.uint8))
        valid = walkability_bytes(np.zeros((2, 2), np.uint8))
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            open(path, "wb").write(valid + b"x")
            with self.assertRaisesRegex(ValueError, "trailing"):
                read_walkability(path)
~~~

- [ ] **Step 2: Run the focused artifact tests and verify missing imports**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts.ArtifactTests.test_support_sidecar_is_exact_g1sp_v1_little_endian \
  tests.python.test_artifacts.ArtifactTests.test_walkability_is_exact_g1wm_v1_row_major_bytes -v
~~~

Expected: import errors for the new sidecar functions.

- [ ] **Step 3: Factor a strict float-matrix codec and add G1SP/v1**

Add these constants and helpers in `artifacts.py`, preserving the existing
G1TF public function names:

~~~python
MATRIX_HEADER = struct.Struct("<4sIII")
UINT32_MAX = (1 << 32) - 1
TERRAIN_FEATURE_MAGIC = b"G1TF"
TERRAIN_FEATURE_DIMS = 4
SUPPORT_MAGIC = b"G1SP"
SUPPORT_DIMS = 3
SUPPORT_COLUMNS = (
    "source_root_height_m",
    "source_left_toe_height_m",
    "source_right_toe_height_m",
)


def _float_matrix_bytes(values, magic, dimensions, label):
    try:
        matrix = np.ascontiguousarray(values, dtype="<f4")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite float values") from error
    if matrix.ndim != 2 or len(matrix) < 1 \
            or matrix.shape[1] != dimensions \
            or len(matrix) > UINT32_MAX or not np.isfinite(matrix).all():
        raise ValueError(
            f"{label} must be finite (N, {dimensions}), got {matrix.shape}")
    return MATRIX_HEADER.pack(
        magic, 1, len(matrix), dimensions) + matrix.tobytes(order="C")


def _read_float_matrix(path, magic, dimensions, label):
    with open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < MATRIX_HEADER.size:
        raise ValueError(f"{path}: truncated {label} header")
    got_magic, version, frames, got_dimensions = MATRIX_HEADER.unpack_from(payload)
    if got_magic != magic or version != 1 or got_dimensions != dimensions:
        raise ValueError(f"{path}: unsupported {label} schema")
    expected = MATRIX_HEADER.size + frames * dimensions * 4
    if len(payload) < expected:
        raise ValueError(f"{path}: truncated {label} payload")
    if len(payload) > expected:
        raise ValueError(f"{path}: trailing {label} bytes")
    if frames < 1:
        raise ValueError(f"{path}: invalid {label} frame count")
    values = np.frombuffer(
        payload, "<f4", frames * dimensions,
        MATRIX_HEADER.size).reshape(frames, dimensions).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: {label} values must be finite")
    return values


def support_sidecar_bytes(values):
    return _float_matrix_bytes(values, SUPPORT_MAGIC, SUPPORT_DIMS, "support")


def write_support_sidecar(path, values):
    with open(path, "wb") as stream:
        stream.write(support_sidecar_bytes(values))


def read_support_sidecar(path):
    return _read_float_matrix(path, SUPPORT_MAGIC, SUPPORT_DIMS, "support")


def terrain_sidecar_bytes(values):
    return _float_matrix_bytes(
        values, TERRAIN_FEATURE_MAGIC, TERRAIN_FEATURE_DIMS,
        "terrain sidecar")


def write_terrain_sidecar(path, values):
    with open(path, "wb") as stream:
        stream.write(terrain_sidecar_bytes(values))


def read_terrain_sidecar(path):
    return _read_float_matrix(
        path, TERRAIN_FEATURE_MAGIC, TERRAIN_FEATURE_DIMS,
        "terrain sidecar")
~~~

- [ ] **Step 4: Implement strict G1WM/v1 bytes and parsing**

Add:

~~~python
WALKABILITY_MAGIC = b"G1WM"
WALKABILITY_VERSION = 1


def walkability_bytes(values):
    grid = np.asarray(values)
    if grid.ndim != 2 or min(grid.shape, default=0) < 2 \
            or max(grid.shape) > UINT32_MAX:
        raise ValueError("walkability grid must have shape (nz>=2, nx>=2)")
    if not np.issubdtype(grid.dtype, np.integer) \
            or np.any((grid < 0) | (grid > 2)):
        raise ValueError("walkability classes must be integer 0, 1, or 2")
    grid = np.ascontiguousarray(grid, np.uint8)
    nz, nx = grid.shape
    return MATRIX_HEADER.pack(
        WALKABILITY_MAGIC, WALKABILITY_VERSION, nx, nz,
    ) + grid.tobytes(order="C")


def write_walkability(path, values):
    with open(path, "wb") as stream:
        stream.write(walkability_bytes(values))


def read_walkability(path):
    with open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < MATRIX_HEADER.size:
        raise ValueError(f"{path}: truncated walkability header")
    magic, version, nx, nz = MATRIX_HEADER.unpack_from(payload)
    if magic != WALKABILITY_MAGIC or version != 1:
        raise ValueError(f"{path}: unsupported walkability schema")
    if nx < 2 or nz < 2:
        raise ValueError(f"{path}: invalid walkability grid dimensions")
    expected = MATRIX_HEADER.size + nx * nz
    if len(payload) < expected:
        raise ValueError(f"{path}: truncated walkability payload")
    if len(payload) > expected:
        raise ValueError(f"{path}: trailing walkability bytes")
    values = np.frombuffer(
        payload, np.uint8, nx * nz, MATRIX_HEADER.size).reshape(nz, nx).copy()
    if np.any(values > 2):
        raise ValueError(f"{path}: invalid walkability classes")
    return values
~~~

- [ ] **Step 5: Include terrain_support.bin in the existing staged publish check**

Write `terrain_support.bin` beside `terrain_features.bin` in
`publish_artifacts`, load it in `_validate_staged_artifacts`, assign it to
`loaded.terrain_support`, and include `("terrain_support", "<f4")` in the exact
array comparisons. Update the existing publication test's expected file set to
include `terrain_support.bin`.

~~~python
write_support_sidecar(
    os.path.join(staging, "terrain_support.bin"),
    artifacts.terrain_support,
)
~~~

- [ ] **Step 6: Run the complete artifact suite**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts -v
~~~

Expected: all artifact tests pass, including prior G1TF and rollback tests plus
the new exact G1SP/G1WM tests.

- [ ] **Step 7: Commit the binary sidecar contracts**

~~~bash
git add resources/g1_terrain_builder/artifacts.py \
  tests/python/test_artifacts.py
git diff --cached --check
git commit -m "feat: serialize G1 support and walkability"
~~~

### Task 8: Lock scene/index JSON and build hashed scene packs

**Files:**
- Create: `resources/g1_terrain_builder/scenes.py`
- Create: `tests/python/test_scenes.py`

**Interfaces:**
- Produces: `SceneRoute`, `SceneDefinition`, `BuiltScene`, and `ScenePack`.
- Produces: `build_scene(definition) -> BuiltScene` and `build_scene_pack(definitions) -> ScenePack`.
- Produces: `canonical_json_bytes(value) -> bytes` and `sha256_hex(payload) -> str`; every JSON SHA is over the exact indented, sorted, newline-terminated bytes written to disk.
- Locks the 14 `REQUIRED_SCENE_IDS` in index order.
- Locks every `scene.json` key, nested key, route outcome, relative path, and hash field consumed by later runtime plans.

- [ ] **Step 1: Write failing schema, hash, route, and ordered-index tests**

Create `tests/python/test_scenes.py`:

~~~python
import hashlib
import json
import struct
import unittest

import numpy as np

from resources.g1_terrain_builder.artifacts import read_walkability
from resources.g1_terrain_builder.scenes import (
    REQUIRED_SCENE_IDS,
    BuiltScene,
    SceneDefinition,
    SceneRoute,
    build_scene,
    build_scene_pack,
    canonical_json_bytes,
)
from resources.g1_terrain_builder.terrain import FlatTerrain, \
    surface_semantics_signature


def flat_definition(scene_id="grail-curb-default"):
    return SceneDefinition(
        scene_id=scene_id,
        label=scene_id.replace("-", " ").title(),
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": {"fixture": True},
        },
        surface=FlatTerrain(),
        heightfield_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        playable_bounds_xz=(-0.5, 0.5, 0.0, 2.0),
        lookahead_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": ({"id": "route", "bounds_xz": [-0.5, 0.5, 0, 2]},),
            "stress": (), "blocked": (),
        },
        routes=(SceneRoute(
            route_id="forward",
            waypoints_xz=(
                (0.0, 0.0), (0.0, 0.5),
                (0.0, 1.0), (0.0, 2.0)),
            expected_outcome="traverse",
            walkability_class=1,
            landing_hold_seconds=2.0,
        ),),
        walkability=lambda x, z: 1 if -0.5 <= x <= 0.5 and 0 <= z <= 2 else 0,
    )


class SceneSchemaTests(unittest.TestCase):
    def test_canonical_json_hash_is_over_exact_written_bytes(self):
        value = {"z": 1, "a": [2, 3]}
        payload = canonical_json_bytes(value)
        self.assertEqual(payload, b'{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n')
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            hashlib.sha256(canonical_json_bytes(value)).hexdigest())

    def test_built_scene_has_locked_keys_paths_schemas_and_hashes(self):
        scene = build_scene(flat_definition())
        metadata = scene.metadata
        self.assertEqual(set(metadata), {
            "schema", "id", "label", "provenance",
            "coordinate_signature", "surface_signature",
            "terrain_feature_distances_m", "heightfield", "mesh",
            "walkability", "bounds", "spawn", "regions", "routes",
        })
        self.assertEqual(metadata["schema"], "g1-terrain-scene/v1")
        self.assertEqual(metadata["heightfield"]["path"], "terrain.bin")
        self.assertEqual(metadata["heightfield"]["schema"], "G1HF/v2")
        self.assertEqual(metadata["heightfield"]["version"], 2)
        self.assertEqual(
            metadata["heightfield"]["diagonal"],
            "min-x-min-z_to_max-x-max-z")
        self.assertEqual(metadata["mesh"], {
            "path": "terrain.obj", "schema": "obj/v1",
            "sha256": hashlib.sha256(scene.terrain_obj).hexdigest(),
        })
        self.assertEqual(metadata["walkability"]["path"], "walkability.bin")
        self.assertEqual(metadata["walkability"]["classes"], {
            "blocked": 0, "certified": 1, "stress": 2,
        })
        self.assertEqual(
            metadata["heightfield"]["sha256"],
            hashlib.sha256(scene.terrain_bin).hexdigest())
        self.assertEqual(
            metadata["walkability"]["sha256"],
            hashlib.sha256(scene.walkability_bin).hexdigest())
        _, _, nx, nz, ox, oz, cell, _ = struct.unpack_from(
            "<4sIII4f", scene.terrain_bin)
        heightfield_min = [float(ox), 0.0, float(oz)]
        heightfield_max = [
            float(ox) + (nx - 1) * float(cell), 0.0,
            float(oz) + (nz - 1) * float(cell),
        ]
        vertex_lines = scene.terrain_obj.decode("ascii").splitlines()[:nx * nz]
        first = vertex_lines[0].split()
        last = vertex_lines[-1].split()
        mesh_min = [
            float(np.float32(first[1])), 0.0,
            float(np.float32(first[3])),
        ]
        mesh_max = [
            float(np.float32(last[1])), 0.0,
            float(np.float32(last[3])),
        ]
        self.assertEqual(metadata["bounds"]["heightfield_min_xyz"],
                         heightfield_min)
        self.assertEqual(metadata["bounds"]["heightfield_max_xyz"],
                         heightfield_max)
        self.assertEqual(metadata["bounds"]["mesh_min_xyz"], mesh_min)
        self.assertEqual(metadata["bounds"]["mesh_max_xyz"], mesh_max)
        self.assertNotEqual(mesh_max, heightfield_max)
        self.assertEqual(metadata["routes"], [{
            "id": "forward",
            "waypoints_xz": [
                [0.0, 0.0], [0.0, 0.5],
                [0.0, 1.0], [0.0, 2.0]],
            "expected_outcome": "traverse",
            "walkability_class": 1,
            "landing_hold_seconds": 2.0,
        }])

    def test_route_outcomes_and_classes_are_locked(self):
        valid = (
            ("traverse", 1),
            ("safe-stop", 0),
            ("traverse-or-safe-stop", 2),
        )
        for outcome, walkability_class in valid:
            SceneRoute(
                "route", ((0, 0), (0, 1)), outcome,
                walkability_class, 0.0).validate()
        with self.assertRaisesRegex(ValueError, "outcome.*class"):
            SceneRoute(
                "bad", ((0, 0), (0, 1)), "traverse", 2, 0.0).validate()
        with self.assertRaisesRegex(ValueError, "hold.*waypoint 2"):
            SceneRoute(
                "bad-hold", ((0, 0), (0, 1), (0, 2)),
                "traverse", 1, 2.0).validate()

    def test_scene_pack_requires_exact_order_and_stable_index_fields(self):
        definitions = [flat_definition(scene_id) for scene_id in REQUIRED_SCENE_IDS]
        pack = build_scene_pack(definitions)
        self.assertEqual(pack.index, {
            "schema": "g1-terrain-scene-index/v1",
            "default_scene_id": "grail-curb-default",
            "scene_ids": list(REQUIRED_SCENE_IDS),
            "coordinate_signature":
                "holden-y-up-right-handed-forward-plus-z",
            "surface_signature": surface_semantics_signature(),
        })
        with self.assertRaisesRegex(ValueError, "required scene order"):
            build_scene_pack(list(reversed(definitions)))

    def test_scene_builder_rejects_route_or_spawn_outside_contract(self):
        definition = flat_definition()
        definition.spawn_position = (5.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "spawn.*playable"):
            build_scene(definition)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the schema tests and verify the missing module failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes -v
~~~

Expected: `ERROR` with `ModuleNotFoundError` for
`resources.g1_terrain_builder.scenes`.

- [ ] **Step 3: Implement the immutable scene, route, and index contracts**

Create `resources/g1_terrain_builder/scenes.py` with:

~~~python
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np

from .artifacts import walkability_bytes
from .terrain import (
    HEIGHTFIELD_DIAGONAL,
    HEIGHTFIELD_INTERPOLATION,
    HeightGrid,
    rasterize_heightfield,
    surface_semantics_signature,
)


REQUIRED_SCENE_IDS = (
    "grail-curb-default",
    "grail-curb-low",
    "grail-curb-medium",
    "grail-curb-high",
    "stairs-shallow",
    "stairs-standard",
    "stairs-unseen-variable",
    "ramp-05-up-down",
    "ramp-10-up-down",
    "ramp-15-stress",
    "cross-slope-05",
    "cross-slope-10",
    "mixed-multilevel",
    "blocked-course",
)
COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
TERRAIN_DISTANCES = [0.25, 0.50, 0.75, 1.00]
SCENE_CELL_SIZE = 0.02
OUTCOME_CLASS = {
    "traverse": 1,
    "safe-stop": 0,
    "traverse-or-safe-stop": 2,
}


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SceneRoute:
    route_id: str
    waypoints_xz: tuple[tuple[float, float], ...]
    expected_outcome: str
    walkability_class: int
    landing_hold_seconds: float

    def validate(self):
        if not self.route_id or len(self.waypoints_xz) < 2:
            raise ValueError("scene route needs an ID and at least two waypoints")
        points = np.asarray(self.waypoints_xz, np.float64)
        if points.shape != (len(self.waypoints_xz), 2) \
                or not np.isfinite(points).all():
            raise ValueError("scene route waypoints must be finite XZ pairs")
        expected = OUTCOME_CLASS.get(self.expected_outcome)
        if expected != self.walkability_class:
            raise ValueError("scene route outcome and walkability class disagree")
        if not np.isfinite(self.landing_hold_seconds) \
                or self.landing_hold_seconds < 0.0:
            raise ValueError("landing hold seconds must be finite and nonnegative")
        if self.landing_hold_seconds > 0.0 and len(self.waypoints_xz) < 4:
            raise ValueError(
                "landing hold requires published waypoint 2 and a later exit")

    def to_json(self):
        self.validate()
        return {
            "id": self.route_id,
            "waypoints_xz": [[float(x), float(z)] for x, z in self.waypoints_xz],
            "expected_outcome": self.expected_outcome,
            "walkability_class": self.walkability_class,
            "landing_hold_seconds": float(self.landing_hold_seconds),
        }


@dataclass
class SceneDefinition:
    scene_id: str
    label: str
    provenance: dict
    surface: object
    heightfield_bounds_xz: tuple[float, float, float, float]
    playable_bounds_xz: tuple[float, float, float, float]
    lookahead_bounds_xz: tuple[float, float, float, float]
    spawn_position: tuple[float, float, float]
    spawn_yaw_radians: float
    regions: dict
    routes: tuple[SceneRoute, ...]
    walkability: Callable[[float, float], int]


@dataclass(frozen=True)
class BuiltScene:
    scene_id: str
    metadata: dict
    terrain_bin: bytes
    terrain_obj: bytes
    walkability_bin: bytes

    @property
    def scene_json(self):
        return canonical_json_bytes(self.metadata)


@dataclass(frozen=True)
class ScenePack:
    index: dict
    scenes: tuple[BuiltScene, ...]

    @property
    def index_json(self):
        return canonical_json_bytes(self.index)
~~~

- [ ] **Step 4: Build exact assets, hashes, bounds, and metadata from one grid**

Add:

~~~python
def _bounds_contains(bounds, x, z):
    xmin, xmax, zmin, zmax = bounds
    return xmin <= x <= xmax and zmin <= z <= zmax


def _classify_grid(definition, grid):
    values = np.empty((grid.nz, grid.nx), np.uint8)
    for iz in range(grid.nz):
        z = grid.origin_z + iz * grid.cell_size
        for ix in range(grid.nx):
            x = grid.origin_x + ix * grid.cell_size
            value = definition.walkability(x, z)
            if value not in (0, 1, 2):
                raise ValueError(
                    f"{definition.scene_id}: invalid walkability class {value}")
            values[iz, ix] = value
    return values


def _validate_definition(definition):
    if not definition.scene_id or not definition.label:
        raise ValueError("scene ID and label must be non-empty")
    if set(definition.provenance) != {"kind", "source_ids", "parameters"}:
        raise ValueError("scene provenance fields are not locked")
    if definition.provenance["kind"] not in ("grail", "procedural"):
        raise ValueError("scene provenance kind must be grail or procedural")
    numbers = (
        *definition.heightfield_bounds_xz,
        *definition.playable_bounds_xz,
        *definition.lookahead_bounds_xz,
        *definition.spawn_position,
        definition.spawn_yaw_radians,
    )
    if not np.isfinite(numbers).all():
        raise ValueError("scene bounds and spawn must be finite")
    sx, _, sz = definition.spawn_position
    if not _bounds_contains(definition.playable_bounds_xz, sx, sz):
        raise ValueError("scene spawn must lie inside playable bounds")
    if set(definition.regions) != {"certified", "stress", "blocked"}:
        raise ValueError("scene regions must define certified, stress, blocked")
    if not definition.routes:
        raise ValueError("scene must define at least one route")
    for route in definition.routes:
        route.validate()
        for x, z in route.waypoints_xz:
            if not _bounds_contains(definition.lookahead_bounds_xz, x, z):
                raise ValueError("scene route leaves lookahead bounds")


def build_scene(definition):
    _validate_definition(definition)
    grid = rasterize_heightfield(
        definition.surface,
        definition.heightfield_bounds_xz,
        SCENE_CELL_SIZE,
    )
    classes = _classify_grid(definition, grid)
    terrain_bin = grid.g1hf_bytes()
    terrain_obj = grid.obj_bytes()
    walkability_bin = walkability_bytes(classes)
    minimum_y = float(grid.heights.min())
    maximum_y = float(grid.heights.max())
    heightfield_xmin = grid.origin_x
    heightfield_xmax = \
        heightfield_xmin + (grid.nx - 1) * grid.cell_size
    heightfield_zmin = grid.origin_z
    heightfield_zmax = \
        heightfield_zmin + (grid.nz - 1) * grid.cell_size
    def obj_coordinate(value):
        encoded = float(np.float32(value))
        return 0.0 if encoded == 0.0 else encoded

    mesh_xmin = obj_coordinate(heightfield_xmin)
    mesh_xmax = obj_coordinate(heightfield_xmax)
    mesh_zmin = obj_coordinate(heightfield_zmin)
    mesh_zmax = obj_coordinate(heightfield_zmax)
    metadata = {
        "schema": "g1-terrain-scene/v1",
        "id": definition.scene_id,
        "label": definition.label,
        "provenance": definition.provenance,
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
        "terrain_feature_distances_m": TERRAIN_DISTANCES,
        "heightfield": {
            "path": "terrain.bin", "schema": "G1HF/v2", "version": 2,
            "nx": grid.nx, "nz": grid.nz,
            "origin_x": grid.origin_x, "origin_z": grid.origin_z,
            "cell_size_m": grid.cell_size,
            "exterior_height_m": grid.exterior_height,
            "interpolation": HEIGHTFIELD_INTERPOLATION,
            "diagonal": HEIGHTFIELD_DIAGONAL,
            "sha256": sha256_hex(terrain_bin),
        },
        "mesh": {
            "path": "terrain.obj", "schema": "obj/v1",
            "sha256": sha256_hex(terrain_obj),
        },
        "walkability": {
            "path": "walkability.bin", "schema": "G1WM/v1", "version": 1,
            "nx": grid.nx, "nz": grid.nz,
            "classes": {"blocked": 0, "certified": 1, "stress": 2},
            "sha256": sha256_hex(walkability_bin),
        },
        "bounds": {
            "mesh_min_xyz": [mesh_xmin, minimum_y, mesh_zmin],
            "mesh_max_xyz": [mesh_xmax, maximum_y, mesh_zmax],
            "heightfield_min_xyz": [
                heightfield_xmin, minimum_y, heightfield_zmin],
            "heightfield_max_xyz": [
                heightfield_xmax, maximum_y, heightfield_zmax],
            "playable_min_xz": [
                definition.playable_bounds_xz[0],
                definition.playable_bounds_xz[2]],
            "playable_max_xz": [
                definition.playable_bounds_xz[1],
                definition.playable_bounds_xz[3]],
            "lookahead_min_xz": [
                definition.lookahead_bounds_xz[0],
                definition.lookahead_bounds_xz[2]],
            "lookahead_max_xz": [
                definition.lookahead_bounds_xz[1],
                definition.lookahead_bounds_xz[3]],
        },
        "spawn": {
            "position": [float(v) for v in definition.spawn_position],
            "yaw_radians": float(definition.spawn_yaw_radians),
        },
        "regions": {
            name: [dict(region) for region in definition.regions[name]]
            for name in ("certified", "stress", "blocked")
        },
        "routes": [route.to_json() for route in definition.routes],
    }
    canonical_json_bytes(metadata)  # rejects non-finite/non-JSON provenance
    return BuiltScene(
        definition.scene_id, metadata,
        terrain_bin, terrain_obj, walkability_bin)


def build_scene_pack(definitions):
    ids = tuple(definition.scene_id for definition in definitions)
    if ids != REQUIRED_SCENE_IDS:
        raise ValueError(
            f"required scene order is {REQUIRED_SCENE_IDS}, got {ids}")
    scenes = tuple(build_scene(definition) for definition in definitions)
    index = {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "grail-curb-default",
        "scene_ids": list(REQUIRED_SCENE_IDS),
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }
    return ScenePack(index, scenes)
~~~

The heightfield bounds above are the exact promoted-double node rectangle used
for queries. OBJ X/Z vertices are explicitly binary32-quantized, so the mesh
bounds record the actual first/last serialized vertex coordinates separately;
do not assume the two maxima are numerically identical.

- [ ] **Step 5: Run the scene-schema tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes -v
~~~

Expected: 5 tests pass; every serialized artifact hash matches the exact bytes
held by `BuiltScene`, and reversing the catalog is rejected.

- [ ] **Step 6: Commit the scene pack boundary**

~~~bash
git add resources/g1_terrain_builder/scenes.py tests/python/test_scenes.py
git diff --cached --check
git commit -m "feat: define hashed G1 scene packs"
~~~

### Task 9: Generate the ten deterministic procedural scene definitions

**Files:**
- Modify: `resources/g1_terrain_builder/scenes.py`
- Modify: `tests/python/test_scenes.py`

**Interfaces:**
- Produces: `procedural_scene_definitions() -> tuple[SceneDefinition, ...]` in
  `REQUIRED_SCENE_IDS[4:]` order.
- Produces exact continuous `LongitudinalProfileSurface`, `CrossSlopeSurface`,
  and `BlockedCourseSurface` providers; scene rasterization still occurs only
  once in `build_scene`.
- Locks certified route IDs to `ascent-landing-descent` for stairs,
  `up-landing-down` for 5/10-degree ramps, `forward-cross-slope` for both
  cross-slopes, and `full-course` for the mixed scene.
- Locks stress/blocked route IDs to `up-landing-down` for the 15-degree ramp
  and `wall-safe-stop`/`ramp-safe-stop` for the blocked course.
- Every course has a flat `2.0 m` spawn lead, and every certified route point
  has at least `1.0 m` of heightfield margin in X and Z.

- [ ] **Step 1: Write failing geometry, route, and walkability tests**

Extend the `scenes` import in `tests/python/test_scenes.py` with
`procedural_scene_definitions`, then add:

~~~python
class ProceduralSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = {
            scene.scene_id: scene
            for scene in procedural_scene_definitions()
        }

    def test_procedural_catalog_order_and_route_contracts(self):
        definitions = procedural_scene_definitions()
        self.assertEqual(
            tuple(scene.scene_id for scene in definitions),
            REQUIRED_SCENE_IDS[4:])
        expected = {
            "stairs-shallow": ("ascent-landing-descent", "traverse", 1),
            "stairs-standard": ("ascent-landing-descent", "traverse", 1),
            "stairs-unseen-variable": (
                "ascent-landing-descent", "traverse", 1),
            "ramp-05-up-down": ("up-landing-down", "traverse", 1),
            "ramp-10-up-down": ("up-landing-down", "traverse", 1),
            "ramp-15-stress": (
                "up-landing-down", "traverse-or-safe-stop", 2),
            "cross-slope-05": ("forward-cross-slope", "traverse", 1),
            "cross-slope-10": ("forward-cross-slope", "traverse", 1),
            "mixed-multilevel": ("full-course", "traverse", 1),
        }
        for scene_id, contract in expected.items():
            route = self.definitions[scene_id].routes[0]
            self.assertEqual(
                (route.route_id, route.expected_outcome,
                 route.walkability_class), contract)
        blocked = self.definitions["blocked-course"].routes
        self.assertEqual(
            [(route.route_id, route.expected_outcome, route.walkability_class)
             for route in blocked],
            [("wall-safe-stop", "safe-stop", 0),
             ("ramp-safe-stop", "safe-stop", 0)])

    def test_stair_dimensions_landings_and_return_to_base_are_exact(self):
        expected = {
            "stairs-shallow": ([0.08] * 4, [0.30] * 4),
            "stairs-standard": ([0.12] * 3, [0.32] * 3),
            "stairs-unseen-variable": (
                [0.06, 0.10, 0.08, 0.12],
                [0.24, 0.34, 0.28, 0.38]),
        }
        for scene_id, (rises, runs) in expected.items():
            scene = self.definitions[scene_id]
            parameters = scene.provenance["parameters"]
            self.assertEqual(parameters["rises_m"], rises)
            self.assertEqual(parameters["runs_m"], runs)
            self.assertEqual(parameters["width_m"], 1.2)
            self.assertEqual(parameters["landing_length_m"], 2.0)
            ascent_end = 2.0 + sum(runs)
            self.assertAlmostEqual(
                scene.surface.height(0.0, ascent_end + 1.0), sum(rises))
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + 2 * sum(runs) + 2.01), 0.0)

    def test_ramp_run_is_exact_formula_and_profile_is_continuous(self):
        for degrees in (5, 10, 15):
            scene_id = f"ramp-{degrees:02d}-" + (
                "stress" if degrees == 15 else "up-down")
            scene = self.definitions[scene_id]
            run = 0.36 / np.tan(np.deg2rad(degrees))
            self.assertAlmostEqual(
                scene.provenance["parameters"]["run_m"], run, places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + run / 2.0), 0.18,
                places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + run + 1.0), 0.36,
                places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + 2 * run + 2.0), 0.0,
                places=12)

    def test_cross_slopes_have_four_metre_grade_and_flat_entry_exit(self):
        for degrees in (5, 10):
            scene = self.definitions[f"cross-slope-{degrees:02d}"]
            parameters = scene.provenance["parameters"]
            self.assertEqual(parameters["grade_length_m"], 4.0)
            self.assertEqual(parameters["flat_spawn_length_m"], 2.0)
            self.assertEqual(parameters["flat_entry_length_m"], 1.0)
            self.assertEqual(parameters["flat_exit_length_m"], 1.0)
            self.assertEqual(scene.surface.height(0.6, 2.5), 0.0)
            self.assertAlmostEqual(
                scene.surface.height(0.6, 5.0),
                0.6 * np.tan(np.deg2rad(degrees)), places=12)
            self.assertEqual(scene.surface.height(0.6, 7.5), 0.0)

    def test_mixed_course_signed_blocks_and_ramp_return_to_zero(self):
        scene = self.definitions["mixed-multilevel"]
        parameters = scene.provenance["parameters"]
        self.assertEqual(parameters["stair_rises_m"], [0.08] * 4)
        self.assertEqual(parameters["elevated_walk_length_m"], 3.0)
        self.assertEqual(parameters["block_height_changes_m"], [0.08, -0.12, 0.04])
        self.assertEqual(parameters["block_top_length_m"], 0.60)
        starts = parameters["block_starts_z_m"]
        for start, expected_height in zip(starts, (0.40, 0.28, 0.32)):
            self.assertAlmostEqual(
                scene.surface.height(0.0, start + 0.30), expected_height)
        self.assertAlmostEqual(
            scene.surface.height(0.0, parameters["course_end_z_m"]), 0.0,
            places=12)

    def test_spawn_margin_corridor_and_walkability_are_locked(self):
        for scene in procedural_scene_definitions():
            self.assertEqual(scene.surface.height(0.0, 0.0), 0.0)
            self.assertGreaterEqual(
                scene.provenance["parameters"]["flat_spawn_length_m"], 2.0)
            xmin, xmax, zmin, zmax = scene.heightfield_bounds_xz
            for route in scene.routes:
                if route.expected_outcome != "traverse":
                    continue
                for x, z in route.waypoints_xz:
                    self.assertGreaterEqual(x - xmin, 1.0)
                    self.assertGreaterEqual(xmax - x, 1.0)
                    self.assertGreaterEqual(z - zmin, 1.0)
                    self.assertGreaterEqual(zmax - z, 1.0)
                    self.assertEqual(scene.walkability(x, z), 1)

    def test_blocked_course_marks_both_obstacle_approaches_blocked(self):
        scene = self.definitions["blocked-course"]
        parameters = scene.provenance["parameters"]
        self.assertEqual(parameters["wall_height_m"], 0.45)
        self.assertEqual(parameters["wall_top_length_m"], 0.50)
        self.assertEqual(parameters["ramp_rise_m"], 0.36)
        self.assertEqual(parameters["ramp_angle_degrees"], 25.0)
        self.assertEqual(scene.walkability(-0.8, 1.5), 1)
        self.assertEqual(scene.walkability(-0.8, 2.01), 0)
        self.assertEqual(scene.walkability(0.8, 2.01), 0)
        self.assertAlmostEqual(scene.surface.height(-0.8, 2.25), 0.45)
        self.assertGreater(scene.surface.height(0.8, 2.25), 0.0)
~~~

- [ ] **Step 2: Run the procedural tests and verify the missing API failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.ProceduralSceneTests -v
~~~

Expected: import error for `procedural_scene_definitions`.

- [ ] **Step 3: Add reusable exact profile providers and scene helpers**

Add to `scenes.py` after the scene dataclasses:

~~~python
COURSE_HALF_WIDTH = 0.60
FLAT_SPAWN_LENGTH = 2.0
LOOKAHEAD_MARGIN = 1.0


@dataclass(frozen=True)
class LongitudinalProfileSurface:
    profile: Callable[[float], float]
    half_width: float = COURSE_HALF_WIDTH
    exterior_height: float = 0.0

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("profile query must be finite")
        if abs(x) > self.half_width:
            return self.exterior_height
        value = float(self.profile(z))
        if not np.isfinite(value):
            raise ValueError("profile height must be finite")
        return value


@dataclass(frozen=True)
class CrossSlopeSurface:
    angle_degrees: float
    grade_start_z: float
    grade_length: float = 4.0
    transition_length: float = 0.5
    half_width: float = COURSE_HALF_WIDTH

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("cross-slope query must be finite")
        if abs(x) > self.half_width:
            return 0.0
        phase = z - self.grade_start_z
        if phase <= 0.0 or phase >= self.grade_length:
            return 0.0
        factor = min(
            1.0,
            phase / self.transition_length,
            (self.grade_length - phase) / self.transition_length,
        )
        return x * np.tan(np.deg2rad(self.angle_degrees)) * factor


@dataclass(frozen=True)
class BlockedCourseSurface:
    wall_center_x: float = -0.8
    ramp_center_x: float = 0.8
    lane_half_width: float = 0.6
    obstacle_start_z: float = FLAT_SPAWN_LENGTH
    wall_height: float = 0.45
    wall_top_length: float = 0.50
    ramp_rise: float = 0.36
    ramp_angle_degrees: float = 25.0
    ramp_top_length: float = 0.50

    @property
    def ramp_run(self):
        return self.ramp_rise / np.tan(np.deg2rad(self.ramp_angle_degrees))

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("blocked-course query must be finite")
        if abs(x - self.wall_center_x) <= self.lane_half_width:
            if self.obstacle_start_z <= z \
                    <= self.obstacle_start_z + self.wall_top_length:
                return self.wall_height
        if abs(x - self.ramp_center_x) <= self.lane_half_width:
            phase = z - self.obstacle_start_z
            if 0.0 <= phase < self.ramp_run:
                return phase * np.tan(np.deg2rad(self.ramp_angle_degrees))
            if self.ramp_run <= phase <= self.ramp_run + self.ramp_top_length:
                return self.ramp_rise
        return 0.0


def _region(region_id, bounds):
    return {"id": region_id, "bounds_xz": [float(v) for v in bounds]}


def _corridor_definition(
    scene_id, label, surface, course_end_z, parameters, route,
    walkability_class,
):
    playable = (-COURSE_HALF_WIDTH, COURSE_HALF_WIDTH, 0.0, course_end_z)
    bounds = (
        -COURSE_HALF_WIDTH - LOOKAHEAD_MARGIN,
        COURSE_HALF_WIDTH + LOOKAHEAD_MARGIN,
        -LOOKAHEAD_MARGIN,
        course_end_z + LOOKAHEAD_MARGIN,
    )
    region_name = "certified" if walkability_class == 1 else "stress"
    regions = {"certified": (), "stress": (), "blocked": ()}
    regions[region_name] = (_region("course", playable),)
    return SceneDefinition(
        scene_id=scene_id,
        label=label,
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": parameters,
        },
        surface=surface,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions=regions,
        routes=(route,),
        walkability=lambda x, z, c=walkability_class, b=playable: (
            c if _bounds_contains(b, x, z) else 0),
    )
~~~

The `0.5 m` transitions are part of each `4.0 m` cross-slope segment; the
preceding `2.0 m` spawn area plus `1.0 m` entry and the following `1.0 m` exit
remain exactly flat. The route centerline stays at height zero while the support
plane changes laterally.

- [ ] **Step 4: Implement exact stairs and longitudinal ramps**

Add:

~~~python
def _stair_profile(rises, runs, landing_length=2.0):
    pairs = tuple(zip(tuple(rises), tuple(runs)))
    ascent_end = FLAT_SPAWN_LENGTH + sum(runs)
    descent_start = ascent_end + landing_length
    course_end = descent_start + sum(runs) + 1.0

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        cursor = FLAT_SPAWN_LENGTH
        height = 0.0
        for rise, run in pairs:
            height += rise
            if z < cursor + run:
                return height
            cursor += run
        if z < cursor + landing_length:
            return height
        cursor += landing_length
        for rise, run in reversed(pairs):
            if z < cursor + run:
                return height
            cursor += run
            height -= rise
        return 0.0

    return profile, ascent_end, descent_start, course_end


def _stair_definition(scene_id, label, rises, runs, unseen):
    profile, ascent_end, descent_start, course_end = _stair_profile(rises, runs)
    parameters = {
        "primitive": "stairs-up-landing-down",
        "rises_m": list(rises),
        "runs_m": list(runs),
        "width_m": 1.2,
        "landing_length_m": 2.0,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "unseen_geometry": unseen,
        "ascent_end_z_m": ascent_end,
        "descent_start_z_m": descent_start,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "ascent-landing-descent",
        ((0.0, 0.0), (0.0, 1.75),
         (0.0, ascent_end + 1.0),
         (0.0, descent_start + sum(runs) + 0.25),
         (0.0, course_end)),
        "traverse", 1, 2.0,
    )
    return _corridor_definition(
        scene_id, label, LongitudinalProfileSurface(profile),
        course_end, parameters, route, 1)


def _ramp_profile(angle_degrees):
    rise = 0.36
    run = rise / np.tan(np.deg2rad(angle_degrees))
    ascent_end = FLAT_SPAWN_LENGTH + run
    descent_start = ascent_end + 2.0
    course_end = descent_start + run + 1.0

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        if z < ascent_end:
            return (z - FLAT_SPAWN_LENGTH) * rise / run
        if z < descent_start:
            return rise
        if z < descent_start + run:
            return rise - (z - descent_start) * rise / run
        return 0.0

    return profile, run, ascent_end, descent_start, course_end


def _ramp_definition(scene_id, label, angle_degrees, stress):
    profile, run, ascent_end, descent_start, course_end = _ramp_profile(
        angle_degrees)
    walkability_class = 2 if stress else 1
    expected_outcome = "traverse-or-safe-stop" if stress else "traverse"
    parameters = {
        "primitive": "ramp-up-landing-down",
        "angle_degrees": float(angle_degrees),
        "rise_m": 0.36,
        "run_m": run,
        "width_m": 1.2,
        "landing_length_m": 2.0,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "ascent_end_z_m": ascent_end,
        "descent_start_z_m": descent_start,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "up-landing-down",
        ((0.0, 0.0), (0.0, 1.75),
         (0.0, ascent_end + 1.0),
         (0.0, descent_start + run + 0.25),
         (0.0, course_end)),
        expected_outcome, walkability_class, 2.0,
    )
    return _corridor_definition(
        scene_id, label, LongitudinalProfileSurface(profile),
        course_end, parameters, route, walkability_class)
~~~

- [ ] **Step 5: Implement cross-slopes, the mixed course, and blocked lanes**

Add:

~~~python
def _cross_slope_definition(angle_degrees):
    flat_entry_length = 1.0
    grade_start = FLAT_SPAWN_LENGTH + flat_entry_length
    course_end = grade_start + 4.0 + 1.0
    scene_id = f"cross-slope-{angle_degrees:02d}"
    parameters = {
        "primitive": "cross-slope",
        "angle_degrees": float(angle_degrees),
        "width_m": 1.2,
        "grade_length_m": 4.0,
        "transition_length_m": 0.5,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_entry_length_m": flat_entry_length,
        "flat_exit_length_m": 1.0,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "forward-cross-slope",
        ((0.0, 0.0), (0.0, 2.75), (0.0, 5.0),
         (0.0, 7.25), (0.0, course_end)),
        "traverse", 1, 0.0,
    )
    return _corridor_definition(
        scene_id, f"Cross Slope {angle_degrees} Degrees",
        CrossSlopeSurface(float(angle_degrees), grade_start), course_end,
        parameters, route, 1)


def _mixed_definition():
    stair_runs = (0.30, 0.30, 0.30, 0.30)
    stair_rises = (0.08, 0.08, 0.08, 0.08)
    ascent_end = FLAT_SPAWN_LENGTH + sum(stair_runs)
    elevated_end = ascent_end + 3.0
    changes = (0.08, -0.12, 0.04)
    block_starts = tuple(elevated_end + 0.60 * i for i in range(3))
    ramp_start = elevated_end + 3 * 0.60
    ramp_run = sum(stair_rises) / np.tan(np.deg2rad(10.0))
    course_end = ramp_start + ramp_run + 1.0

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        cursor = FLAT_SPAWN_LENGTH
        height = 0.0
        for rise, run in zip(stair_rises, stair_runs):
            height += rise
            if z < cursor + run:
                return height
            cursor += run
        if z < elevated_end:
            return height
        block_cursor = elevated_end
        for change in changes:
            height += change
            if z < block_cursor + 0.60:
                return height
            block_cursor += 0.60
        if z < ramp_start + ramp_run:
            return height * (1.0 - (z - ramp_start) / ramp_run)
        return 0.0

    parameters = {
        "primitive": "mixed-multilevel",
        "stair_rises_m": list(stair_rises),
        "stair_runs_m": list(stair_runs),
        "width_m": 1.2,
        "elevated_walk_length_m": 3.0,
        "block_height_changes_m": list(changes),
        "block_top_length_m": 0.60,
        "block_starts_z_m": list(block_starts),
        "return_ramp_angle_degrees": 10.0,
        "return_ramp_run_m": ramp_run,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "course_end_z_m": course_end,
    }
    route_points = [
        (0.0, 0.0), (0.0, 1.75), (0.0, ascent_end + 1.5),
    ]
    route_points.extend((0.0, start + 0.30) for start in block_starts)
    route_points.extend(((0.0, ramp_start + ramp_run), (0.0, course_end)))
    return _corridor_definition(
        "mixed-multilevel", "Mixed Multilevel Course",
        LongitudinalProfileSurface(profile), course_end, parameters,
        SceneRoute(
            "full-course", tuple(route_points), "traverse", 1, 2.0),
        1)


def _blocked_definition():
    surface = BlockedCourseSurface()
    obstacle_start = surface.obstacle_start_z
    course_end = obstacle_start + surface.ramp_run \
        + surface.ramp_top_length + 1.0
    playable = (-1.4, 1.4, 0.0, course_end)
    bounds = (-2.4, 2.4, -1.0, course_end + 1.0)

    def walkability(x, z):
        if not _bounds_contains(playable, x, z):
            return 0
        return 1 if z <= obstacle_start - SCENE_CELL_SIZE else 0

    parameters = {
        "primitive": "blocked-course",
        "lane_width_m": 1.2,
        "wall_center_x_m": surface.wall_center_x,
        "wall_height_m": surface.wall_height,
        "wall_top_length_m": surface.wall_top_length,
        "ramp_center_x_m": surface.ramp_center_x,
        "ramp_rise_m": surface.ramp_rise,
        "ramp_angle_degrees": surface.ramp_angle_degrees,
        "ramp_run_m": surface.ramp_run,
        "ramp_top_length_m": surface.ramp_top_length,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "course_end_z_m": course_end,
    }
    return SceneDefinition(
        scene_id="blocked-course",
        label="Blocked Wall And Ramp Course",
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": parameters,
        },
        surface=surface,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": (_region(
                "approach", (-1.4, 1.4, 0.0,
                             obstacle_start - SCENE_CELL_SIZE)),),
            "stress": (),
            "blocked": (
                _region("wall", (-1.4, -0.2, obstacle_start, course_end)),
                _region("ramp", (0.2, 1.4, obstacle_start, course_end)),
            ),
        },
        routes=(
            SceneRoute(
                "wall-safe-stop",
                ((0.0, 0.0), (-0.8, 1.5), (-0.8, 2.25)),
                "safe-stop", 0, 0.0),
            SceneRoute(
                "ramp-safe-stop",
                ((0.0, 0.0), (0.8, 1.5),
                 (0.8, obstacle_start + surface.ramp_run / 2.0)),
                "safe-stop", 0, 0.0),
        ),
        walkability=walkability,
    )
~~~

Finally add the ordered public factory:

~~~python
def procedural_scene_definitions():
    return (
        _stair_definition(
            "stairs-shallow", "Shallow Stairs",
            (0.08, 0.08, 0.08, 0.08),
            (0.30, 0.30, 0.30, 0.30), False),
        _stair_definition(
            "stairs-standard", "Standard Stairs",
            (0.12, 0.12, 0.12), (0.32, 0.32, 0.32), False),
        _stair_definition(
            "stairs-unseen-variable", "Unseen Variable Stairs",
            (0.06, 0.10, 0.08, 0.12),
            (0.24, 0.34, 0.28, 0.38), True),
        _ramp_definition(
            "ramp-05-up-down", "Ramp 5 Degrees Up And Down", 5, False),
        _ramp_definition(
            "ramp-10-up-down", "Ramp 10 Degrees Up And Down", 10, False),
        _ramp_definition(
            "ramp-15-stress", "Ramp 15 Degree Stress Case", 15, True),
        _cross_slope_definition(5),
        _cross_slope_definition(10),
        _mixed_definition(),
        _blocked_definition(),
    )
~~~

- [ ] **Step 6: Run all scene tests and inspect one serialized procedural pack**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import numpy as np

from resources.g1_terrain_builder.scenes import (
    build_scene, procedural_scene_definitions,
)
for definition in procedural_scene_definitions():
    scene = build_scene(definition)
    assert scene.metadata["heightfield"]["cell_size_m"] \
        == float(np.float32(0.02))
    assert scene.metadata["heightfield"]["sha256"]
    assert scene.metadata["walkability"]["sha256"]
print("VALID procedural-scenes=10")
PY
~~~

Expected: the scene suite passes and the inspection prints exactly
`VALID procedural-scenes=10`. The 15-degree route is class 2; both blocked
routes target class-0 cells after a class-1 approach.

- [ ] **Step 7: Commit the deterministic procedural catalog**

~~~bash
git add resources/g1_terrain_builder/scenes.py tests/python/test_scenes.py
git diff --cached --check
git commit -m "feat: generate deterministic G1 terrain courses"
~~~

### Task 10: Select, route, and parity-check the four exact GRAIL scenes

**Files:**
- Modify: `resources/g1_terrain_builder/terrain.py`
- Modify: `resources/g1_terrain_builder/scenes.py`
- Modify: `tests/python/test_terrain.py`
- Modify: `tests/python/test_scenes.py`

**Interfaces:**
- Produces: `select_grail_scene_bases(measured_max_heights) -> dict[str,str]`;
  ties are resolved with `min((absolute_error, base_name))`, never filesystem
  order.
- Produces: `grail_scene_definition(scene_id, base, clip,
  target_height_m) -> SceneDefinition` using the matching converted Holden root
  path and root facing.
- Produces: `grail_scene_definitions(measured_max_heights,
  clips_by_terrain) -> tuple[SceneDefinition, ...]` and
  `all_scene_definitions(measured_max_heights, clips_by_terrain)
  -> tuple[SceneDefinition, ...]`.
- Produces: `grail_surface_parity(terrain, grid) -> dict`; it independently
  reports grid-node, within-cell, away-from-edge source, and top-footprint edge
  errors.
- Locks measured corpus choices to default `terrain_curbs__curb_000__000`, low
  `terrain_curbs__curb_186__004`, medium
  `terrain_curbs__curb_022__001`, and high
  `terrain_curbs__curb_165__006` for the source corpus in this repository.

- [ ] **Step 1: Write failing lexical-selection and converted-route tests**

Extend imports in `tests/python/test_scenes.py` with `glob`, `os`,
`select_grail_scene_bases`, `grail_scene_definition`,
`grail_scene_definitions`, and `all_scene_definitions`. Add:

~~~python
GRAIL_ROBOT_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/robot"


def fake_grail_clip(base):
    clip = HoldenClip.empty(frames=5, bones=31)
    clip.name = base + "-clip"
    clip.terrain_id = base
    clip.positions[:, 0, 0] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    clip.positions[:, 0, 2] = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
    return clip


class GrailSceneTests(unittest.TestCase):
    def test_nearest_height_selection_uses_lexical_tie_break(self):
        measured = {
            "terrain_curbs__curb_000__000": 0.29,
            "a-low-tie": 0.13,
            "z-low-tie": 0.13,
            "medium": 0.241,
            "high": 0.358,
        }
        selected = select_grail_scene_bases(measured)
        self.assertEqual(selected, {
            "grail-curb-default": "terrain_curbs__curb_000__000",
            "grail-curb-low": "a-low-tie",
            "grail-curb-medium": "medium",
            "grail-curb-high": "high",
        })

    def test_grail_scene_uses_matching_converted_root_path_and_facing(self):
        base = "terrain_curbs__curb_000__000"
        clip = fake_grail_clip(base)
        scene = grail_scene_definition(
            "grail-curb-default", base, clip, None)
        self.assertEqual(scene.spawn_position, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(scene.spawn_yaw_radians, 0.0)
        self.assertEqual(scene.routes[0].route_id, "curb-forward")
        self.assertEqual(scene.routes[0].waypoints_xz[0], (0.0, 0.0))
        self.assertEqual(scene.routes[0].waypoints_xz[-1], (0.4, 1.6))
        self.assertEqual(scene.provenance["source_ids"], [base, clip.name])
        wrong = fake_grail_clip("different-base")
        with self.assertRaisesRegex(ValueError, "matching converted clip"):
            grail_scene_definition(
                "grail-curb-default", base, wrong, None)

    def test_real_corpus_selection_is_stable(self):
        paths = sorted(glob.glob(os.path.join(GRAIL_ROBOT_DIR, "*.pkl")))
        self.assertEqual(len(paths), 1769)
        measured = {}
        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            measured[base] = GrailTerrain.from_base(base).footprint()["height"]
        self.assertEqual(select_grail_scene_bases(measured), {
            "grail-curb-default": "terrain_curbs__curb_000__000",
            "grail-curb-low": "terrain_curbs__curb_186__004",
            "grail-curb-medium": "terrain_curbs__curb_022__001",
            "grail-curb-high": "terrain_curbs__curb_165__006",
        })

    def test_full_definition_catalog_has_locked_order_and_classes(self):
        selected = {
            "grail-curb-default": "terrain_curbs__curb_000__000",
            "grail-curb-low": "terrain_curbs__curb_186__004",
            "grail-curb-medium": "terrain_curbs__curb_022__001",
            "grail-curb-high": "terrain_curbs__curb_165__006",
        }
        measured = {
            selected["grail-curb-default"]: 0.292,
            selected["grail-curb-low"]: 0.122,
            selected["grail-curb-medium"]: 0.240,
            selected["grail-curb-high"]: 0.360,
        }
        clips = {
            base: fake_grail_clip(base) for base in selected.values()
        }
        definitions = all_scene_definitions(measured, clips)
        self.assertEqual(
            tuple(scene.scene_id for scene in definitions), REQUIRED_SCENE_IDS)
        grail = definitions[:4]
        self.assertEqual(
            [(scene.routes[0].expected_outcome,
              scene.routes[0].walkability_class) for scene in grail],
            [("traverse-or-safe-stop", 2), ("traverse", 1),
             ("traverse-or-safe-stop", 2),
             ("traverse-or-safe-stop", 2)])
~~~

Also add these imports at the top of the test file:

~~~python
from resources.g1_terrain_builder.schema import HoldenClip
from resources.g1_terrain_builder.terrain import GrailTerrain
~~~

- [ ] **Step 2: Run selection tests and verify the missing functions fail**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.GrailSceneTests.test_nearest_height_selection_uses_lexical_tie_break \
  tests.python.test_scenes.GrailSceneTests.test_grail_scene_uses_matching_converted_root_path_and_facing -v
~~~

Expected: import errors for `select_grail_scene_bases` and
`grail_scene_definition`. Do not run the 1,769-mesh integration test until the
small RED/GREEN cycle passes.

- [ ] **Step 3: Implement stable GRAIL selection and converted-clip routes**

Import Holden quaternion helpers in `scenes.py`:

~~~python
from resources import quat as holden_quat
~~~

Then add:

~~~python
GRAIL_DEFAULT_BASE = "terrain_curbs__curb_000__000"
GRAIL_TARGETS = (
    ("grail-curb-low", 0.12),
    ("grail-curb-medium", 0.24),
    ("grail-curb-high", 0.36),
)


def select_grail_scene_bases(measured_max_heights):
    if not isinstance(measured_max_heights, dict) or not measured_max_heights:
        raise ValueError("GRAIL measurements must be a non-empty mapping")
    measured = {}
    for base, height in measured_max_heights.items():
        if not isinstance(base, str) or not base or not np.isfinite(height):
            raise ValueError("GRAIL base names and measured heights are invalid")
        measured[base] = float(height)
    if GRAIL_DEFAULT_BASE not in measured:
        raise ValueError(f"missing default GRAIL base {GRAIL_DEFAULT_BASE}")
    result = {"grail-curb-default": GRAIL_DEFAULT_BASE}
    for scene_id, target in GRAIL_TARGETS:
        result[scene_id] = min(
            measured,
            key=lambda base: (abs(measured[base] - target), base),
        )
    return result


def _root_route_and_yaw(clip):
    positions = np.asarray(clip.positions, np.float64)
    rotations = np.asarray(clip.rotations, np.float64)
    if positions.ndim != 3 or positions.shape[0] < 2 \
            or positions.shape[1] < 1 or positions.shape[2] != 3:
        raise ValueError("converted GRAIL clip has invalid root positions")
    if rotations.shape[:2] != positions.shape[:2] \
            or rotations.shape[2] != 4 \
            or not np.isfinite(positions).all() \
            or not np.isfinite(rotations).all():
        raise ValueError("converted GRAIL root transform is invalid")
    indices = sorted(set(
        int(round(value))
        for value in np.linspace(0, len(positions) - 1, 5)
    ))
    if len(indices) < 2:
        raise ValueError("converted GRAIL route needs two distinct frames")
    path = positions[:, 0][:, (0, 2)]
    route = tuple(
        (float(path[index, 0]), float(path[index, 1]))
        for index in indices
    )
    facing = holden_quat.mul_vec(
        rotations[0, 0], np.array([0.0, 0.0, 1.0], np.float64))
    horizontal = np.array([facing[0], facing[2]], np.float64)
    if not np.isfinite(horizontal).all() or np.linalg.norm(horizontal) < 1e-8:
        raise ValueError("converted GRAIL root facing is invalid")
    yaw = float(np.arctan2(horizontal[0], horizontal[1]))
    spawn = tuple(float(value) for value in positions[0, 0])
    return path, route, spawn, yaw


def grail_scene_definition(
    scene_id, base, clip, target_height_m,
):
    if scene_id not in REQUIRED_SCENE_IDS[:4]:
        raise ValueError(f"unknown GRAIL scene ID {scene_id}")
    if clip.terrain_id != base:
        raise ValueError("GRAIL scene requires its matching converted clip")
    terrain = GrailTerrain.from_base(base)
    maximum_height = float(terrain.footprint()["height"])
    path, route_points, spawn, yaw = _root_route_and_yaw(clip)
    mesh_xmin, mesh_xmax, mesh_zmin, mesh_zmax = terrain.xz_bounds()
    path_xmin, path_zmin = path.min(axis=0)
    path_xmax, path_zmax = path.max(axis=0)
    playable = (
        float(path_xmin - COURSE_HALF_WIDTH),
        float(path_xmax + COURSE_HALF_WIDTH),
        float(path_zmin - COURSE_HALF_WIDTH),
        float(path_zmax + COURSE_HALF_WIDTH),
    )
    bounds = (
        float(min(mesh_xmin, playable[0]) - LOOKAHEAD_MARGIN),
        float(max(mesh_xmax, playable[1]) + LOOKAHEAD_MARGIN),
        float(min(mesh_zmin, playable[2]) - LOOKAHEAD_MARGIN),
        float(max(mesh_zmax, playable[3]) + LOOKAHEAD_MARGIN),
    )
    certified = maximum_height <= 0.16
    walkability_class = 1 if certified else 2
    expected_outcome = "traverse" if certified else "traverse-or-safe-stop"
    region_name = "certified" if certified else "stress"
    regions = {"certified": (), "stress": (), "blocked": ()}
    regions[region_name] = (_region("curb-route", playable),)
    labels = {
        "grail-curb-default": "GRAIL Default Curb",
        "grail-curb-low": "GRAIL Low Curb",
        "grail-curb-medium": "GRAIL Medium Curb",
        "grail-curb-high": "GRAIL High Curb",
    }
    return SceneDefinition(
        scene_id=scene_id,
        label=labels[scene_id],
        provenance={
            "kind": "grail",
            "source_ids": [base, clip.name],
            "parameters": {
                "selection_rule": (
                    "fixed-default" if target_height_m is None
                    else "nearest-measured-maximum-then-lexical"),
                "target_height_m": (
                    None if target_height_m is None
                    else float(target_height_m)),
                "measured_maximum_height_m": maximum_height,
                "route_source": "converted-holden-root-path",
            },
        },
        surface=terrain,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=spawn,
        spawn_yaw_radians=yaw,
        regions=regions,
        routes=(SceneRoute(
            "curb-forward", route_points, expected_outcome,
            walkability_class, 0.0),),
        walkability=lambda x, z, c=walkability_class, b=playable: (
            c if _bounds_contains(b, x, z) else 0),
    )


def grail_scene_definitions(measured_max_heights, clips_by_terrain):
    selected = select_grail_scene_bases(measured_max_heights)
    targets = {scene_id: target for scene_id, target in GRAIL_TARGETS}
    targets["grail-curb-default"] = None
    definitions = []
    for scene_id in REQUIRED_SCENE_IDS[:4]:
        base = selected[scene_id]
        if base not in clips_by_terrain:
            raise ValueError(f"missing converted scene clip for {base}")
        definitions.append(grail_scene_definition(
            scene_id, base, clips_by_terrain[base], targets[scene_id]))
    return tuple(definitions)


def all_scene_definitions(measured_max_heights, clips_by_terrain):
    definitions = grail_scene_definitions(
        measured_max_heights, clips_by_terrain) \
        + procedural_scene_definitions()
    if tuple(scene.scene_id for scene in definitions) != REQUIRED_SCENE_IDS:
        raise ValueError("complete scene definition order changed")
    return definitions
~~~

- [ ] **Step 4: Write failing real-surface parity tests**

Import `grail_surface_parity` and `rasterize_heightfield` in
`tests/python/test_terrain.py`, then add:

~~~python
    def test_default_grail_grid_meets_all_surface_parity_tolerances(self):
        terrain = GrailTerrain.from_base(builder.DEFAULTS["runtime_terrain"])
        xmin, xmax, zmin, zmax = terrain.xz_bounds()
        grid = rasterize_heightfield(
            terrain, (xmin - 1.0, xmax + 1.0,
                      zmin - 1.0, zmax + 1.0), 0.02)
        report = grail_surface_parity(terrain, grid)
        self.assertLessEqual(report["node_error_m"], 1e-6)
        self.assertLessEqual(report["within_cell_error_m"], 1e-4)
        self.assertLessEqual(report["source_away_edge_error_m"], 0.005)
        self.assertLessEqual(report["top_edge_movement_m"], 0.02 + 1e-9)
        self.assertGreater(report["away_edge_probe_count"], 0)

    def test_real_grail_obj_vertices_and_faces_are_the_grid(self):
        terrain = GrailTerrain.from_base(builder.DEFAULTS["runtime_terrain"])
        xmin, xmax, zmin, zmax = terrain.xz_bounds()
        grid = rasterize_heightfield(
            terrain, (xmin - 0.04, xmax + 0.04,
                      zmin - 0.04, zmax + 0.04), 0.02)
        lines = grid.obj_bytes().decode("utf-8").splitlines()
        vertex_lines = lines[:grid.nx * grid.nz]
        face_lines = lines[grid.nx * grid.nz:]
        vertices = np.array([
            [float(value) for value in line.split()[1:]]
            for line in vertex_lines
        ], dtype=np.float32)
        def runtime_coordinate(value):
            encoded = np.float32(value)
            return np.float32(0.0) if encoded == 0.0 else encoded
        expected = []
        for iz in range(grid.nz):
            for ix in range(grid.nx):
                expected.append([
                    runtime_coordinate(
                        grid.origin_x + ix * grid.cell_size),
                    grid.heights[iz, ix],
                    runtime_coordinate(
                        grid.origin_z + iz * grid.cell_size),
                ])
        expected = np.asarray(expected, dtype=np.float32)
        np.testing.assert_array_equal(
            vertices.view(np.uint32), expected.view(np.uint32))
        expected_faces = []
        for iz in range(grid.nz - 1):
            for ix in range(grid.nx - 1):
                p00 = iz * grid.nx + ix + 1
                p10, p01, p11 = p00 + 1, p00 + grid.nx, p00 + grid.nx + 1
                expected_faces.extend(
                    [f"f {p00} {p11} {p10}", f"f {p00} {p01} {p11}"])
        self.assertEqual(face_lines, expected_faces)
~~~

- [ ] **Step 5: Run parity tests and verify the missing report API**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain.TerrainTests.test_default_grail_grid_meets_all_surface_parity_tolerances \
  tests.python.test_terrain.TerrainTests.test_real_grail_obj_vertices_and_faces_are_the_grid -v
~~~

Expected: import error for `grail_surface_parity`; the OBJ test may already
pass because Task 4 established the common grid.

- [ ] **Step 6: Implement deterministic node, triangle, source, and edge metrics**

Add to `terrain.py` after `rasterize_heightfield`:

~~~python
def _fixed_triangle_height(grid, ix, iz, tx, tz):
    h00 = float(grid.heights[iz, ix])
    h10 = float(grid.heights[iz, ix + 1])
    h01 = float(grid.heights[iz + 1, ix])
    h11 = float(grid.heights[iz + 1, ix + 1])
    if tx >= tz:
        return h00 + tx * (h10 - h00) + tz * (h11 - h10)
    return h00 + tx * (h11 - h01) + tz * (h01 - h00)


def grail_surface_parity(terrain, grid):
    if not isinstance(terrain, GrailTerrain) or not isinstance(grid, HeightGrid):
        raise TypeError("GRAIL parity requires GrailTerrain and HeightGrid")
    source_nodes = np.empty_like(grid.heights, dtype=np.float64)
    for iz in range(grid.nz):
        z = grid.origin_z + iz * grid.cell_size
        for ix in range(grid.nx):
            x = grid.origin_x + ix * grid.cell_size
            source_nodes[iz, ix] = terrain.height(x, z)
    node_error = float(np.max(np.abs(
        source_nodes - grid.heights.astype(np.float64))))

    cell_count = (grid.nx - 1) * (grid.nz - 1)
    sample_count = min(cell_count, 4096)
    linear_cells = np.unique(np.linspace(
        0, cell_count - 1, sample_count, dtype=np.int64))
    local_probes = ((0.25, 0.125), (0.75, 0.25),
                    (0.25, 0.75), (0.75, 0.875))
    within_cell_error = 0.0
    source_away_error = 0.0
    away_edge_probe_count = 0
    for linear in linear_cells:
        iz, ix = divmod(int(linear), grid.nx - 1)
        corner_heights = [
            float(grid.heights[iz, ix]),
            float(grid.heights[iz, ix + 1]),
            float(grid.heights[iz + 1, ix]),
            float(grid.heights[iz + 1, ix + 1]),
        ]
        probes = []
        for tx, tz in local_probes:
            x = grid.origin_x + (ix + tx) * grid.cell_size
            z = grid.origin_z + (iz + tz) * grid.cell_size
            queried = grid.height(x, z)
            explicit = _fixed_triangle_height(grid, ix, iz, tx, tz)
            within_cell_error = max(
                within_cell_error, abs(queried - explicit))
            probes.append((queried, terrain.height(x, z)))
        local_source = corner_heights + [source for _, source in probes]
        if max(local_source) - min(local_source) <= 0.005:
            for queried, source in probes:
                source_away_error = max(
                    source_away_error, abs(queried - source))
                away_edge_probe_count += 1

    footprint = terrain.footprint()
    threshold = footprint["height"] - 0.02
    top_iz, top_ix = np.nonzero(grid.heights >= threshold)
    if not len(top_ix):
        raise ValueError("GRAIL grid contains no measured top nodes")
    grid_top_bounds = (
        grid.origin_x + int(top_ix.min()) * grid.cell_size,
        grid.origin_x + int(top_ix.max()) * grid.cell_size,
        grid.origin_z + int(top_iz.min()) * grid.cell_size,
        grid.origin_z + int(top_iz.max()) * grid.cell_size,
    )
    source_top_bounds = (
        footprint["x"][0], footprint["x"][1],
        footprint["z"][0], footprint["z"][1],
    )
    edge_movement = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(grid_top_bounds, source_top_bounds)
    )
    return {
        "node_error_m": node_error,
        "within_cell_error_m": float(within_cell_error),
        "source_away_edge_error_m": float(source_away_error),
        "top_edge_movement_m": float(edge_movement),
        "away_edge_probe_count": away_edge_probe_count,
    }
~~~

This function deliberately compares the exact source provider only in cells
whose source corners and deterministic probes span no more than `0.005 m`;
edge location is checked separately from the `height >= max_height - 0.02`
top footprint.

- [ ] **Step 7: Run the complete terrain/scene suites and the corpus selector**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_terrain tests.python.test_scenes -v
~~~

Expected: all tests pass, including the scan of exactly 1,769 GRAIL candidates;
the four measured base names match the locked choices and every parity metric
meets its independent tolerance.

- [ ] **Step 8: Commit exact GRAIL scene selection and parity**

~~~bash
git add resources/g1_terrain_builder/terrain.py \
  resources/g1_terrain_builder/scenes.py \
  tests/python/test_terrain.py tests/python/test_scenes.py
git diff --cached --check
git commit -m "feat: build exact GRAIL curb scenes"
~~~

### Task 11: Assemble the v2 manifest and atomically publish the complete tree

**Files:**
- Modify: `resources/g1_terrain_builder/artifacts.py`
- Modify: `resources/build_g1_terrain_database.py`
- Modify: `tests/python/test_artifacts.py`
- Modify: `tests/python/test_build_cli.py`

**Interfaces:**
- Changes: `publish_artifacts(output_dir, artifacts, manifest_base, scene_pack,
  validate_candidate) -> dict`; the return value is the finalized manifest
  containing hashes of the exact staged bytes.
- Produces: `sha256_file(path) -> str`, `_write_scene_pack(staging, pack)`,
  `_validate_staged_artifacts(staging, artifacts, manifest, scene_pack)`, and
  `_fsync_tree(path)`.
- The builder always creates all 14 scenes. `--grail-limit` limits motion clips
  only; it does not shrink the candidate set used for low/medium/high selection
  or the published scene catalog.
- Removes obsolete top-level `terrain.bin` and `terrain.obj`; terrain is always
  selected through `scenes/index.json` and a scene directory.
- Publication writes only a unique sibling staging directory and unique sibling
  backup, validates and fsyncs staging before rename, restores the prior output
  on a rename failure, and removes only scratch paths it created itself.

The manifest top-level keys are exact:

~~~text
schema, output_fps, feature_dimensions, terrain_dimensions,
support_dimensions, terrain_feature_distances_m, total_clips, grail_clips,
skipped_clips, database_frames, diagnostic_mode, sources, skeleton, contact,
surface, database, sidecars, scene_index, validation_file, validation
~~~

The new objects are exact:

~~~json
{
  "schema": "g1-terrain-artifacts/v2",
  "surface": {
    "semantics": {
      "schema": "g1-terrain-surface/v1",
      "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
      "source_query": "vertical-triangle-top",
      "polygon_triangulation": "fan-from-first-index",
      "overlap_height_policy": "maximum-y",
      "projected_boundary_policy": "closed",
      "triangle_winding_policy": "orientation-independent",
      "degenerate_projected_triangle_policy": "ignore",
      "projected_area_measure": "absolute-two-times-area",
      "bbox_tolerance_m": 1e-12,
      "projected_area_epsilon_m2": 1e-12,
      "barycentric_tolerance": 1e-10,
      "heightfield_schema": "G1HF/v2",
      "heightfield_version": 2,
      "heightfield_interpolation": "fixed-diagonal-triangles",
      "heightfield_diagonal": "min-x-min-z_to_max-x-max-z",
      "heightfield_scalar_encoding": "ieee754-binary32-little-endian",
      "heightfield_domain_policy": "inclusive-authoritative-node-rectangle",
      "heightfield_grid_line_policy": "positive-index-cell-except-maximum-edge",
      "heightfield_diagonal_tie_policy": "tx-greater-or-equal-tz-uses-p00-p10-p11",
      "heightfield_exterior_normal": [0.0, 1.0, 0.0],
      "heightfield_source_node_encoding": "binary32-header-values-promoted-to-binary64-arithmetic",
      "heightfield_runtime_query_encoding": "normal-or-zero-binary32-canonicalized-positive-and-promoted-to-binary64",
      "heightfield_scalar_domain": "normal-or-zero-binary32",
      "heightfield_cell_domain": "positive-normal-binary32",
      "heightfield_runtime_node_domain": "normal-or-zero-binary32",
      "heightfield_runtime_query_domain": "normal-or-zero-binary32-coordinates",
      "heightfield_runtime_parity_domain": "normal-or-zero-binary32-coordinates",
      "heightfield_denormal_policy": "reject-nonzero-binary32-subnormals",
      "heightfield_evaluation_precision": "binary64-from-binary32-samples-and-promoted-node-weights",
      "heightfield_runtime_height_output": "finite-binary64-interpolation-rounded-to-binary32",
      "heightfield_normal_evaluation": "selected-triangle-binary64-gradient-scale-safe-unit-normalization",
      "heightfield_runtime_normal_output": "unit-normal-components-rounded-to-binary32",
      "heightfield_runtime_output_ftz_policy": "binary32-subnormals-and-signed-zero-canonicalized-to-positive-zero",
      "heightfield_zero_encoding": "canonical-positive-zero",
      "heightfield_runtime_node_distinguishability_policy": "normal-or-positive-zero-strictly-increasing-proven-by-endpoints-near-zero-candidates-max-binary32-spacing-and-aligned-equality",
      "heightfield_obj_coordinate_quantization": "binary32-round-of-promoted-origin-plus-index-times-cell",
      "heightfield_raster_bounds_policy": "float32-minimum-rounded-down-and-maximum-ceil-covered",
      "heightfield_obj_vertex_order": "z-major-x-minor",
      "heightfield_obj_face_order": "p00-p11-p10_then_p00-p01-p11",
      "heightfield_obj_float_format": ".9g-final-newline",
      "cell_size_m": 0.02,
      "exterior_height_m": 0.0
    },
    "signature": "64 lowercase hex: SHA-256 of canonical compact semantics JSON"
  },
  "database": {
    "path": "database.bin",
    "schema": "holden-database/v1",
    "sha256": "64 lowercase hex: SHA-256 of database.bin"
  },
  "sidecars": {
    "terrain_features": {
      "path": "terrain_features.bin",
      "schema": "G1TF/v1",
      "version": 1,
      "dimensions": 4,
      "sha256": "64 lowercase hex: SHA-256 of terrain_features.bin"
    },
    "terrain_support": {
      "path": "terrain_support.bin",
      "schema": "G1SP/v1",
      "version": 1,
      "dimensions": 3,
      "columns": [
        "source_root_height_m",
        "source_left_toe_height_m",
        "source_right_toe_height_m"
      ],
      "sha256": "64 lowercase hex: SHA-256 of terrain_support.bin"
    }
  },
  "scene_index": {
    "path": "scenes/index.json",
    "schema": "g1-terrain-scene-index/v1",
    "sha256": "64 lowercase hex: SHA-256 of scenes/index.json"
  },
  "validation_file": {
    "path": "validation.json",
    "schema": "g1-terrain-validation/v1",
    "sha256": "64 lowercase hex: SHA-256 of validation.json"
  }
}
~~~

`validation` is the exact object written to `validation.json`; its exact keys
are `schema`, `duration_error_s`, `fk_max_error_m`, and
`quaternion_norm_max_error`. All JSON hashes use sorted, two-space-indented,
newline-terminated bytes. `manifest.json` is not self-hashed.

- [ ] **Step 1: Replace legacy top-level-terrain publication tests with full-tree tests**

In `tests/python/test_artifacts.py`, delete `_terrain_writer` and every legacy
publication test that expects root `terrain.bin`/`terrain.obj`. Extend imports
with `hashlib`, `FlatTerrain`, scene dataclasses/builders, support readers, and
`surface_semantics`/`surface_semantics_signature`. Add these fixtures:

~~~python
def tiny_scene_pack():
    definitions = []
    for scene_id in REQUIRED_SCENE_IDS:
        definitions.append(SceneDefinition(
            scene_id=scene_id,
            label=scene_id,
            provenance={
                "kind": "procedural", "source_ids": [],
                "parameters": {"fixture": True},
            },
            surface=FlatTerrain(),
            heightfield_bounds_xz=(-0.02, 0.02, -0.02, 0.02),
            playable_bounds_xz=(-0.01, 0.01, -0.01, 0.01),
            lookahead_bounds_xz=(-0.02, 0.02, -0.02, 0.02),
            spawn_position=(0.0, 0.0, 0.0),
            spawn_yaw_radians=0.0,
            regions={
                "certified": ({
                    "id": "fixture", "bounds_xz": [-0.01, 0.01, -0.01, 0.01],
                },),
                "stress": (), "blocked": (),
            },
            routes=(SceneRoute(
                "fixture", ((0.0, -0.01), (0.0, 0.01)),
                "traverse", 1, 0.0),),
            walkability=lambda x, z: 1,
        ))
    return build_scene_pack(definitions)


def tiny_manifest_base(artifacts):
    return {
        "schema": "g1-terrain-artifacts/v2",
        "output_fps": 25.0,
        "feature_dimensions": 31,
        "terrain_dimensions": 4,
        "support_dimensions": 3,
        "terrain_feature_distances_m": [0.25, 0.5, 0.75, 1.0],
        "total_clips": 1,
        "grail_clips": 0,
        "skipped_clips": 0,
        "database_frames": len(artifacts.positions),
        "diagnostic_mode": True,
        "sources": [{
            "name": "fixture", "terrain_id": "flat", "source_fps": 25.0,
            "source_frames": len(artifacts.positions),
            "output_frames": len(artifacts.positions), "range_start": 0,
            "range_stop": len(artifacts.positions),
            "source_frame_map": list(range(len(artifacts.positions))),
        }],
        "skeleton": {
            "names": ["Simulation", "Hips"], "parents": [-1, 0],
            "signature": "0" * 64,
        },
        "contact": {
            "speed_threshold": 0.15, "height_threshold": 0.06,
            "median_filter_frames": 3,
        },
        "surface": {
            "semantics": surface_semantics(),
            "signature": surface_semantics_signature(),
        },
        "validation": {
            "schema": "g1-terrain-validation/v1",
            "duration_error_s": [0.0], "fk_max_error_m": [0.0],
            "quaternion_norm_max_error": [0.0],
        },
    }


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()
~~~

Add the publication tests:

~~~python
    def test_publish_writes_exact_complete_tree_and_final_hash_manifest(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_features[:] = np.arange(16).reshape(4, 4)
        artifacts.terrain_support[:] = np.arange(12).reshape(4, 3)
        pack = tiny_scene_pack()
        validated = []
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            with open(os.path.join(output, "old"), "w") as stream:
                stream.write("last-good")
            manifest = publish_artifacts(
                output, artifacts, tiny_manifest_base(artifacts), pack,
                lambda path: validated.append(path),
            )
            self.assertEqual(validated, [mock.ANY])
            self.assertEqual(set(os.listdir(output)), {
                "database.bin", "terrain_features.bin", "terrain_support.bin",
                "manifest.json", "validation.json", "scenes",
            })
            self.assertEqual(
                sorted(os.listdir(os.path.join(output, "scenes"))),
                ["index.json", *sorted(REQUIRED_SCENE_IDS)])
            for scene_id in REQUIRED_SCENE_IDS:
                self.assertEqual(set(os.listdir(
                    os.path.join(output, "scenes", scene_id))), {
                    "scene.json", "terrain.bin", "terrain.obj",
                    "walkability.bin",
                })
            self.assertEqual(
                manifest["database"]["sha256"],
                file_sha256(os.path.join(output, "database.bin")))
            self.assertEqual(
                manifest["sidecars"]["terrain_features"]["sha256"],
                file_sha256(os.path.join(output, "terrain_features.bin")))
            self.assertEqual(
                manifest["sidecars"]["terrain_support"]["sha256"],
                file_sha256(os.path.join(output, "terrain_support.bin")))
            self.assertEqual(
                manifest["scene_index"]["sha256"],
                file_sha256(os.path.join(output, "scenes", "index.json")))
            self.assertEqual(
                manifest["validation_file"]["sha256"],
                file_sha256(os.path.join(output, "validation.json")))
            with open(os.path.join(output, "manifest.json")) as stream:
                self.assertEqual(json.load(stream), manifest)
            self.assertFalse(any(
                name.startswith(".published.") for name in os.listdir(temporary)))

    def test_candidate_failure_preserves_every_byte_of_previous_output(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            with open(sentinel, "wb") as stream:
                stream.write(b"last-good")
            with self.assertRaisesRegex(ValueError, "injected candidate"):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts), pack,
                    lambda path: (_ for _ in ()).throw(
                        ValueError("injected candidate failure")),
                )
            with open(sentinel, "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertEqual(os.listdir(output), ["old"])
            self.assertFalse(any(
                name.startswith(".published.") for name in os.listdir(temporary)))

    def test_publish_rename_failure_restores_previous_directory(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            with open(sentinel, "wb") as stream:
                stream.write(b"last-good")
            failed = False

            def replace(source, destination):
                nonlocal failed
                if destination == output and ".published.staging-" in source \
                        and not failed:
                    failed = True
                    raise OSError("injected publish rename failure")
                return real_replace(source, destination)

            with mock.patch(
                "resources.g1_terrain_builder.artifacts.os.replace",
                side_effect=replace,
            ):
                with self.assertRaisesRegex(OSError, "injected publish"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        lambda path: None,
                    )
            with open(sentinel, "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertEqual(os.listdir(output), ["old"])

    def test_publish_rejects_scene_byte_corruption_before_rename(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_write = artifacts_module._write_scene_pack
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            open(sentinel, "wb").write(b"last-good")

            def corrupt(staging, scene_pack):
                real_write(staging, scene_pack)
                path = os.path.join(
                    staging, "scenes", REQUIRED_SCENE_IDS[0], "terrain.bin")
                with open(path, "ab") as stream:
                    stream.write(b"corrupt")

            with mock.patch.object(
                artifacts_module, "_write_scene_pack", side_effect=corrupt,
            ):
                with self.assertRaisesRegex(ValueError, "scene bytes"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        lambda path: None,
                    )
            self.assertEqual(open(sentinel, "rb").read(), b"last-good")
~~~

Import the module itself as
`from resources.g1_terrain_builder import artifacts as artifacts_module` so the
corruption injection patches the implementation symbol.

- [ ] **Step 2: Run the publication tests and verify the old signature/tree fail**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts.ArtifactTests.test_publish_writes_exact_complete_tree_and_final_hash_manifest \
  tests.python.test_artifacts.ArtifactTests.test_candidate_failure_preserves_every_byte_of_previous_output \
  tests.python.test_artifacts.ArtifactTests.test_publish_rename_failure_restores_previous_directory -v
~~~

Expected: failures because `publish_artifacts` still accepts a terrain-writer
callback, returns `None`, and emits the obsolete root terrain files.

- [ ] **Step 3: Implement canonical writes, exact file-tree checks, and manifest finalization**

In `artifacts.py`, add these imports and constants:

~~~python
import hashlib
import re


MOTION_MANIFEST_KEYS = {
    "schema", "output_fps", "feature_dimensions", "terrain_dimensions",
    "support_dimensions", "terrain_feature_distances_m", "total_clips",
    "grail_clips", "skipped_clips", "database_frames", "diagnostic_mode",
    "sources", "skeleton", "contact", "surface", "database", "sidecars",
    "scene_index", "validation_file", "validation",
}
MANIFEST_BASE_KEYS = MOTION_MANIFEST_KEYS - {
    "database", "sidecars", "scene_index", "validation_file",
}
SCENE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")


def canonical_json_bytes(value):
    return (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_fsync(path, payload):
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
~~~

Replace `_normalized_manifest` with exact base validation and add pack writing:

~~~python
def _normalized_manifest_base(manifest):
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_BASE_KEYS:
        raise ValueError(
            f"manifest base keys must be {sorted(MANIFEST_BASE_KEYS)}")
    normalized = json.loads(canonical_json_bytes(manifest))
    if normalized["schema"] != "g1-terrain-artifacts/v2":
        raise ValueError("manifest schema must be g1-terrain-artifacts/v2")
    validation = normalized["validation"]
    if not isinstance(validation, dict) or set(validation) != {
        "schema", "duration_error_s", "fk_max_error_m",
        "quaternion_norm_max_error",
    } or validation["schema"] != "g1-terrain-validation/v1":
        raise ValueError("manifest validation object has invalid schema or keys")
    return normalized


def _write_scene_pack(staging, scene_pack):
    scenes_root = os.path.join(staging, "scenes")
    os.mkdir(scenes_root)
    index_ids = tuple(scene_pack.index.get("scene_ids", ()))
    scene_ids = tuple(scene.scene_id for scene in scene_pack.scenes)
    if index_ids != scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene index and scene bytes have different IDs")
    _write_bytes_fsync(
        os.path.join(scenes_root, "index.json"), scene_pack.index_json)
    for scene in scene_pack.scenes:
        if not SCENE_ID_PATTERN.fullmatch(scene.scene_id):
            raise ValueError(f"unsafe scene ID {scene.scene_id!r}")
        scene_dir = os.path.join(scenes_root, scene.scene_id)
        os.mkdir(scene_dir)
        _write_bytes_fsync(
            os.path.join(scene_dir, "scene.json"), scene.scene_json)
        _write_bytes_fsync(
            os.path.join(scene_dir, "terrain.bin"), scene.terrain_bin)
        _write_bytes_fsync(
            os.path.join(scene_dir, "terrain.obj"), scene.terrain_obj)
        _write_bytes_fsync(
            os.path.join(scene_dir, "walkability.bin"),
            scene.walkability_bin)


def _finalize_manifest(staging, manifest_base):
    manifest = dict(manifest_base)
    manifest["database"] = {
        "path": "database.bin", "schema": "holden-database/v1",
        "sha256": sha256_file(os.path.join(staging, "database.bin")),
    }
    manifest["sidecars"] = {
        "terrain_features": {
            "path": "terrain_features.bin", "schema": "G1TF/v1",
            "version": 1, "dimensions": 4,
            "sha256": sha256_file(
                os.path.join(staging, "terrain_features.bin")),
        },
        "terrain_support": {
            "path": "terrain_support.bin", "schema": "G1SP/v1",
            "version": 1, "dimensions": 3,
            "columns": list(SUPPORT_COLUMNS),
            "sha256": sha256_file(
                os.path.join(staging, "terrain_support.bin")),
        },
    }
    manifest["scene_index"] = {
        "path": "scenes/index.json",
        "schema": "g1-terrain-scene-index/v1",
        "sha256": sha256_file(
            os.path.join(staging, "scenes", "index.json")),
    }
    manifest["validation_file"] = {
        "path": "validation.json", "schema": "g1-terrain-validation/v1",
        "sha256": sha256_file(os.path.join(staging, "validation.json")),
    }
    if set(manifest) != MOTION_MANIFEST_KEYS:
        raise ValueError("final motion manifest key set changed")
    return manifest
~~~

Add an exact expected-tree helper and extend the existing array comparison to
include support:

~~~python
def _relative_files(root):
    return {
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _, files in os.walk(root)
        for name in files
    }


def _expected_files(scene_pack):
    expected = {
        "database.bin", "terrain_features.bin", "terrain_support.bin",
        "manifest.json", "validation.json", os.path.join("scenes", "index.json"),
    }
    for scene in scene_pack.scenes:
        for name in (
            "scene.json", "terrain.bin", "terrain.obj", "walkability.bin",
        ):
            expected.add(os.path.join("scenes", scene.scene_id, name))
    return expected


def _validate_staged_artifacts(
    staging, artifacts, manifest, scene_pack,
):
    if _relative_files(staging) != _expected_files(scene_pack):
        raise ValueError("staged artifact file tree is incomplete or has extras")
    loaded = read_holden_database(os.path.join(staging, "database.bin"))
    loaded.terrain_features = read_terrain_sidecar(
        os.path.join(staging, "terrain_features.bin"))
    loaded.terrain_support = read_support_sidecar(
        os.path.join(staging, "terrain_support.bin"))
    loaded.validate()
    for name, dtype in (
        ("positions", "<f4"), ("velocities", "<f4"),
        ("rotations", "<f4"), ("angular_velocities", "<f4"),
        ("parents", "<i4"), ("range_starts", "<i4"),
        ("range_stops", "<i4"), ("contacts", "u1"),
        ("terrain_features", "<f4"), ("terrain_support", "<f4"),
    ):
        expected = np.ascontiguousarray(getattr(artifacts, name), dtype=dtype)
        if not np.array_equal(getattr(loaded, name), expected):
            raise ValueError(f"staged {name} does not match requested artifacts")
    if open(os.path.join(staging, "manifest.json"), "rb").read() \
            != canonical_json_bytes(manifest):
        raise ValueError("staged manifest bytes changed")
    if open(os.path.join(staging, "validation.json"), "rb").read() \
            != canonical_json_bytes(manifest["validation"]):
        raise ValueError("staged validation bytes changed")
    if open(os.path.join(staging, "scenes", "index.json"), "rb").read() \
            != scene_pack.index_json:
        raise ValueError("staged scene index bytes changed")
    for scene in scene_pack.scenes:
        scene_dir = os.path.join(staging, "scenes", scene.scene_id)
        expected_payloads = {
            "scene.json": scene.scene_json,
            "terrain.bin": scene.terrain_bin,
            "terrain.obj": scene.terrain_obj,
            "walkability.bin": scene.walkability_bin,
        }
        for name, expected in expected_payloads.items():
            if open(os.path.join(scene_dir, name), "rb").read() != expected:
                raise ValueError(
                    f"{scene.scene_id}/{name}: staged scene bytes changed")
~~~

- [ ] **Step 4: Implement fsync and rollback-safe directory replacement**

Replace `publish_artifacts` with:

~~~python
def _fsync_tree(root):
    directories = []
    for directory, child_directories, files in os.walk(root):
        child_directories.sort()
        files.sort()
        directories.append(directory)
        for name in files:
            _fsync_file(os.path.join(directory, name))
    for directory in reversed(directories):
        _fsync_directory(directory)


def publish_artifacts(
    output_dir, artifacts, manifest_base, scene_pack, validate_candidate,
):
    manifest_base = _normalized_manifest_base(manifest_base)
    artifacts.validate()
    if not callable(validate_candidate):
        raise TypeError("validate_candidate must be callable")
    output_dir = os.path.abspath(os.fspath(output_dir))
    parent = os.path.dirname(output_dir)
    basename = os.path.basename(output_dir)
    if not basename:
        raise ValueError("output directory must have a basename")
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(
        prefix=f".{basename}.staging-", dir=parent)
    backup = tempfile.mkdtemp(
        prefix=f".{basename}.previous-", dir=parent)
    os.rmdir(backup)
    previous_moved = False
    try:
        write_holden_database(os.path.join(staging, "database.bin"), artifacts)
        _fsync_file(os.path.join(staging, "database.bin"))
        _write_bytes_fsync(
            os.path.join(staging, "terrain_features.bin"),
            terrain_sidecar_bytes(artifacts.terrain_features))
        _write_bytes_fsync(
            os.path.join(staging, "terrain_support.bin"),
            support_sidecar_bytes(artifacts.terrain_support))
        _write_scene_pack(staging, scene_pack)
        _write_bytes_fsync(
            os.path.join(staging, "validation.json"),
            canonical_json_bytes(manifest_base["validation"]))
        manifest = _finalize_manifest(staging, manifest_base)
        _write_bytes_fsync(
            os.path.join(staging, "manifest.json"),
            canonical_json_bytes(manifest))
        _validate_staged_artifacts(
            staging, artifacts, manifest, scene_pack)
        validate_candidate(staging)
        _fsync_tree(staging)

        if os.path.lexists(output_dir):
            os.replace(output_dir, backup)
            previous_moved = True
            _fsync_directory(parent)
        try:
            os.replace(staging, output_dir)
            _fsync_directory(parent)
        except BaseException:
            if os.path.lexists(output_dir):
                os.replace(output_dir, staging)
            if previous_moved:
                os.replace(backup, output_dir)
                previous_moved = False
                _fsync_directory(parent)
            raise
        if previous_moved:
            _remove_path(backup)
            previous_moved = False
            _fsync_directory(parent)
        return manifest
    finally:
        if os.path.lexists(staging):
            _remove_path(staging)
        if previous_moved and not os.path.lexists(output_dir) \
                and os.path.lexists(backup):
            os.replace(backup, output_dir)
            previous_moved = False
            _fsync_directory(parent)
        if not previous_moved and os.path.lexists(backup):
            _remove_path(backup)
~~~

Do not catch `KeyboardInterrupt`, validation errors, or I/O errors inside the
staging phase; `finally` removes only the unique paths created by this call. If
even the rollback rename itself fails, leave the unique backup intact and
surface the I/O error rather than deleting the last-good bytes. The prior output
is not moved until all byte comparisons, the independent candidate validator,
and staging fsync have succeeded.

- [ ] **Step 5: Run publication tests and verify full-tree GREEN**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts -v
~~~

Expected: all binary-codec and publication tests pass. Injected candidate,
scene-byte, and rename failures leave the sentinel as the only prior-output
file, and successful publication leaves no sibling scratch path.

- [ ] **Step 6: Refactor the builder to measure all candidates and build all scenes**

In `resources/build_g1_terrain_database.py`, set
`SCHEMA = "g1-terrain-artifacts/v2"`; remove `HEIGHTFIELD_BORDER`,
`_heightfield_contract`, `export_heightfield`, `--runtime-terrain`, and the
terrain-writer closure. Import:

~~~python
from resources.g1_terrain_builder.scenes import (
    all_scene_definitions,
    build_scene_pack,
    select_grail_scene_bases,
)
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    build_facing_centerline,
    sample_terrain_features,
    surface_semantics,
    surface_semantics_signature,
)
from resources.validate_g1_terrain_database import validate_artifact_directory
~~~

At discovery, retain both the full stable corpus and the motion subset:

~~~python
all_grail_paths = sorted(glob.glob(args.grail_glob))
if not all_grail_paths:
    raise FileNotFoundError(f"GRAIL glob matched no clips: {args.grail_glob}")
for path in all_grail_paths:
    _require_file(path, "GRAIL clip")
grail_paths = all_grail_paths
if args.grail_limit is not None:
    grail_paths = all_grail_paths[:args.grail_limit]
~~~

`--grail-limit 0` is no longer a missing-corpus escape hatch: even a Takara-only
diagnostic requires the real candidate meshes to publish the fixed scene
catalog. Keep the existing nonnegative limit check.

Before the conversion loop, create:

~~~python
path_by_base = {
    os.path.splitext(os.path.basename(path))[0]: path
    for path in all_grail_paths
}
if len(path_by_base) != len(all_grail_paths):
    raise ValueError("duplicate GRAIL terrain base name")
measured_max_heights = {}
clips_by_terrain = {}
~~~

Inside the conversion loop, after constructing its one `terrain` provider and
after `finalize_clip`, retain only immutable selection/route inputs:

~~~python
if source.terrain_id != "flat":
    measured_max_heights[source.terrain_id] = \
        terrain.footprint()["height"]
    clips_by_terrain[source.terrain_id] = clip
~~~

After the loop, measure candidates not present in a diagnostic motion subset,
then convert only selected missing clips for route metadata without adding them
to `clips`, `sources`, `reports`, or the database:

~~~python
for base in sorted(path_by_base):
    if base not in measured_max_heights:
        measured_max_heights[base] = \
            GrailTerrain.from_base(base).footprint()["height"]
selected_scene_bases = select_grail_scene_bases(measured_max_heights)
for base in selected_scene_bases.values():
    if base in clips_by_terrain:
        continue
    route_source = load_grail(path_by_base[base])
    route_clip, route_skeleton, _ = convert_source_clip(
        route_source, kin, OUTPUT_FPS)
    if route_skeleton.signature() != expected_skeleton.signature():
        raise ValueError(f"{base}: scene-route skeleton signature changed")
    clips_by_terrain[base] = route_clip
scene_pack = build_scene_pack(all_scene_definitions(
    measured_max_heights, clips_by_terrain))
~~~

This is intentionally an all-candidate measurement in diagnostics: otherwise
`--grail-limit 1` could silently publish a different low/medium/high catalog.

- [ ] **Step 7: Build the exact manifest base and publish through the independent validator**

Replace the old manifest dictionary with:

~~~python
contact_config = ContactConfig()
manifest_base = {
    "schema": SCHEMA,
    "output_fps": OUTPUT_FPS,
    "feature_dimensions": 31,
    "terrain_dimensions": 4,
    "support_dimensions": 3,
    "terrain_feature_distances_m": TERRAIN_DISTANCES,
    "total_clips": len(sources),
    "grail_clips": len(grail_paths),
    "skipped_clips": 0,
    "database_frames": len(artifacts.positions),
    "diagnostic_mode": args.grail_limit is not None,
    "sources": source_manifest,
    "skeleton": {
        "names": list(expected_skeleton.names),
        "parents": expected_skeleton.parents.tolist(),
        "signature": expected_skeleton.signature(),
    },
    "contact": {
        "speed_threshold": contact_config.speed_threshold,
        "height_threshold": contact_config.height_threshold,
        "median_filter_frames": contact_config.median_filter_frames,
    },
    "surface": {
        "semantics": surface_semantics(),
        "signature": surface_semantics_signature(),
    },
    "validation": {
        "schema": "g1-terrain-validation/v1",
        "fk_max_error_m": [
            float(report["fk_max_error_m"]) for report in reports],
        "duration_error_s": [
            float(report["duration_error_s"]) for report in reports],
        "quaternion_norm_max_error": [
            float(report["quaternion_norm_max_error"])
            for report in reports
        ],
    },
}
manifest = publish_artifacts(
    args.output, artifacts, manifest_base, scene_pack,
    lambda staging: validate_artifact_directory(
        staging,
        full_source_validation=args.grail_limit is None,
        source_options={
            "grail_glob": args.grail_glob,
            "g1_xml": args.g1_xml,
            "takara": args.takara,
            "remap": args.remap,
        }),
)
return manifest
~~~

Thus a diagnostic candidate gets complete byte/schema/scene validation before
rename, while a full candidate also recomputes all 459,682 source rows before
the old output moves. A full-source failure cannot publish a partially trusted
pack.

Task 12 implements `validate_artifact_directory`; during this task, first use
a strict test double and land Steps 6--7 in the same commit as Task 12's initial
public validator extraction so the import is never broken at a commit boundary.
Update the CLI success line to:

~~~python
print(
    f"BUILT {manifest['schema']} frames={manifest['database_frames']} "
    f"clips={manifest['total_clips']} scenes={len(scene_pack.scenes)} "
    f"output={args.output}")
~~~

- [ ] **Step 8: Update mocked builder tests for immutable all-scene publication**

In `tests/python/test_build_cli.py`, remove `runtime_terrain` from each
`SimpleNamespace`. For mocked unit tests, patch
`all_scene_definitions`, `build_scene_pack`, and `publish_artifacts` only after
the source/skeleton condition under test. Add a focused test proving a limit
does not limit scene measurement:

~~~python
    def test_grail_limit_does_not_limit_scene_candidate_measurement(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        paths = ["clips/a.pkl", "clips/b.pkl", "clips/c.pkl"]
        takara = SimpleNamespace(
            name="takara", terrain_id="flat", fps=25.0,
            qpos=np.zeros((2, 36), np.float32))
        grail = SimpleNamespace(
            name="a", terrain_id="a", fps=25.0,
            qpos=np.zeros((2, 36), np.float32))
        clip_a = HoldenClip.empty(2, 31)
        clip_b = HoldenClip.empty(2, 31)
        skeleton = SkeletonSpec(
            tuple(["Simulation", "Hips", "LeftToe", "RightToe"]
                  + [f"Bone{i}" for i in range(27)]),
            np.array([-1] + list(range(30)), np.int32))
        report = {
            "fk_max_error_m": 0.0, "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }
        surfaces = {}

        class Surface:
            def __init__(self, base):
                self.base = base
            def footprint(self):
                return {"height": {"a": 0.1, "b": 0.2, "c": 0.3}[self.base]}

        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(builder.glob, "glob", return_value=paths),
            mock.patch.object(builder, "G1Kinematics"),
            mock.patch.object(builder, "load_takara", return_value=takara),
            mock.patch.object(builder, "load_grail", return_value=grail),
            mock.patch.object(
                builder, "finalize_clip",
                side_effect=((clip_a, skeleton, report),
                             (clip_b, skeleton, report))),
            mock.patch.object(
                builder.GrailTerrain, "from_base",
                side_effect=lambda base: surfaces.setdefault(base, Surface(base))),
            mock.patch.object(
                builder, "select_grail_scene_bases",
                side_effect=ValueError("measured all candidates")) as select,
        ):
            with self.assertRaisesRegex(ValueError, "measured all"):
                builder.build_artifacts(args)
        self.assertEqual(set(select.call_args.args[0]), {"a", "b", "c"})
~~~

- [ ] **Step 9: Run publisher and mocked builder tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts \
  tests.python.test_build_cli.BuildCliTests.test_invalid_build_arguments_fail_before_publication \
  tests.python.test_build_cli.BuildCliTests.test_duplicate_source_names_are_rejected_before_conversion \
  tests.python.test_build_cli.BuildCliTests.test_skeleton_changes_are_rejected_before_combination \
  tests.python.test_build_cli.BuildCliTests.test_grail_limit_does_not_limit_scene_candidate_measurement -v
~~~

Expected: all selected tests pass; no test or implementation writes into
`resources/g1_terrain/`.

- [ ] **Step 10: Commit the manifest and publication boundary with the validator extraction**

After completing Task 12 Steps 1--4 so the validator import exists, commit only
source/tests/docs, never generated artifacts:

~~~bash
git add resources/g1_terrain_builder/artifacts.py \
  resources/build_g1_terrain_database.py \
  resources/validate_g1_terrain_database.py \
  tests/python/test_artifacts.py tests/python/test_build_cli.py
git diff --cached --check
git commit -m "feat: atomically publish complete G1 scene packs"
~~~

### Task 12: Independently validate motion, scenes, routes, hashes, and source rows

**Files:**
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `tests/python/test_build_cli.py`

**Interfaces:**
- Changes: `validate_artifact_directory(artifact_dir,
  full_source_validation=False, source_options=None) -> dict`.
- CLI accepts `--full-source-validation`, `--grail-glob`, `--g1-xml`,
  `--takara`, and `--remap`; source-path options matter only with the full flag.
- The normal validator independently reloads every published file, rejects
  noncanonical/duplicate/non-finite JSON, enforces exact key sets and safe
  relative paths, verifies all SHA-256 descriptors, validates every route and
  grid cell contract, and runs surface parity for all scenes.
- The full-source mode additionally reconverts every source clip and recomputes
  pose rows, contacts, terrain features, and three support columns from the
  canonical exact surface before comparing the published rows.
- Walkability route checks use nearest grid node with half-away-free Python
  `floor(value + 0.5)` rounding. Traversal routes must remain class 1; stress
  routes class 2; safe-stop routes begin class 1, cross once into class 0, never
  re-enter class 1, and remain inside heightfield/lookahead bounds.

- [ ] **Step 1: Rewrite the diagnostic CLI test around the v2 complete pack**

In `tests/python/test_build_cli.py`, extend imports with
`read_support_sidecar`, `read_walkability`, and `REQUIRED_SCENE_IDS`. Replace
`test_one_grail_clip_builds_and_validates` with this v2 acceptance test; keep
the invalid-argument and mocked builder tests from Task 11:

~~~python
    def test_one_grail_clip_builds_fourteen_scenes_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "g1_terrain")
            built = subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", output, "--grail-limit", "1",
            ], check=True, text=True, capture_output=True)
            validated = subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py", output,
            ], check=True, text=True, capture_output=True)
            with open(os.path.join(output, "manifest.json")) as stream:
                manifest = json.load(stream)
            with open(os.path.join(output, "scenes", "index.json")) as stream:
                index = json.load(stream)
            database = read_holden_database(
                os.path.join(output, "database.bin"))
            terrain_features = read_terrain_sidecar(
                os.path.join(output, "terrain_features.bin"))
            terrain_support = read_support_sidecar(
                os.path.join(output, "terrain_support.bin"))
            default_terrain = os.path.join(
                output, "scenes", "grail-curb-default", "terrain.bin")
            with open(default_terrain, "rb") as stream:
                terrain_header = struct.unpack("<4sIII4f", stream.read(32))

        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v2")
        self.assertIn(
            "BUILT g1-terrain-artifacts/v2", built.stdout)
        self.assertIn("clips=2 scenes=14", built.stdout)
        self.assertEqual(built.stderr, "")
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["skipped_clips"], 0)
        self.assertTrue(manifest["diagnostic_mode"])
        self.assertEqual(manifest["support_dimensions"], 3)
        self.assertEqual(index["scene_ids"], list(REQUIRED_SCENE_IDS))
        self.assertEqual(terrain_header[:2], (b"G1HF", 2))
        self.assertEqual(len(database.positions), len(terrain_features))
        self.assertEqual(len(database.positions), len(terrain_support))
        takara_stop = manifest["sources"][0]["range_stop"]
        np.testing.assert_array_equal(
            terrain_support[:takara_stop],
            np.zeros((takara_stop, 3), np.float32))
        self.assertEqual(validated.stderr, "")
        self.assertEqual(
            validated.stdout,
            f"VALID g1-terrain-artifacts/v2 frames={len(database.positions)} "
            "clips=2 bones=31 terrain_dims=4 support_dims=3 scenes=14 "
            "source_rows=0\n")
~~~

- [ ] **Step 2: Run the diagnostic test and verify the v1 validator fails RED**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_build_cli.BuildCliTests.test_one_grail_clip_builds_fourteen_scenes_and_validates -v
~~~

Expected before the validator rewrite: `FAIL`; the old validator requires
root `terrain.bin`, requires schema v1, and does not load G1SP or scenes.

- [ ] **Step 3: Make JSON, path, hash, and manifest validation strict and canonical**

Update imports in `validate_g1_terrain_database.py`:

~~~python
import glob

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import (
    SUPPORT_COLUMNS,
    read_support_sidecar,
    read_terrain_sidecar,
    read_walkability,
)
from resources.g1_terrain_builder.database import (
    ContactConfig,
    derive_contacts,
    derive_velocities,
    forward_kinematics_arrays,
    read_holden_database,
    sample_terrain_support,
)
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
)
from resources.g1_terrain_builder.scenes import (
    COORDINATE_SIGNATURE,
    REQUIRED_SCENE_IDS,
    TERRAIN_DISTANCES,
    build_scene,
    grail_scene_definition,
    procedural_scene_definitions,
    select_grail_scene_bases,
)
from resources.g1_terrain_builder.sources import load_grail, load_takara
from resources.g1_terrain_builder.terrain import (
    HEIGHTFIELD_DIAGONAL,
    HEIGHTFIELD_INTERPOLATION,
    GrailTerrain,
    HeightGrid,
    build_facing_centerline,
    grail_surface_parity,
    sample_terrain_features,
    surface_semantics,
    surface_semantics_signature,
    FlatTerrain,
)
~~~

Set exact constants:

~~~python
SCHEMA = "g1-terrain-artifacts/v2"
OUTPUT_FPS = 25.0
FEATURE_DIMENSIONS = 31
TERRAIN_DIMENSIONS = 4
SUPPORT_DIMENSIONS = 3
MANIFEST_KEYS = {
    "schema", "output_fps", "feature_dimensions", "terrain_dimensions",
    "support_dimensions", "terrain_feature_distances_m", "total_clips",
    "grail_clips", "skipped_clips", "database_frames", "diagnostic_mode",
    "sources", "skeleton", "contact", "surface", "database", "sidecars",
    "scene_index", "validation_file", "validation",
}
GRAIL_EXPECTED_BASES = {
    "grail-curb-default": "terrain_curbs__curb_000__000",
    "grail-curb-low": "terrain_curbs__curb_186__004",
    "grail-curb-medium": "terrain_curbs__curb_022__001",
    "grail-curb-high": "terrain_curbs__curb_165__006",
}
EXPECTED_ROUTE_IDS = {
    "grail-curb-default": ("curb-forward",),
    "grail-curb-low": ("curb-forward",),
    "grail-curb-medium": ("curb-forward",),
    "grail-curb-high": ("curb-forward",),
    "stairs-shallow": ("ascent-landing-descent",),
    "stairs-standard": ("ascent-landing-descent",),
    "stairs-unseen-variable": ("ascent-landing-descent",),
    "ramp-05-up-down": ("up-landing-down",),
    "ramp-10-up-down": ("up-landing-down",),
    "ramp-15-stress": ("up-landing-down",),
    "cross-slope-05": ("forward-cross-slope",),
    "cross-slope-10": ("forward-cross-slope",),
    "mixed-multilevel": ("full-course",),
    "blocked-course": ("wall-safe-stop", "ramp-safe-stop"),
}
DEFAULT_SOURCE_OPTIONS = {
    "grail_glob": "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl",
    "g1_xml": "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
    "takara": "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz",
    "remap": "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy",
}
_HEIGHTFIELD_HEADER = struct.Struct("<4sIII4f")
_WALKABILITY_HEADER = struct.Struct("<4sIII")
~~~

Replace `_read_json` with a byte-preserving canonical reader:

~~~python
def _canonical_json_bytes(value):
    return (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def _read_json(path):
    def object_without_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = open(path, "rb").read()
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path} is invalid JSON: {error}") from error
    _require(
        payload == _canonical_json_bytes(value),
        f"{path} is not canonical sorted indented newline JSON")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_relative_file(root, relative, label):
    _require(
        isinstance(relative, str) and relative \
        == os.path.normpath(relative) and not os.path.isabs(relative)
        and relative != ".." and not relative.startswith(".." + os.sep),
        f"{label} path is unsafe")
    path = os.path.join(root, relative)
    _require(os.path.isfile(path) and not os.path.islink(path),
             f"missing or non-regular {label}")
    _require(os.path.commonpath([root, os.path.realpath(path)]) == root,
             f"{label} path escapes artifact directory")
    _require(os.path.getsize(path) > 0, f"{label} must be non-empty")
    return path


def _hash_descriptor(root, value, label, fixed):
    _require(isinstance(value, dict) and set(value) == set(fixed) | {"sha256"},
             f"{label} descriptor keys are invalid")
    for key, expected in fixed.items():
        _require(value.get(key) == expected, f"{label} {key} mismatch")
    digest = value.get("sha256")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
             f"{label} SHA-256 is invalid")
    path = _regular_relative_file(root, fixed["path"], label)
    _require(_sha256_file(path) == digest, f"{label} SHA-256 mismatch")
    return path
~~~

Update `_validate_manifest_header` to require `set(manifest) == MANIFEST_KEYS`,
schema/dimensions/fps/distances exactly, and full-corpus counts when
`diagnostic_mode` is false:

~~~python
def _validate_manifest_header(manifest):
    _require(isinstance(manifest, dict), "manifest.json root must be an object")
    _require(set(manifest) == MANIFEST_KEYS, "manifest.json key set changed")
    _require(manifest["schema"] == SCHEMA, f"schema must be {SCHEMA}")
    _require(manifest["output_fps"] == OUTPUT_FPS, "output_fps must be 25.0")
    _require(manifest["feature_dimensions"] == FEATURE_DIMENSIONS,
             "feature_dimensions must be 31")
    _require(manifest["terrain_dimensions"] == TERRAIN_DIMENSIONS,
             "terrain_dimensions must be 4")
    _require(manifest["support_dimensions"] == SUPPORT_DIMENSIONS,
             "support_dimensions must be 3")
    _require(manifest["terrain_feature_distances_m"] == TERRAIN_DISTANCES,
             "terrain feature distances changed")
    _require(isinstance(manifest["diagnostic_mode"], bool),
             "diagnostic_mode must be boolean")
    if not manifest["diagnostic_mode"]:
        _require(manifest["total_clips"] == 1770,
                 "full pack total_clips must be 1770")
        _require(manifest["grail_clips"] == 1769,
                 "full pack grail_clips must be 1769")
        _require(manifest["database_frames"] == 459682,
                 "full pack database_frames must be 459682")
~~~

Validate the surface and root descriptors before loading payloads:

~~~python
def _validate_motion_descriptors(root, manifest):
    surface = manifest["surface"]
    _require(isinstance(surface, dict)
             and set(surface) == {"semantics", "signature"},
             "surface descriptor keys are invalid")
    _require(surface["semantics"] == surface_semantics(),
             "surface semantics changed")
    _require(surface["signature"] == surface_semantics_signature(),
             "surface signature changed")
    database_path = _hash_descriptor(root, manifest["database"], "database", {
        "path": "database.bin", "schema": "holden-database/v1",
    })
    sidecars = manifest["sidecars"]
    _require(isinstance(sidecars, dict)
             and set(sidecars) == {"terrain_features", "terrain_support"},
             "sidecar descriptor keys are invalid")
    features_path = _hash_descriptor(
        root, sidecars["terrain_features"], "terrain features", {
            "path": "terrain_features.bin", "schema": "G1TF/v1",
            "version": 1, "dimensions": 4,
        })
    support_path = _hash_descriptor(
        root, sidecars["terrain_support"], "terrain support", {
            "path": "terrain_support.bin", "schema": "G1SP/v1",
            "version": 1, "dimensions": 3,
            "columns": list(SUPPORT_COLUMNS),
        })
    index_path = _hash_descriptor(root, manifest["scene_index"], "scene index", {
        "path": "scenes/index.json", "schema": "g1-terrain-scene-index/v1",
    })
    validation_path = _hash_descriptor(
        root, manifest["validation_file"], "validation file", {
            "path": "validation.json", "schema": "g1-terrain-validation/v1",
        })
    return database_path, features_path, support_path, index_path, validation_path
~~~

Keep the existing skeleton, source-range, duration, quaternion, contact, and
source-map checks, but remove every v1 `terrain` object check. Replace
`_validate_parameters` with:

~~~python
def _validate_parameters(manifest, clip_count):
    contact = manifest["contact"]
    _require(isinstance(contact, dict) and set(contact) == {
        "speed_threshold", "height_threshold", "median_filter_frames",
    }, "contact keys are invalid")
    _require(contact == {
        "speed_threshold": 0.15, "height_threshold": 0.06,
        "median_filter_frames": 3,
    }, "contact parameters changed")
    validation = manifest["validation"]
    _require(isinstance(validation, dict) and set(validation) == {
        "schema", "duration_error_s", "fk_max_error_m",
        "quaternion_norm_max_error",
    }, "validation keys are invalid")
    _require(validation["schema"] == "g1-terrain-validation/v1",
             "validation schema changed")
    limits = {
        "fk_max_error_m": 0.001,
        "duration_error_s": 1.0 / OUTPUT_FPS + 1e-12,
        "quaternion_norm_max_error": 1e-4,
    }
    for name, limit in limits.items():
        values = validation[name]
        _require(isinstance(values, list) and len(values) == clip_count,
                 f"validation {name} must contain one value per clip")
        numbers = [
            _finite_number(value, f"validation {name}[{index}]")
            for index, value in enumerate(values)
        ]
        _require(all(0.0 <= value <= limit for value in numbers),
                 f"validation {name} exceeds {limit}")
~~~

Define these immediately before the existing `_validate_sources` loop:

~~~python
source_keys = {
    "name", "terrain_id", "source_fps", "source_frames", "output_frames",
    "range_start", "range_stop", "source_frame_map",
}
ordered_names = []
~~~

Immediately after the loop's existing `label = f"sources[{index}]"` line,
insert:

~~~python
_require(isinstance(source, dict) and set(source) == source_keys,
         f"{label} keys are invalid")
ordered_names.append(source["name"])
~~~

Immediately after the loop and before the final cursor-coverage gate, insert:

~~~python

_require(ordered_names[1:] == sorted(ordered_names[1:]),
         "GRAIL sources must be lexically sorted")
~~~

- [ ] **Step 4: Parse G1HF/v2, G1WM/v1, exact OBJ bytes, and route samples**

Replace the v1 `_parse_heightfield`, `_parse_obj`, and coverage functions with:

~~~python
def _v2_normal_or_positive_zero(values):
    encoded = np.asarray(values, dtype="<f4")
    bits = encoded.view("<u4")
    exponent = bits & np.uint32(0x7f800000)
    return (bits == 0) | (
        (exponent != 0) & (exponent != np.uint32(0x7f800000)))


def _v2_positive_normal(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7f800000
    return (bits & 0x80000000) == 0 \
        and exponent != 0 and exponent != 0x7f800000


def _parse_heightfield(path, metadata):
    payload = open(path, "rb").read()
    _require(len(payload) >= _HEIGHTFIELD_HEADER.size,
             f"{path}: truncated G1HF header")
    magic, version, nx, nz, ox, oz, cell, exterior = \
        _HEIGHTFIELD_HEADER.unpack_from(payload)
    _require(magic == b"G1HF" and version == 2,
             f"{path}: scene heightfield must be G1HF/v2")
    _require(nx >= 2 and nz >= 2, f"{path}: invalid G1HF dimensions")
    _require(nx <= (len(payload) - 32) // 4,
             f"{path}: overflowing G1HF dimensions")
    expected = 32 + nx * nz * 4
    _require(len(payload) == expected,
             f"{path}: truncated or trailing G1HF payload")
    _require(_v2_normal_or_positive_zero([ox, oz, exterior]).all()
             and _v2_positive_normal(cell),
             f"{path}: G1HF/v2 header must be normal-or-positive-zero "
             "with positive-normal cell")
    heights = np.frombuffer(payload, "<f4", nx * nz, 32).reshape(nz, nx).copy()
    _require(_v2_normal_or_positive_zero(heights).all(),
             f"{path}: G1HF/v2 heights must be normal-or-positive-zero")
    expected_keys = {
        "path", "schema", "version", "nx", "nz", "origin_x", "origin_z",
        "cell_size_m", "exterior_height_m", "interpolation", "diagonal",
        "sha256",
    }
    _require(isinstance(metadata, dict) and set(metadata) == expected_keys,
             "scene heightfield metadata keys are invalid")
    fixed = {
        "path": "terrain.bin", "schema": "G1HF/v2", "version": 2,
        "nx": nx, "nz": nz, "interpolation": HEIGHTFIELD_INTERPOLATION,
        "diagonal": HEIGHTFIELD_DIAGONAL,
    }
    for key, value in fixed.items():
        _require(metadata[key] == value, f"scene heightfield {key} mismatch")
    for key, actual in (
        ("origin_x", ox), ("origin_z", oz),
        ("cell_size_m", cell), ("exterior_height_m", exterior),
    ):
        expected_value = _finite_number(metadata[key], f"heightfield {key}")
        _require(expected_value == actual
                 and struct.pack("<f", expected_value)
                 == struct.pack("<f", actual),
                 f"scene heightfield {key} mismatch")
    _require(metadata["cell_size_m"] == float(np.float32(0.02))
             and cell == float(np.float32(0.02)),
             "scene heightfield cell size must be 0.02")
    _require(metadata["exterior_height_m"] == 0.0 and exterior == 0.0,
             "scene heightfield exterior height must be zero")
    _require(_sha256_file(path) == metadata["sha256"],
             "scene heightfield SHA-256 mismatch")
    return HeightGrid(
        heights,
        float(metadata["origin_x"]), float(metadata["origin_z"]),
        float(metadata["cell_size_m"]),
        float(metadata["exterior_height_m"]),
    )


def _parse_walkability(path, metadata, grid):
    values = read_walkability(path)
    _require(values.shape == (grid.nz, grid.nx),
             "walkability dimensions differ from heightfield")
    expected_keys = {
        "path", "schema", "version", "nx", "nz", "classes", "sha256",
    }
    _require(isinstance(metadata, dict) and set(metadata) == expected_keys,
             "walkability metadata keys are invalid")
    _require(metadata == {
        "path": "walkability.bin", "schema": "G1WM/v1", "version": 1,
        "nx": grid.nx, "nz": grid.nz,
        "classes": {"blocked": 0, "certified": 1, "stress": 2},
        "sha256": metadata["sha256"],
    }, "walkability metadata values are invalid")
    _require(_sha256_file(path) == metadata["sha256"],
             "walkability SHA-256 mismatch")
    return values


def _validate_obj(path, metadata, grid):
    _require(isinstance(metadata, dict) and set(metadata) == {
        "path", "schema", "sha256",
    }, "mesh metadata keys are invalid")
    _require(metadata["path"] == "terrain.obj"
             and metadata["schema"] == "obj/v1",
             "mesh path or schema mismatch")
    payload = open(path, "rb").read()
    _require(hashlib.sha256(payload).hexdigest() == metadata["sha256"],
             "mesh SHA-256 mismatch")
    _require(payload == grid.obj_bytes(),
             "terrain.obj is not the exact fixed-diagonal heightfield mesh")


def _xz_inside(bounds, x, z, tolerance=1e-9):
    return bounds[0] - tolerance <= x <= bounds[1] + tolerance \
        and bounds[2] - tolerance <= z <= bounds[3] + tolerance


def _walkability_at(grid, values, x, z):
    gx = (x - grid.origin_x) / grid.cell_size
    gz = (z - grid.origin_z) / grid.cell_size
    _require(0.0 <= gx <= grid.nx - 1 and 0.0 <= gz <= grid.nz - 1,
             "route sample is outside walkability grid")
    ix = min(int(np.floor(gx + 0.5)), grid.nx - 1)
    iz = min(int(np.floor(gz + 0.5)), grid.nz - 1)
    return int(values[iz, ix])


def _route_samples(points, maximum_step):
    output = [points[0]]
    for start, stop in zip(points, points[1:]):
        distance = float(np.linalg.norm(np.asarray(stop) - np.asarray(start)))
        count = max(1, int(np.ceil(distance / maximum_step)))
        output.extend(tuple(
            (1.0 - alpha) * np.asarray(start) + alpha * np.asarray(stop)
        ) for alpha in np.linspace(1.0 / count, 1.0, count))
    return output


def _validate_route(route, grid, walkability, lookahead_bounds):
    _require(isinstance(route, dict) and set(route) == {
        "id", "waypoints_xz", "expected_outcome", "walkability_class",
        "landing_hold_seconds",
    }, "scene route keys are invalid")
    _require(isinstance(route["id"], str) and route["id"],
             "scene route ID is invalid")
    points = np.asarray(route["waypoints_xz"], np.float64)
    _require(points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2
             and np.isfinite(points).all(), "scene route points are invalid")
    hold = _finite_number(route["landing_hold_seconds"], "landing hold")
    _require(hold >= 0.0, "landing hold must be nonnegative")
    _require(hold == 0.0 or points.shape[0] >= 4,
             "landing hold requires waypoint 2 and a later exit")
    expected_class = {
        "traverse": 1, "safe-stop": 0, "traverse-or-safe-stop": 2,
    }.get(route["expected_outcome"])
    _require(expected_class == route["walkability_class"],
             "scene route outcome/class mismatch")
    samples = _route_samples(points, grid.cell_size / 2.0)
    for x, z in samples:
        _require(_xz_inside(lookahead_bounds, float(x), float(z)),
                 "scene route leaves lookahead bounds")
    classes = [
        _walkability_at(grid, walkability, float(x), float(z))
        for x, z in samples
    ]
    if expected_class in (1, 2):
        _require(all(value == expected_class for value in classes),
                 "expected route enters wrong walkability class")
    else:
        _require(classes[0] == 1 and classes[-1] == 0,
                 "safe-stop route must approach from certified into blocked")
        first_blocked = classes.index(0)
        _require(all(value == 1 for value in classes[:first_blocked])
                 and all(value == 0 for value in classes[first_blocked:]),
                 "safe-stop route re-enters traversable cells")
~~~

- [ ] **Step 5: Validate exact scene/index schemas, bounds, procedural bytes, and GRAIL parity**

Add:

~~~python
def _bounds_pair(metadata, minimum_key, maximum_key, dimensions, label):
    minimum = np.asarray(metadata[minimum_key], np.float64)
    maximum = np.asarray(metadata[maximum_key], np.float64)
    _require(minimum.shape == (dimensions,) and maximum.shape == (dimensions,)
             and np.isfinite(minimum).all() and np.isfinite(maximum).all()
             and np.all(minimum <= maximum), f"{label} bounds are invalid")
    return minimum, maximum


def _validate_scene(root, scene_id, manifest_surface_signature):
    scene_root = os.path.join(root, "scenes", scene_id)
    _require(os.path.isdir(scene_root) and not os.path.islink(scene_root),
             f"missing scene directory {scene_id}")
    scene_path = _regular_relative_file(
        root, os.path.join("scenes", scene_id, "scene.json"),
        f"{scene_id} metadata")
    scene = _read_json(scene_path)
    _require(isinstance(scene, dict) and set(scene) == {
        "schema", "id", "label", "provenance", "coordinate_signature",
        "surface_signature", "terrain_feature_distances_m", "heightfield",
        "mesh", "walkability", "bounds", "spawn", "regions", "routes",
    }, f"{scene_id}: scene metadata keys are invalid")
    _require(scene["schema"] == "g1-terrain-scene/v1"
             and scene["id"] == scene_id,
             f"{scene_id}: scene schema or ID mismatch")
    _require(isinstance(scene["label"], str) and scene["label"],
             f"{scene_id}: label is invalid")
    _require(scene["coordinate_signature"] == COORDINATE_SIGNATURE,
             f"{scene_id}: coordinate signature mismatch")
    _require(scene["surface_signature"] == manifest_surface_signature,
             f"{scene_id}: surface signature mismatch")
    _require(scene["terrain_feature_distances_m"] == TERRAIN_DISTANCES,
             f"{scene_id}: terrain distances mismatch")
    provenance = scene["provenance"]
    _require(isinstance(provenance, dict) and set(provenance) == {
        "kind", "source_ids", "parameters",
    } and provenance["kind"] in ("grail", "procedural")
             and isinstance(provenance["source_ids"], list)
             and isinstance(provenance["parameters"], dict),
             f"{scene_id}: provenance is invalid")

    terrain_path = _regular_relative_file(
        scene_root, scene["heightfield"].get("path"),
        f"{scene_id} heightfield")
    obj_path = _regular_relative_file(
        scene_root, scene["mesh"].get("path"), f"{scene_id} mesh")
    walkability_path = _regular_relative_file(
        scene_root, scene["walkability"].get("path"),
        f"{scene_id} walkability")
    grid = _parse_heightfield(terrain_path, scene["heightfield"])
    _validate_obj(obj_path, scene["mesh"], grid)
    walkability = _parse_walkability(
        walkability_path, scene["walkability"], grid)

    bounds = scene["bounds"]
    _require(isinstance(bounds, dict) and set(bounds) == {
        "mesh_min_xyz", "mesh_max_xyz", "heightfield_min_xyz",
        "heightfield_max_xyz", "playable_min_xz", "playable_max_xz",
        "lookahead_min_xz", "lookahead_max_xz",
    }, f"{scene_id}: bounds keys are invalid")
    mesh_min, mesh_max = _bounds_pair(
        bounds, "mesh_min_xyz", "mesh_max_xyz", 3, "mesh")
    height_min, height_max = _bounds_pair(
        bounds, "heightfield_min_xyz", "heightfield_max_xyz", 3,
        "heightfield")
    playable_min, playable_max = _bounds_pair(
        bounds, "playable_min_xz", "playable_max_xz", 2, "playable")
    lookahead_min, lookahead_max = _bounds_pair(
        bounds, "lookahead_min_xz", "lookahead_max_xz", 2, "lookahead")
    heightfield_derived_min = np.array([
        grid.origin_x, float(grid.heights.min()), grid.origin_z])
    heightfield_derived_max = np.array([
        grid.origin_x + (grid.nx - 1) * grid.cell_size,
        float(grid.heights.max()),
        grid.origin_z + (grid.nz - 1) * grid.cell_size])
    def obj_coordinate(value):
        encoded = float(np.float32(value))
        return 0.0 if encoded == 0.0 else encoded

    mesh_derived_min = np.array([
        obj_coordinate(heightfield_derived_min[0]),
        heightfield_derived_min[1],
        obj_coordinate(heightfield_derived_min[2]),
    ])
    mesh_derived_max = np.array([
        obj_coordinate(heightfield_derived_max[0]),
        heightfield_derived_max[1],
        obj_coordinate(heightfield_derived_max[2]),
    ])
    for label, actual, expected in (
        ("mesh minimum", mesh_min, mesh_derived_min),
        ("mesh maximum", mesh_max, mesh_derived_max),
        ("heightfield minimum", height_min, heightfield_derived_min),
        ("heightfield maximum", height_max, heightfield_derived_max),
    ):
        _require(np.array_equal(
            np.asarray(actual, np.float64).view(np.uint64),
            np.asarray(expected, np.float64).view(np.uint64)),
                 f"{scene_id}: {label} mismatch")
    grid_bounds = (
        heightfield_derived_min[0], heightfield_derived_max[0],
        heightfield_derived_min[2], heightfield_derived_max[2])
    playable = (
        playable_min[0], playable_max[0], playable_min[1], playable_max[1])
    lookahead = (
        lookahead_min[0], lookahead_max[0],
        lookahead_min[1], lookahead_max[1])
    for x, z in (
        (playable[0], playable[2]), (playable[1], playable[3]),
        (lookahead[0], lookahead[2]), (lookahead[1], lookahead[3]),
    ):
        _require(_xz_inside(grid_bounds, x, z),
                 f"{scene_id}: playable/lookahead bounds leave heightfield")
    spawn = scene["spawn"]
    _require(isinstance(spawn, dict) and set(spawn) == {
        "position", "yaw_radians",
    }, f"{scene_id}: spawn keys are invalid")
    position = np.asarray(spawn["position"], np.float64)
    yaw = _finite_number(spawn["yaw_radians"], "spawn yaw")
    _require(position.shape == (3,) and np.isfinite(position).all()
             and np.isfinite(yaw)
             and _xz_inside(playable, position[0], position[2]),
             f"{scene_id}: spawn is invalid or outside playable bounds")
    regions = scene["regions"]
    _require(isinstance(regions, dict) and set(regions) == {
        "certified", "stress", "blocked",
    }, f"{scene_id}: region keys are invalid")
    for class_name, entries in regions.items():
        _require(isinstance(entries, list),
                 f"{scene_id}: {class_name} regions must be a list")
        for entry in entries:
            _require(isinstance(entry, dict) and set(entry) == {
                "id", "bounds_xz",
            } and isinstance(entry["id"], str) and entry["id"],
                     f"{scene_id}: invalid {class_name} region")
            region = np.asarray(entry["bounds_xz"], np.float64)
            _require(region.shape == (4,) and np.isfinite(region).all()
                     and region[0] <= region[1] and region[2] <= region[3],
                     f"{scene_id}: invalid {class_name} region bounds")
    _require(isinstance(scene["routes"], list) and scene["routes"],
             f"{scene_id}: routes must be non-empty")
    route_ids = []
    for route in scene["routes"]:
        _validate_route(route, grid, walkability, lookahead)
        route_ids.append(route["id"])
        _require(np.allclose(
            np.asarray(route["waypoints_xz"][0], np.float64),
            position[[0, 2]], rtol=0.0, atol=1e-9),
            f"{scene_id}: route does not start at scene spawn")
        if provenance["kind"] == "procedural" \
                and route["expected_outcome"] == "traverse":
            for x, z in route["waypoints_xz"]:
                _require(
                    x - grid_bounds[0] >= 1.0 - 1e-9
                    and grid_bounds[1] - x >= 1.0 - 1e-9
                    and z - grid_bounds[2] >= 1.0 - 1e-9
                    and grid_bounds[3] - z >= 1.0 - 1e-9,
                    f"{scene_id}: certified route lacks 1 m lookahead margin")
    _require(len(route_ids) == len(set(route_ids)),
             f"{scene_id}: duplicate route ID")
    _require(tuple(route_ids) == EXPECTED_ROUTE_IDS[scene_id],
             f"{scene_id}: deterministic route IDs changed")
    return scene, grid, walkability


def _validate_scene_catalog(root, manifest):
    index_path = os.path.join(root, "scenes", "index.json")
    index = _read_json(index_path)
    _require(index == {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "grail-curb-default",
        "scene_ids": list(REQUIRED_SCENE_IDS),
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }, "scene index contract changed")
    scenes = {}
    for scene_id in REQUIRED_SCENE_IDS:
        scenes[scene_id] = _validate_scene(
            root, scene_id, manifest["surface"]["signature"])

    expected_procedural = {
        definition.scene_id: build_scene(definition)
        for definition in procedural_scene_definitions()
    }
    for scene_id, expected in expected_procedural.items():
        scene_root = os.path.join(root, "scenes", scene_id)
        for name, payload in (
            ("scene.json", expected.scene_json),
            ("terrain.bin", expected.terrain_bin),
            ("terrain.obj", expected.terrain_obj),
            ("walkability.bin", expected.walkability_bin),
        ):
            _require(open(os.path.join(scene_root, name), "rb").read() == payload,
                     f"{scene_id}: deterministic procedural bytes changed")

    for scene_id, expected_base in GRAIL_EXPECTED_BASES.items():
        scene, grid, _ = scenes[scene_id]
        source_ids = scene["provenance"]["source_ids"]
        _require(scene["provenance"]["kind"] == "grail"
                 and len(source_ids) == 2 and source_ids[0] == expected_base,
                 f"{scene_id}: GRAIL provenance base changed")
        terrain = GrailTerrain.from_base(expected_base)
        report = grail_surface_parity(terrain, grid)
        _require(report["node_error_m"] <= 1e-6,
                 f"{scene_id}: node surface parity failed")
        _require(report["within_cell_error_m"] <= 1e-4,
                 f"{scene_id}: within-cell parity failed")
        _require(report["source_away_edge_error_m"] <= 0.005,
                 f"{scene_id}: source surface parity failed")
        _require(report["top_edge_movement_m"] <= 0.02 + 1e-9,
                 f"{scene_id}: source edge moved by more than one cell")
    return scenes
~~~

After validating the catalog, add the exact tree check:

~~~python
def _validate_exact_file_tree(root):
    expected_files = {
        "database.bin", "terrain_features.bin", "terrain_support.bin",
        "manifest.json", "validation.json", os.path.join("scenes", "index.json"),
    }
    expected_directories = {"scenes"}
    for scene_id in REQUIRED_SCENE_IDS:
        directory = os.path.join("scenes", scene_id)
        expected_directories.add(directory)
        for name in (
            "scene.json", "terrain.bin", "terrain.obj", "walkability.bin",
        ):
            expected_files.add(os.path.join(directory, name))
    actual_files = set()
    actual_directories = set()
    for directory, child_directories, files in os.walk(
            root, topdown=True, followlinks=False):
        relative_directory = os.path.relpath(directory, root)
        for name in child_directories:
            path = os.path.join(directory, name)
            _require(not os.path.islink(path),
                     f"artifact tree contains symlink {path}")
            relative = os.path.normpath(os.path.join(
                relative_directory, name))
            actual_directories.add(
                name if relative_directory == "." else relative)
        for name in files:
            path = os.path.join(directory, name)
            _require(os.path.isfile(path) and not os.path.islink(path),
                     f"artifact tree contains non-regular file {path}")
            relative = os.path.normpath(os.path.join(
                relative_directory, name))
            actual_files.add(name if relative_directory == "." else relative)
    _require(actual_files == expected_files,
             "artifact tree has missing, stale, or unexpected files")
    _require(actual_directories == expected_directories,
             "artifact tree has missing, stale, or unexpected directories")
~~~

This prevents an unmanifested partial or stale scene from surviving
publication.

- [ ] **Step 6: Wire the normal public validator and get diagnostic GREEN**

Replace the public function with:

~~~python
def validate_artifact_directory(
    artifact_dir, full_source_validation=False, source_options=None,
):
    root = os.path.abspath(os.fspath(artifact_dir))
    _require(os.path.isdir(root) and not os.path.islink(root),
             "artifact directory does not exist or is a symlink")
    manifest = _read_json(os.path.join(root, "manifest.json"))
    _validate_manifest_header(manifest)
    database_path, features_path, support_path, _, validation_path = \
        _validate_motion_descriptors(root, manifest)
    validation_file = _read_json(validation_path)
    _require(validation_file == manifest["validation"],
             "validation.json does not match manifest validation")
    database = _load_database(database_path)
    database.terrain_features = _load_terrain_features(features_path)
    try:
        database.terrain_support = read_support_sidecar(support_path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"terrain_support.bin is invalid: {error}") from error
    database.validate()
    _require(len(database.positions) == len(database.terrain_features)
             == len(database.terrain_support),
             "database and sidecar frame counts differ")
    names, _ = _validate_skeleton(manifest, database)
    sources = _validate_sources(manifest, database)
    _validate_parameters(manifest, len(sources))
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(database.rotations, axis=-1) - 1.0)))
    _require(quaternion_error <= 1e-4,
             "database quaternion norm error exceeds 0.0001")
    scenes = _validate_scene_catalog(root, manifest)
    _validate_exact_file_tree(root)
    source_rows = 0
    if full_source_validation:
        source_rows = _validate_all_source_rows(
            manifest, database, scenes, source_options)
    return {
        "frames": len(database.positions), "clips": len(sources),
        "bones": len(names), "scenes": len(scenes),
        "source_rows": source_rows,
    }
~~~

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_build_cli.BuildCliTests.test_one_grail_clip_builds_fourteen_scenes_and_validates -v
~~~

Expected: the diagnostic build and normal validator pass; stdout ends with
`terrain_dims=4 support_dims=3 scenes=14 source_rows=0`.

- [ ] **Step 7: Write failing coordinated-corruption tests**

Add a second CLI test that builds one diagnostic pack once, mutates one contract
at a time, runs the validator, and restores the exact original bytes after each
subtest:

~~~python
    def test_validator_rejects_coordinated_motion_scene_and_route_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "g1_terrain")
            subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", output, "--grail-limit", "1",
            ], check=True, text=True, capture_output=True)

            def rejection(label, expected_message, mutate):
                originals = {}

                def preserve(relative):
                    path = os.path.join(output, relative)
                    if relative not in originals:
                        with open(path, "rb") as stream:
                            originals[relative] = stream.read()
                    return path

                mutate(preserve)
                result = subprocess.run([
                    PYTHON, "resources/validate_g1_terrain_database.py", output,
                ], text=True, capture_output=True)
                for relative, payload in originals.items():
                    with open(os.path.join(output, relative), "wb") as stream:
                        stream.write(payload)
                with self.subTest(label=label):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_message, result.stderr)

            def duplicate_manifest_key(preserve):
                with open(preserve("manifest.json"), "wb") as stream:
                    stream.write(b'{"schema":"x","schema":"y"}\n')

            rejection(
                "duplicate_manifest_key", "duplicate JSON key",
                duplicate_manifest_key)

            def corrupt_database_hash(preserve):
                with open(preserve("database.bin"), "ab") as stream:
                    stream.write(b"x")

            rejection(
                "database_hash", "database SHA-256 mismatch",
                corrupt_database_hash)

            def corrupt_support(preserve):
                support_path = preserve("terrain_support.bin")
                with open(support_path, "ab") as stream:
                    stream.write(b"x")
                manifest_path = preserve("manifest.json")
                with open(manifest_path) as stream:
                    manifest = json.load(stream)
                manifest["sidecars"]["terrain_support"]["sha256"] = \
                    file_sha256(support_path)
                with open(manifest_path, "wb") as stream:
                    stream.write(canonical_json_bytes(manifest))

            rejection("coordinated_support", "trailing support", corrupt_support)

            scene_relative = os.path.join(
                "scenes", "stairs-shallow", "scene.json")

            def corrupt_diagonal(preserve):
                path = preserve(scene_relative)
                with open(path) as stream:
                    scene = json.load(stream)
                scene["heightfield"]["diagonal"] = "opposite-diagonal"
                with open(path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection("diagonal", "heightfield diagonal mismatch", corrupt_diagonal)

            def corrupt_exact_origin_metadata(preserve):
                path = preserve(scene_relative)
                with open(path) as stream:
                    scene = json.load(stream)
                scene["heightfield"]["origin_x"] = float(np.nextafter(
                    np.float32(scene["heightfield"]["origin_x"]),
                    np.float32(np.inf)))
                with open(path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "exact_origin_metadata", "heightfield origin_x mismatch",
                corrupt_exact_origin_metadata)

            def corrupt_distinct_bounds(preserve):
                path = preserve(scene_relative)
                with open(path) as stream:
                    scene = json.load(stream)
                # These maxima intentionally have different authorities: the
                # heightfield uses promoted-double nodes; OBJ uses binary32.
                scene["bounds"]["mesh_max_xyz"] = \
                    list(scene["bounds"]["heightfield_max_xyz"])
                with open(path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "distinct_mesh_bounds", "mesh maximum mismatch",
                corrupt_distinct_bounds)

            def corrupt_subnormal_header(preserve):
                relative = os.path.join(
                    "scenes", "stairs-shallow", "terrain.bin")
                terrain_path = preserve(relative)
                with open(terrain_path, "rb") as stream:
                    payload = bytearray(stream.read())
                struct.pack_into("<I", payload, 28, 1)  # exterior = +min subnormal
                with open(terrain_path, "wb") as stream:
                    stream.write(payload)
                scene_path = preserve(scene_relative)
                with open(scene_path) as stream:
                    scene = json.load(stream)
                scene["heightfield"]["exterior_height_m"] = \
                    float(np.frombuffer(struct.pack("<I", 1), "<f4")[0])
                scene["heightfield"]["sha256"] = \
                    hashlib.sha256(payload).hexdigest()
                with open(scene_path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "coordinated_subnormal_header", "normal-or-positive-zero",
                corrupt_subnormal_header)

            def corrupt_negative_zero_header(preserve):
                relative = os.path.join(
                    "scenes", "stairs-shallow", "terrain.bin")
                terrain_path = preserve(relative)
                with open(terrain_path, "rb") as stream:
                    payload = bytearray(stream.read())
                struct.pack_into("<I", payload, 28, 0x80000000)
                with open(terrain_path, "wb") as stream:
                    stream.write(payload)
                scene_path = preserve(scene_relative)
                with open(scene_path) as stream:
                    scene = json.load(stream)
                scene["heightfield"]["exterior_height_m"] = -0.0
                scene["heightfield"]["sha256"] = \
                    hashlib.sha256(payload).hexdigest()
                with open(scene_path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "coordinated_negative_zero_header", "positive-zero",
                corrupt_negative_zero_header)

            def corrupt_subnormal_payload(preserve):
                relative = os.path.join(
                    "scenes", "stairs-shallow", "terrain.bin")
                terrain_path = preserve(relative)
                with open(terrain_path, "rb") as stream:
                    payload = bytearray(stream.read())
                struct.pack_into("<I", payload, 32, 1)  # first height sample
                with open(terrain_path, "wb") as stream:
                    stream.write(payload)
                scene_path = preserve(scene_relative)
                with open(scene_path) as stream:
                    scene = json.load(stream)
                scene["heightfield"]["sha256"] = \
                    hashlib.sha256(payload).hexdigest()
                with open(scene_path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "coordinated_subnormal_payload", "normal-or-positive-zero",
                corrupt_subnormal_payload)

            def corrupt_route(preserve):
                path = preserve(scene_relative)
                with open(path) as stream:
                    scene = json.load(stream)
                scene["routes"][0]["waypoints_xz"][1] = [999.0, 999.0]
                with open(path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection("route", "route leaves lookahead", corrupt_route)

            def corrupt_walkability(preserve):
                low_scene_relative = os.path.join(
                    "scenes", "grail-curb-low", "scene.json")
                scene_path = preserve(low_scene_relative)
                with open(scene_path) as stream:
                    scene = json.load(stream)
                relative = os.path.join(
                    "scenes", "grail-curb-low", "walkability.bin")
                path = preserve(relative)
                with open(path, "rb") as stream:
                    payload = bytearray(stream.read())
                _, _, nx, nz = struct.unpack_from("<4sIII", payload)
                x, z = scene["routes"][0]["waypoints_xz"][0]
                heightfield = scene["heightfield"]
                ix = min(int(np.floor(
                    (x - heightfield["origin_x"])
                    / heightfield["cell_size_m"] + 0.5)), nx - 1)
                iz = min(int(np.floor(
                    (z - heightfield["origin_z"])
                    / heightfield["cell_size_m"] + 0.5)), nz - 1)
                payload[16 + iz * nx + ix] = 0
                with open(path, "wb") as stream:
                    stream.write(payload)
                scene["walkability"]["sha256"] = \
                    hashlib.sha256(payload).hexdigest()
                with open(scene_path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection(
                "coordinated_walkability", "wrong walkability class",
                corrupt_walkability)

            def corrupt_obj(preserve):
                relative = os.path.join(
                    "scenes", "stairs-shallow", "terrain.obj")
                path = preserve(relative)
                with open(path, "rb") as stream:
                    payload = stream.read() + b"v 0 0 0\n"
                with open(path, "wb") as stream:
                    stream.write(payload)
                scene_path = preserve(scene_relative)
                with open(scene_path) as stream:
                    scene = json.load(stream)
                scene["mesh"]["sha256"] = hashlib.sha256(payload).hexdigest()
                with open(scene_path, "wb") as stream:
                    stream.write(canonical_json_bytes(scene))

            rejection("coordinated_obj", "exact fixed-diagonal", corrupt_obj)
~~~

Define `canonical_json_bytes` and `file_sha256` in the test file exactly like
the production canonical writer and chunked SHA helper. Add `hashlib` to
imports.

- [ ] **Step 8: Run corruption tests and close every semantic escape**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_build_cli.BuildCliTests.test_validator_rejects_coordinated_motion_scene_and_route_corruption -v
~~~

Expected after Steps 3--6: every subtest fails validation for the named reason.
If a corruption fails earlier at a hash, update that subtest's coordinated hash
as shown rather than weakening the semantic check.

- [ ] **Step 9: Implement full independent source-row recomputation**

Add:

~~~python
def _recompute_clip(source, terrain, kinematics, skeleton_names):
    clip, skeleton, report = convert_source_clip(
        source, kinematics, OUTPUT_FPS)
    _require(tuple(skeleton.names) == tuple(skeleton_names),
             f"{source.name}: converted skeleton names changed")
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)
    with np.errstate(divide="ignore", invalid="ignore"):
        clip.velocities, clip.angular_velocities = derive_velocities(
            clip.positions, clip.rotations, OUTPUT_FPS)
    left_toe = skeleton.names.index("LeftToe")
    right_toe = skeleton.names.index("RightToe")
    root = skeleton.names.index("Simulation")
    clip.contacts = derive_contacts(
        gp, terrain, left_toe, right_toe, OUTPUT_FPS, ContactConfig())
    clip.terrain_support = sample_terrain_support(
        gp, terrain, root, left_toe, right_toe)
    for frame in range(len(clip.positions)):
        stop = min(frame + 51, len(clip.positions))
        path = gp[frame:stop, root][:, [0, 2]]
        headings3 = holden_quat.mul_vec(
            gq[frame:stop, root],
            np.array([0.0, 0.0, 1.0], np.float64))
        centerline = build_facing_centerline(
            path[0], headings3[:, [0, 2]], path)
        clip.terrain_features[frame] = sample_terrain_features(
            terrain, centerline)
    clip.validate()
    return clip, report


def _validate_all_source_rows(manifest, database, scenes, source_options):
    _require(not manifest["diagnostic_mode"],
             "full source validation requires the full non-diagnostic pack")
    options = dict(DEFAULT_SOURCE_OPTIONS)
    if source_options is not None:
        _require(isinstance(source_options, dict)
                 and set(source_options) <= set(options),
                 "unknown full-source option")
        options.update(source_options)
    for label, path in options.items():
        if label != "grail_glob":
            _require(os.path.isfile(path), f"missing full-source {label}: {path}")
    grail_paths = sorted(glob.glob(options["grail_glob"]))
    _require(len(grail_paths) == 1769,
             "full-source GRAIL glob must contain 1769 clips")
    path_by_base = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in grail_paths
    }
    _require(len(path_by_base) == 1769,
             "full-source GRAIL bases are not unique")
    sources = [load_takara(options["takara"], options["remap"])]
    for entry in manifest["sources"][1:]:
        _require(entry["terrain_id"] in path_by_base,
                 f"missing source file for {entry['terrain_id']}")
        sources.append(load_grail(path_by_base[entry["terrain_id"]]))
    kinematics = G1Kinematics(options["g1_xml"])
    measured = {}
    clips_by_terrain = {}
    rows = 0
    for index, (entry, source) in enumerate(zip(manifest["sources"], sources)):
        _require(source.name == entry["name"],
                 f"sources[{index}] name differs from source file")
        terrain = FlatTerrain() if index == 0 \
            else GrailTerrain.from_base(entry["terrain_id"])
        if index > 0:
            measured[entry["terrain_id"]] = terrain.footprint()["height"]
        clip, report = _recompute_clip(
            source, terrain, kinematics, manifest["skeleton"]["names"])
        if index > 0:
            clips_by_terrain[entry["terrain_id"]] = clip
        start, stop = entry["range_start"], entry["range_stop"]
        _require(stop - start == len(clip.positions),
                 f"sources[{index}] rebuilt frame count changed")
        for name in (
            "positions", "velocities", "rotations", "angular_velocities",
            "contacts", "terrain_features", "terrain_support",
        ):
            published = np.asarray(getattr(database, name)[start:stop])
            rebuilt = np.asarray(getattr(clip, name), dtype=published.dtype)
            _require(np.array_equal(published, rebuilt),
                     f"sources[{index}] rebuilt {name} rows differ")
        _require(clip.source_frames.tolist() == entry["source_frame_map"],
                 f"sources[{index}] rebuilt source map differs")
        for metric in (
            "fk_max_error_m", "duration_error_s",
            "quaternion_norm_max_error",
        ):
            _require(np.isclose(
                report[metric], manifest["validation"][metric][index],
                rtol=0.0, atol=1e-12),
                f"sources[{index}] rebuilt {metric} differs")
        rows += len(clip.positions)
    _require(np.array_equal(
        database.terrain_support[
            manifest["sources"][0]["range_start"]:
            manifest["sources"][0]["range_stop"]],
        np.zeros((manifest["sources"][0]["output_frames"], 3), np.float32)),
        "Takara support rows must be exact zero")
    selected = select_grail_scene_bases(measured)
    targets = {
        "grail-curb-default": None,
        "grail-curb-low": 0.12,
        "grail-curb-medium": 0.24,
        "grail-curb-high": 0.36,
    }
    for scene_id, base in selected.items():
        _require(
            scenes[scene_id][0]["provenance"]["source_ids"][0] == base,
            f"{scene_id}: rebuilt GRAIL selection differs")
        expected = build_scene(grail_scene_definition(
            scene_id, base, clips_by_terrain[base], targets[scene_id]))
        _require(
            _canonical_json_bytes(scenes[scene_id][0]) == expected.scene_json,
            f"{scene_id}: deterministic converted-source scene changed")
    _require(rows == len(database.positions),
             "full source validation did not cover every frame")
    return rows
~~~

The import block in Step 3 includes `FlatTerrain`; use the explicit
`ContactConfig()` call shown above so full-source validation locks the same
`0.15`, `0.06`, and 3-frame median configuration as the manifest.

- [ ] **Step 10: Expose full-source CLI options and exact summary output**

Replace `_parser`/`main` with the following option wiring while preserving the
existing `INVALID <absolute-path>: <reason>` error prefix:

~~~python
def _parser():
    parser = argparse.ArgumentParser(
        description="Validate published Holden G1 terrain motion artifacts")
    parser.add_argument("artifact_directory")
    parser.add_argument("--full-source-validation", action="store_true")
    parser.add_argument("--grail-glob", default=DEFAULT_SOURCE_OPTIONS["grail_glob"])
    parser.add_argument("--g1-xml", default=DEFAULT_SOURCE_OPTIONS["g1_xml"])
    parser.add_argument("--takara", default=DEFAULT_SOURCE_OPTIONS["takara"])
    parser.add_argument("--remap", default=DEFAULT_SOURCE_OPTIONS["remap"])
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    path = os.path.abspath(args.artifact_directory)
    options = {
        "grail_glob": args.grail_glob, "g1_xml": args.g1_xml,
        "takara": args.takara, "remap": args.remap,
    }
    try:
        summary = validate_artifact_directory(
            path, args.full_source_validation, options)
    except Exception as error:
        print(f"INVALID {path}: {error}", file=sys.stderr)
        return 1
    print(
        f"VALID {SCHEMA} frames={summary['frames']} clips={summary['clips']} "
        f"bones={summary['bones']} terrain_dims={TERRAIN_DIMENSIONS} "
        f"support_dims={SUPPORT_DIMENSIONS} scenes={summary['scenes']} "
        f"source_rows={summary['source_rows']}")
    return 0
~~~

- [ ] **Step 11: Run the normal validator, corruption matrix, and all Python tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_build_cli.BuildCliTests.test_one_grail_clip_builds_fourteen_scenes_and_validates \
  tests.python.test_build_cli.BuildCliTests.test_validator_rejects_coordinated_motion_scene_and_route_corruption -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
~~~

Expected: both integration tests pass, every coordinated corruption is rejected
for its semantic reason, and the complete Python suite passes. The full-source
mode is intentionally reserved for the final 459,682-frame pack in Task 13.

- [ ] **Step 12: Commit the independent validation boundary**

If Task 11 Step 10 already committed the initial extraction, commit only the
remaining validator/corruption changes:

~~~bash
git add resources/validate_g1_terrain_database.py \
  tests/python/test_build_cli.py
git diff --cached --check
git commit -m "test: validate complete G1 scene artifacts"
~~~

### Task 13: Pass Gate B and publish the validated 1,770-clip pack

**Files:**
- Verify only: all files in the map below
- Publish generated output: `resources/g1_terrain/` (ignored; do not commit)
- Preserve unchanged: `resources/database.bin`, `resources/features.bin`, all
  prior diagnostic media/logs, and every unrelated dirty or untracked file

**Interfaces:**
- This task changes no source contract. It exercises the normal and strict test
  matrices, a two-clip diagnostic candidate, the C++ G1HF/v2 consumer, the
  rollback path, and finally the full candidate whose pre-rename validator
  recomputes all 459,682 rows.
- Passing this task establishes artifact/surface Gate B only. Runtime Gates
  C--F remain owned by the support/runtime and IK plans.

- [ ] **Step 1: Snapshot user state and immutable legacy artifact hashes**

Run after all source commits and immediately before any diagnostic/full build:

~~~bash
git status --short > /tmp/g1_status_before_artifact_build.txt
sha256sum resources/database.bin resources/features.bin \
  > /tmp/g1_legacy_artifacts_before.sha256
if test -d resources/g1_terrain
then
  find resources/g1_terrain -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > /tmp/g1_last_good_pack_before.sha256
fi
~~~

Expected: both legacy files exist and hash successfully. Do not clean the
worktree, remove untracked files, or use the old generated pack as a staging
directory.

- [ ] **Step 2: Run every Python unit/integration test from a clean process**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
~~~

Expected: all tests pass. In particular, the diagnostic CLI test publishes 14
scenes to a temporary directory, the coordinated corruption matrix rejects
every mutation, and rollback tests preserve their sentinels.

- [ ] **Step 3: Run the owned C++ surface matrix**

Run the four configurations independently so one failure cannot be hidden by a
later command:

~~~bash
! rg -n '(^|[^_[:alnum:]])assert[[:space:]]*\(' \
  tests/cpp/test_terrain_runtime.cpp
g++ -std=c++17 -O0 -g -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_debug
/tmp/test_terrain_runtime_debug
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_strict
/tmp/test_terrain_runtime_strict
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG -I. \
  tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_release
/tmp/test_terrain_runtime_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_san
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1 /tmp/test_terrain_runtime_san
~~~

Expected: every compile and executable exits 0, strict compilation emits no
warning, and ASan/UBSan emits no finding. The migration fixture still samples
G1HF/v1 bilinearly and every new scene fixture samples G1HF/v2 triangles.

- [ ] **Step 4: Build and independently validate an unpublished diagnostic pack**

Remove only the known scratch output for this plan, then build Takara plus the
first sorted GRAIL clip while retaining all 14 scenes:

~~~bash
rm -rf /tmp/g1_terrain_scene_diagnostic
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --output /tmp/g1_terrain_scene_diagnostic \
  --grail-limit 1
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  /tmp/g1_terrain_scene_diagnostic
~~~

Expected build summary: schema v2, `clips=2`, `scenes=14`. Expected validator
summary: `support_dims=3 scenes=14 source_rows=0`. The builder has scanned all
1,769 candidate meshes for stable GRAIL selection even though it converted only
one GRAIL motion into the diagnostic database.

- [ ] **Step 5: Probe every diagnostic G1HF/v2 scene through the C++ consumer**

The artifact-owned C++ test binary accepts a G1TF path, one G1HF path, and the
exact expected G1HF version after its self-tests. Run it once per ordered scene:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY' \
  > /tmp/g1_scene_ids.txt
import json
with open("/tmp/g1_terrain_scene_diagnostic/scenes/index.json") as stream:
    index = json.load(stream)
for scene_id in index["scene_ids"]:
    print(scene_id)
PY
while IFS= read -r scene_id
do
  /tmp/test_terrain_runtime_strict \
    /tmp/g1_terrain_scene_diagnostic/terrain_features.bin \
    "/tmp/g1_terrain_scene_diagnostic/scenes/${scene_id}/terrain.bin" 2
done < /tmp/g1_scene_ids.txt
~~~

Expected: 14 exit-0 probes, with G1TF rows equal to the diagnostic database
frame count and every terrain reporting G1HF/v2. Exact independent Python/C++
byte parity is already locked by Task 5's awkward-metadata oracle; this loop
checks every produced scene through the real loader/sampler. When the sibling runtime plan
has added G1SP/G1WM arguments to this same probe, use its extended exact CLI as
well; that extension must not replace this G1HF parity loop.

- [ ] **Step 6: Re-run atomic-failure drills before touching the real output**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts.ArtifactTests.test_candidate_failure_preserves_every_byte_of_previous_output \
  tests.python.test_artifacts.ArtifactTests.test_publish_rename_failure_restores_previous_directory \
  tests.python.test_artifacts.ArtifactTests.test_publish_rejects_scene_byte_corruption_before_rename -v
~~~

Expected: all three pass and each temporary last-good sentinel remains exact.
Do not simulate rename failure against `resources/g1_terrain/`.

- [ ] **Step 7: Build, full-source validate, and atomically publish the full pack**

Run without `--grail-limit`. `bash -o pipefail` ensures a builder failure is not
masked by `tee`:

~~~bash
bash -o pipefail -c '/usr/bin/time -v \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --output resources/g1_terrain \
  2>&1 | tee /tmp/g1_full_scene_build.log'
~~~

Expected only after the staged full-source validator succeeds:

~~~text
BUILT g1-terrain-artifacts/v2 frames=459682 clips=1770 scenes=14 output=resources/g1_terrain
~~~

Before that line, the publisher has written a sibling staging tree, reloaded
every byte, validated every scene, recomputed every source pose/contact/feature/
support row against its exact surface, fsynced the tree, and only then renamed
the previous output aside and the candidate into place. Any nonzero exit means
the prior `resources/g1_terrain/` must still be authoritative; stop and diagnose
rather than deleting or rebuilding in place.

- [ ] **Step 8: Revalidate the published full pack, including all source rows**

Run both modes against the bytes at their final path:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  resources/g1_terrain
bash -o pipefail -c '/usr/bin/time -v \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  resources/g1_terrain \
  --full-source-validation \
  2>&1 | tee /tmp/g1_full_source_validation.log'
~~~

Expected exact normal summary fields:

~~~text
VALID g1-terrain-artifacts/v2 frames=459682 clips=1770 bones=31 terrain_dims=4 support_dims=3 scenes=14 source_rows=0
~~~

Expected full-source summary differs only in `source_rows=459682`. Both must
exit 0; every source range remains contiguous, every Takara support row is
exact zero, every GRAIL row was recomputed from its matching exact triangle
surface, and the four selected GRAIL scene bases remain locked.

- [ ] **Step 9: Assert publication structure, counts, contracts, and hashes**

Run this read-only audit:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import hashlib
import json
import os
import struct

root = "resources/g1_terrain"
with open(os.path.join(root, "manifest.json")) as stream:
    manifest = json.load(stream)
with open(os.path.join(root, "scenes", "index.json")) as stream:
    index = json.load(stream)
expected_ids = [
    "grail-curb-default", "grail-curb-low", "grail-curb-medium",
    "grail-curb-high", "stairs-shallow", "stairs-standard",
    "stairs-unseen-variable", "ramp-05-up-down", "ramp-10-up-down",
    "ramp-15-stress", "cross-slope-05", "cross-slope-10",
    "mixed-multilevel", "blocked-course",
]
assert manifest["schema"] == "g1-terrain-artifacts/v2"
assert manifest["database_frames"] == 459682
assert manifest["total_clips"] == 1770
assert manifest["grail_clips"] == 1769
assert manifest["skipped_clips"] == 0
assert manifest["diagnostic_mode"] is False
assert manifest["terrain_feature_distances_m"] == [0.25, 0.5, 0.75, 1.0]
assert manifest["support_dimensions"] == 3
assert manifest["sidecars"]["terrain_support"]["columns"] == [
    "source_root_height_m", "source_left_toe_height_m",
    "source_right_toe_height_m",
]
assert index["scene_ids"] == expected_ids
assert set(os.listdir(root)) == {
    "database.bin", "terrain_features.bin", "terrain_support.bin",
    "manifest.json", "validation.json", "scenes",
}

def digest(relative):
    value = hashlib.sha256()
    with open(os.path.join(root, relative), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

assert digest(manifest["database"]["path"]) == manifest["database"]["sha256"]
for descriptor in manifest["sidecars"].values():
    assert digest(descriptor["path"]) == descriptor["sha256"]
assert digest(manifest["scene_index"]["path"]) \
    == manifest["scene_index"]["sha256"]
assert digest(manifest["validation_file"]["path"]) \
    == manifest["validation_file"]["sha256"]
expected_grail = {
    "grail-curb-default": "terrain_curbs__curb_000__000",
    "grail-curb-low": "terrain_curbs__curb_186__004",
    "grail-curb-medium": "terrain_curbs__curb_022__001",
    "grail-curb-high": "terrain_curbs__curb_165__006",
}
for scene_id in expected_ids:
    scene_root = os.path.join(root, "scenes", scene_id)
    assert set(os.listdir(scene_root)) == {
        "scene.json", "terrain.bin", "terrain.obj", "walkability.bin",
    }
    with open(os.path.join(scene_root, "scene.json")) as stream:
        scene = json.load(stream)
    with open(os.path.join(scene_root, "terrain.bin"), "rb") as stream:
        header = struct.unpack("<4sIII4f", stream.read(32))
    assert header[0:2] == (b"G1HF", 2)
    assert scene["heightfield"]["diagonal"] \
        == "min-x-min-z_to_max-x-max-z"
    assert scene["heightfield"]["cell_size_m"] == header[6]
    assert abs(header[6] - 0.02) < 1e-8
    for descriptor, filename in (
        (scene["heightfield"], "terrain.bin"),
        (scene["mesh"], "terrain.obj"),
        (scene["walkability"], "walkability.bin"),
    ):
        value = hashlib.sha256()
        with open(os.path.join(scene_root, filename), "rb") as stream:
            value.update(stream.read())
        assert value.hexdigest() == descriptor["sha256"]
    if scene_id in expected_grail:
        assert scene["provenance"]["source_ids"][0] == expected_grail[scene_id]
print("VALID full-pack-audit frames=459682 clips=1770 scenes=14")
PY
~~~

Expected: `VALID full-pack-audit frames=459682 clips=1770 scenes=14`.

- [ ] **Step 10: Probe every published scene through the C++ G1HF consumer**

Regenerate the ordered ID list from the final index and run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY' \
  > /tmp/g1_published_scene_ids.txt
import json
with open("resources/g1_terrain/scenes/index.json") as stream:
    for scene_id in json.load(stream)["scene_ids"]:
        print(scene_id)
PY
while IFS= read -r scene_id
do
  /tmp/test_terrain_runtime_strict \
    resources/g1_terrain/terrain_features.bin \
    "resources/g1_terrain/scenes/${scene_id}/terrain.bin" 2
done < /tmp/g1_published_scene_ids.txt
~~~

Expected: 14 exit-0 probes, each observing `459682x4` feature rows and
G1HF/v2 fixed-diagonal terrain. Run the sibling runtime plan's extended
G1SP/G1WM/scene-catalog probe immediately afterward when that plan is present.

- [ ] **Step 11: Prove legacy/user resources and worktree state were preserved**

Run:

~~~bash
sha256sum -c /tmp/g1_legacy_artifacts_before.sha256
git status --short > /tmp/g1_status_after_artifact_build.txt
diff -u \
  /tmp/g1_status_before_artifact_build.txt \
  /tmp/g1_status_after_artifact_build.txt
find resources -maxdepth 1 \
  \( -name '.g1_terrain.staging-*' -o -name '.g1_terrain.previous-*' \) \
  -print
~~~

Expected: both legacy hashes print `OK`, the status diff is empty, and `find`
prints nothing. The old generated pack is allowed to differ because the task
explicitly replaces it only after validation; all other user-owned files remain
untouched.

- [ ] **Step 12: Leave generated output uncommitted and record Gate B evidence**

Run:

~~~bash
git status --short
git log --oneline -12
~~~

Expected: `resources/g1_terrain/`, `/tmp` logs, diagnostic output, and other
generated media do not appear in staged/tracked changes. Do not `git add -f`
the pack and do not create a commit for generated bytes. Record the Python test
count, four C++ configurations, diagnostic summary, full build summary, normal
and full-source validator summaries, full-pack audit line, and preservation
hash results in the implementation handoff.
