import contextlib
import io
import struct
import tempfile
import unittest

from resources import check_g1_runtime_log as runtime_log
from resources.check_g1_runtime_log import (
    CSV_COLUMNS,
    GATE_A_COLUMNS,
    RUNTIME_COLUMNS,
    check_rows,
    compare_control,
    diagnose_gate_a,
    read_rows,
)


PRE_SOLE_RUNTIME_SUFFIX = (
    "source_name", "source_terrain", "source_index", "continuation_cost",
    "source_root_height", "source_left_toe_height", "source_right_toe_height",
    "runtime_support_root_height", "runtime_support_left_toe_height",
    "runtime_support_right_toe_height", "support_root_delta",
    "support_left_toe_delta", "support_right_toe_delta", "support_height",
    "support_velocity", "support_source", "airborne_frames", "left_contact",
    "right_contact", "support_retargeted_hips_y", "ik_adjusted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed", "route_waypoint",
    "route_complete", "route_target_height", "scene_generation",
    "scene_frame", "scene_reset_count", "scene_switch_failed",
    "motion_pack_load_count", "model_load_count", "model_unload_count",
    "live_model_count",
)
SOLE_DIAGNOSTIC_SUFFIX = (
    "left_sole_clearance_0", "left_sole_clearance_1",
    "left_sole_clearance_2", "left_sole_clearance_3",
    "right_sole_clearance_0", "right_sole_clearance_1",
    "right_sole_clearance_2", "right_sole_clearance_3",
    "left_sole_min_clearance", "right_sole_min_clearance",
    "sole_min_clearance", "left_stance_slip", "right_stance_slip",
    "left_stance_slip_reset", "right_stance_slip_reset",
)
BANKED_MOTION_SUFFIX = (
    "terrain4", "terrain5", "terrain6", "terrain7",
    "terrain8", "terrain9", "terrain10", "terrain11",
    "terrain_root_point_x", "terrain_root_point_y", "terrain_root_point_z",
    *(f"terrain_center_point{sample}_{axis}"
      for sample in range(8) for axis in "xyz"),
    *(f"terrain_left_point{sample}_{axis}"
      for sample in range(4) for axis in "xyz"),
    *(f"terrain_right_point{sample}_{axis}"
      for sample in range(4) for axis in "xyz"),
    "requested_family", "active_family", "source_family",
    "direction_mask", "speed_mask", "elevation_mode",
    "classifier_confidence", "bank_transition", "bank_transition_reason",
    "eligible_frame_count", "evaluated_frame_count",
    "considered_bound_count", "skipped_bound_count",
    "empty_compatible_set",
)
RUNTIME_SUFFIX = (
    PRE_SOLE_RUNTIME_SUFFIX + SOLE_DIAGNOSTIC_SUFFIX + BANKED_MOTION_SUFFIX
)

TASK11_SCENE_IDS = (
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


def row(frame, database_frame, **changes):
    values = {
        "frame": str(frame),
        "fixed_dt": "0.04",
        "scene_id": "grail-curb-default",
        "mode": "terrain",
        "route": "curb-forward",
        "query_bits_hex": "00000000" * 39,
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
        query_values = [0.0] * 27 + [
            float(values[f"terrain{sample}"]) for sample in range(4)] + [
            0.0 for _ in range(8)]
        values["query_bits_hex"] = "".join(
            struct.pack(">f", value).hex() for value in query_values)
    return values


def sequential_rows(count, start=0):
    rows = []
    for frame in range(count):
        current = start + frame
        query = current if frame == 0 else current - 1
        rows.append(row(
            frame, current,
            query_database_frame=query,
            selected_database_frame=query,
        ))
    return rows


def float32_offset(value, ulps):
    bits = int.from_bytes(struct.pack(">f", value), "big")
    return struct.unpack(">f", (bits + ulps).to_bytes(4, "big"))[0]


def runtime_row(frame, **changes):
    database_frame = 100 + frame
    query_frame = database_frame if frame == 0 else database_frame - 1
    values = row(
        frame, database_frame,
        query_database_frame=query_frame,
        selected_database_frame=query_frame,
    )
    values.update({
        "source_name": "terrain_curbs__fixture", "source_terrain": "fixture",
        "source_index": "1", "continuation_cost": "2.0",
        "source_root_height": "0", "source_left_toe_height": "0",
        "source_right_toe_height": "0", "runtime_support_root_height": "0",
        "runtime_support_left_toe_height": "0",
        "runtime_support_right_toe_height": "0", "support_root_delta": "0",
        "support_left_toe_delta": "0", "support_right_toe_delta": "0",
        "support_height": "0", "support_velocity": "0",
        "support_source": "both", "airborne_frames": "0",
        "left_contact": "1", "right_contact": "1",
        "support_retargeted_hips_y": "0.8", "ik_adjusted_hips_y": "0.8",
        "simulation_x": "0", "simulation_z": str(frame * .02),
        "walkability_class": "1", "blocked": "0",
        "blocked_reason": "clear", "blocked_distance": "3.4e38",
        "blocked_point_x": "0", "blocked_point_z": "0",
        "commanded_speed": ".5", "applied_speed": ".5",
        "route_waypoint": "1", "route_complete": "0",
        "route_target_height": ".36", "scene_generation": "0",
        "scene_frame": str(frame), "scene_reset_count": "1",
        "scene_switch_failed": "0", "motion_pack_load_count": "1",
        "model_load_count": "1", "model_unload_count": "0",
        "live_model_count": "1", "matching_enabled": "1",
        "support_retargeting_enabled": "1", "ik_enabled": "0",
        "adjustment_y": "0", "clamp_y": "0", "fixed_dt": ".04",
        "mode": "route", "route": "fixture-route", "scene_id": "fixture",
    })
    values.update({key: str(value) for key, value in changes.items()})
    searched = int(values["searched"])
    left_clearances = (0.010, 0.011, 0.012, 0.013)
    right_clearances = (0.020, 0.021, 0.022, 0.023)
    for probe, clearance in enumerate(left_clearances):
        values[f"left_sole_clearance_{probe}"] = str(clearance)
    for probe, clearance in enumerate(right_clearances):
        values[f"right_sole_clearance_{probe}"] = str(clearance)
    values["left_sole_min_clearance"] = str(min(left_clearances))
    values["right_sole_min_clearance"] = str(min(right_clearances))
    values["sole_min_clearance"] = str(
        min(left_clearances + right_clearances))
    values["left_stance_slip"] = "0"
    values["right_stance_slip"] = "0"
    reset = int(values["scene_frame"]) == 0
    values["left_stance_slip_reset"] = str(int(reset))
    values["right_stance_slip_reset"] = str(int(reset))
    for sample in range(4, 12):
        values[f"terrain{sample}"] = "0"
    for axis in "xyz":
        values[f"terrain_root_point_{axis}"] = "0"
    for label, count in (("center", 8), ("left", 4), ("right", 4)):
        for sample in range(count):
            for axis in "xyz":
                values[f"terrain_{label}_point{sample}_{axis}"] = "0"
    values.update({
        "requested_family": "flat", "active_family": "flat",
        "source_family": "flat", "direction_mask": "2",
        "speed_mask": "2", "elevation_mode": "0",
        "classifier_confidence": "1", "bank_transition": "0",
        "bank_transition_reason": "retained_same",
        "eligible_frame_count": str(searched),
        "evaluated_frame_count": str(searched),
        "considered_bound_count": str(searched), "skipped_bound_count": "0",
        "empty_compatible_set": "0",
    })
    # Appended-schema overrides intentionally win after deterministic defaults.
    values.update({
        key: str(value) for key, value in changes.items()
        if key in SOLE_DIAGNOSTIC_SUFFIX + BANKED_MOTION_SUFFIX
    })
    if "query_bits_hex" not in changes:
        query_values = [0.0] * 27 + [
            float(values[f"terrain{sample}"]) for sample in range(12)]
        values["query_bits_hex"] = "".join(
            struct.pack(">f", value).hex() for value in query_values)
    return {name: values[name] for name in RUNTIME_COLUMNS}


def gate_c_rows():
    rows = []
    for frame in range(170):
        if frame < 20:
            height = 0.0
        elif frame < 56:
            height = min(.36, (frame - 20) * .01)
        elif frame < 110:
            height = .36
        elif frame < 146:
            height = max(0.0, .36 - (frame - 110) * .01)
        else:
            height = 0.0
        rows.append(runtime_row(
            frame, terrain0=(.12 if frame >= 10 else 0),
            runtime_support_root_height=height,
            runtime_support_left_toe_height=height,
            runtime_support_right_toe_height=height,
            support_root_delta=height, support_left_toe_delta=height,
            support_right_toe_delta=height, support_height=height,
            support_retargeted_hips_y=.8 + height,
            ik_adjusted_hips_y=.8 + height, rendered_hips_y=.8 + height,
            route_complete=int(frame >= 146)))
    return rows


def mixed_multilevel_rows():
    rows = []
    for frame in range(271):
        z = frame * .04
        if z < 3.20:
            height = z * .10
        elif z < 6.20:
            height = .32
        elif z < 6.80:
            height = .40
        elif z < 7.40:
            height = .28
        elif z < 8.00:
            height = .32
        elif z <= 9.8148:
            height = max(0.0, .32 * (9.8148 - z) / 1.8148)
        else:
            height = 0.0
        continued = 90 <= frame < 140
        rows.append(runtime_row(
            frame, scene_id="mixed-multilevel", route="full-course",
            simulation_z=z, terrain0=(.12 if frame >= 5 else 0),
            source_index=(1 if continued else 0),
            source_terrain=("fixture" if continued else "flat"),
            runtime_support_root_height=height,
            runtime_support_left_toe_height=height,
            runtime_support_right_toe_height=height,
            support_root_delta=height, support_left_toe_delta=height,
            support_right_toe_delta=height, support_height=height,
            raw_selected_hips_y=.8 + height,
            inertialized_hips_y=.8 + height,
            support_retargeted_hips_y=.8 + height,
            ik_adjusted_hips_y=.8 + height, rendered_hips_y=.8 + height,
            route_target_height=.32, route_complete=int(z > 9.8148)))
    return rows


def mutate_mixed_region(rows, region, offset, field, value):
    bounds = {
        "elevated": (3.20, 6.20),
        "block-2": (6.85, 7.35),
        "ramp": (8.00, 9.8148),
        "base": (9.8148, float("inf")),
    }
    lower, upper = bounds[region]
    candidates = [
        item for item in rows
        if lower <= float(item["simulation_z"]) <= upper
    ]
    targets = candidates if region == "block-2" else [candidates[offset]]
    for item in targets:
        item[field] = str(value)


def mixed_support_span_rows(start_z):
    rows = mixed_multilevel_rows()
    rows[79]["runtime_support_root_height"] = "0"
    rows[80]["simulation_z"] = str(start_z)
    rows[154]["simulation_z"] = "6.15"
    return rows


def gate_d_rows():
    rows = []
    for frame in range(100):
        blocked = frame >= 20
        speed = max(0.0, .5 - max(0, frame - 20) * .05)
        rows.append(runtime_row(
            frame, scene_id="blocked-course", route="wall-safe-stop",
            blocked=int(blocked),
            blocked_reason=("blocked-cell" if blocked else "clear"),
            blocked_distance=(.03 if blocked else 3.4e38),
            applied_speed=speed, simulation_z=min(frame * .02, .58),
            route_complete=int(frame >= 60)))
    return rows


def scene_segment_rows(segments):
    rows = []
    frame = 0
    for generation, (scene_id, dwell) in enumerate(segments):
        for scene_frame in range(dwell):
            rows.append(runtime_row(
                frame, scene_id=scene_id, mode="scene-cycle", route="",
                scene_generation=generation, scene_frame=scene_frame,
                scene_reset_count=generation + 1,
                model_load_count=generation + 1,
                model_unload_count=generation, live_model_count=1,
                route_waypoint=0, commanded_speed=0, applied_speed=0))
            frame += 1
    return rows


def scene_cycle_rows(ids, dwell=2, cycles=2):
    return scene_segment_rows([
        (scene_id, dwell) for scene_id in ids * cycles
    ])


def write_runtime_rows(stream, rows):
    stream.write(",".join(RUNTIME_COLUMNS) + "\n")
    for item in rows:
        stream.write(",".join(item[name] for name in RUNTIME_COLUMNS) + "\n")
    stream.flush()


def set_terrain(values, sample, value):
    values[f"terrain{sample}"] = str(value)
    query_values = [0.0] * 27 + [
        float(values[f"terrain{query_sample}"])
        for query_sample in range(12)
    ]
    values["query_bits_hex"] = "".join(
        struct.pack(">f", query_value).hex() for query_value in query_values)


class RuntimeLogTests(unittest.TestCase):
    def test_runtime_columns_append_after_gate_a(self):
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[-len(RUNTIME_SUFFIX):]), RUNTIME_SUFFIX)
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[:-len(RUNTIME_SUFFIX)]), GATE_A_COLUMNS)

    def test_sole_columns_append_without_mutating_the_existing_runtime_prefix(self):
        expected_prefix = tuple(GATE_A_COLUMNS) + PRE_SOLE_RUNTIME_SUFFIX
        self.assertEqual(tuple(RUNTIME_COLUMNS[:len(expected_prefix)]),
                         expected_prefix)
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[
                len(expected_prefix):len(expected_prefix) +
                len(SOLE_DIAGNOSTIC_SUFFIX)]),
            SOLE_DIAGNOSTIC_SUFFIX)

    def test_banked_motion_columns_append_after_the_entire_old_runtime_schema(self):
        old_schema = (
            tuple(GATE_A_COLUMNS) + PRE_SOLE_RUNTIME_SUFFIX +
            SOLE_DIAGNOSTIC_SUFFIX
        )
        self.assertEqual(tuple(RUNTIME_COLUMNS[:len(old_schema)]), old_schema)
        self.assertEqual(tuple(RUNTIME_COLUMNS[len(old_schema):]),
                         BANKED_MOTION_SUFFIX)

    def test_banked_runtime_snapshot_has_exactly_39_finite_float32_values(self):
        changes = {f"terrain{sample}": sample / 32.0
                   for sample in range(12)}
        item = runtime_row(0, **changes)
        self.assertEqual(len(item["query_bits_hex"]), 312)
        self.assertEqual(item["query_bits_hex"], "".join(
            struct.pack(">f", value).hex()
            for value in [0.0] * 27 + [sample / 32.0
                                       for sample in range(12)]))
        check_rows([item])

    def test_rejects_nonfinite_complete_descriptor_provenance(self):
        for field in (
                "terrain11", "terrain_root_point_y",
                "terrain_center_point7_z", "terrain_left_point3_x",
                "terrain_right_point3_y"):
            item = runtime_row(0)
            item[field] = "nan"
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"non-finite {field}"):
                    check_rows([item])

    def test_rejects_invalid_banked_motion_domains_and_relationships(self):
        cases = (
            ("requested_family", "unknown", "requested_family"),
            ("active_family", "none", "active_family"),
            ("source_family", "fixture", "source family"),
            ("direction_mask", "0", "direction_mask"),
            ("direction_mask", str(0x002 | 0x008), "direction_mask"),
            ("speed_mask", "4", "speed_mask"),
            ("elevation_mode", "2", "elevation_mode"),
            ("classifier_confidence", "1.01", "classifier_confidence"),
            ("bank_transition", "2", "bank_transition"),
            ("bank_transition_reason", "invented", "bank_transition_reason"),
            ("eligible_frame_count", "-1", "eligible_frame_count"),
            ("evaluated_frame_count", "2", "evaluated_frame_count"),
            ("skipped_bound_count", "2", "skipped_bound_count"),
            ("empty_compatible_set", "2", "empty_compatible_set"),
        )
        for field, value, message in cases:
            item = runtime_row(0)
            item[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    check_rows([item])

        item = runtime_row(0, active_family="slope", source_family="flat")
        with self.assertRaisesRegex(ValueError, "source_family.*active_family"):
            check_rows([item])

    def test_explicit_empty_set_allows_checked_repeated_last_pose(self):
        rows = [runtime_row(0)]
        rows.append(runtime_row(
            1,
            query_database_frame=100,
            selected_database_frame=100,
            database_frame=100,
            searched=1,
            applied_speed=0,
            active_family="stair",
            source_family="flat",
            requested_family="stair",
            elevation_mode=1,
            bank_transition=1,
            bank_transition_reason="confirmed",
            eligible_frame_count=0,
            evaluated_frame_count=0,
            considered_bound_count=3,
            skipped_bound_count=3,
            empty_compatible_set=1,
        ))
        rows.append(runtime_row(
            2,
            query_database_frame=100,
            selected_database_frame=100,
            database_frame=100,
            searched=1,
            applied_speed=0,
            active_family="stair",
            source_family="flat",
            requested_family="stair",
            elevation_mode=1,
            bank_transition=0,
            bank_transition_reason="retained_same",
            eligible_frame_count=0,
            evaluated_frame_count=0,
            considered_bound_count=3,
            skipped_bound_count=3,
            empty_compatible_set=1,
        ))
        check_rows(rows)

        mutations = (
            ("searched", "0", "must be searched"),
            ("applied_speed", ".1", "applied_speed"),
            ("eligible_frame_count", "1", "eligible_frame_count"),
            ("evaluated_frame_count", "1", "evaluated_frame_count"),
            ("database_frame", "101", "preserve the last accepted pose"),
        )
        for field, value, message in mutations:
            changed = [dict(item) for item in rows]
            changed[1][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    check_rows(changed)

    def test_requires_ordered_physical_sole_clearances_and_consistent_minima(self):
        item = runtime_row(0)
        self.assertEqual(
            tuple(item)[
                -len(BANKED_MOTION_SUFFIX) - len(SOLE_DIAGNOSTIC_SUFFIX):
                -len(BANKED_MOTION_SUFFIX)],
            SOLE_DIAGNOSTIC_SUFFIX)
        check_rows([item])

        for field, message in (
                ("left_sole_min_clearance", "left sole minimum"),
                ("right_sole_min_clearance", "right sole minimum"),
                ("sole_min_clearance", "global sole minimum")):
            changed = dict(item)
            changed[field] = "0.5"
            with self.assertRaisesRegex(ValueError, message):
                check_rows([changed])

    def test_rejects_nonfinite_or_negative_sole_and_slip_values(self):
        for field, value, message in (
                ("left_sole_clearance_2", "nan", "non-finite"),
                ("right_sole_min_clearance", "inf", "non-finite"),
                ("left_stance_slip", "-0.001", "nonnegative"),
                ("right_stance_slip_reset", "2", "must be 0 or 1")):
            item = runtime_row(0)
            item[field] = value
            with self.assertRaisesRegex(ValueError, message):
                check_rows([item])

    def test_slip_history_resets_on_contact_edges_and_scene_generation(self):
        rows = [runtime_row(frame) for frame in range(5)]
        rows[1].update({
            "left_stance_slip": "0.01",
            "right_stance_slip": "0.02",
        })
        rows[2].update({
            "left_contact": "0", "left_stance_slip": "0",
            "left_stance_slip_reset": "1",
            "right_stance_slip": "0.03",
        })
        rows[3].update({
            "left_contact": "1", "left_stance_slip": "0",
            "left_stance_slip_reset": "1",
            "right_stance_slip": "0.04",
        })
        rows[4].update({
            "scene_generation": "1", "scene_frame": "0",
            "scene_reset_count": "2",
            "left_stance_slip": "0", "right_stance_slip": "0",
            "left_stance_slip_reset": "1",
            "right_stance_slip_reset": "1",
        })
        check_rows(rows)

        for field, message in (
                ("left_stance_slip_reset", "contact edge"),
                ("right_stance_slip_reset", "scene reset")):
            changed = [dict(item) for item in rows]
            target = 2 if field.startswith("left") else 4
            changed[target][field] = "0"
            with self.assertRaisesRegex(ValueError, message):
                check_rows(changed)

    def test_surface_height_names_are_frozen_before_runtime_schema(self):
        for name in (
                "runtime_root_surface_height",
                "runtime_left_toe_surface_height",
                "runtime_right_toe_surface_height"):
            self.assertIn(name, CSV_COLUMNS)
        for old_name in (
                "runtime_root_height",
                "runtime_left_toe_height",
                "runtime_right_toe_height"):
            self.assertNotIn(old_name, CSV_COLUMNS)

    def test_gate_c_accepts_persistent_landing_and_descent(self):
        self.assertGreaterEqual(
            runtime_log.check_gate_c(gate_c_rows())["landing_frames"], 50)

    def test_gate_c_accepts_raw_source_jump_with_smooth_final_hips(self):
        rows = gate_c_rows()
        rows[80]["raw_selected_hips_y"] = "1.3"
        report = runtime_log.check_gate_c(rows)
        self.assertLessEqual(report["maximum_rendered_hips_step"], 0.05)

    def test_gate_c_rejects_rendered_hips_step_over_limit(self):
        rows = gate_c_rows()
        for name in (
                "support_retargeted_hips_y", "ik_adjusted_hips_y",
                "rendered_hips_y"):
            rows[80][name] = "1.3"
        with self.assertRaisesRegex(ValueError, "rendered Hips"):
            runtime_log.check_gate_c(rows)

    def test_gate_c_accepts_exactly_fifty_combined_landing_rows(self):
        rows = gate_c_rows()
        for item in rows[54:62]:
            item["runtime_support_left_toe_height"] = ".50"
        self.assertEqual(runtime_log.check_gate_c(rows)["landing_frames"], 50)

    def test_gate_c_rejects_forty_nine_or_split_landing_rows(self):
        cases = []
        rows = gate_c_rows()
        for item in rows[54:63]:
            item["runtime_support_left_toe_height"] = ".50"
        cases.append(rows)
        rows = gate_c_rows()
        rows[80]["runtime_support_left_toe_height"] = ".50"
        cases.append(rows)
        for rows in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "landing block"):
                    runtime_log.check_gate_c(rows)

    def test_mixed_checker_requires_elevated_blocks_and_return_ramp(self):
        report = runtime_log.check_mixed_multilevel(mixed_multilevel_rows())
        self.assertGreaterEqual(report["elevated_matching_frames"], 50)
        for mutation, diagnostic in (
                (("elevated", 25, "matching_enabled", "0"),
                 "elevated matching"),
                (("elevated", 25, "runtime_support_left_toe_height", ".50"),
                 "elevated matching"),
                (("block-2", 4, "runtime_support_root_height", ".40"),
                 "block plateaus"),
                (("ramp", 8, "runtime_support_root_height", ".40"),
                 "return ramp"),
                (("base", 2, "runtime_support_root_height", ".08"),
                 "returned to base")):
            rows = mixed_multilevel_rows()
            mutate_mixed_region(rows, *mutation)
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_mixed_multilevel(rows)

    def test_mixed_checker_accepts_flat_elevated_source_metadata(self):
        rows = mixed_multilevel_rows()
        for item in rows:
            if 3.20 <= float(item["simulation_z"]) <= 6.20:
                item["source_index"] = "0"
                item["source_terrain"] = "flat"
        self.assertGreaterEqual(
            runtime_log.check_mixed_multilevel(rows)["elevated_matching_frames"],
            50)

    def test_mixed_checker_accepts_exactly_fifty_elevated_rows(self):
        rows = mixed_multilevel_rows()
        for item in rows[80:105]:
            item["matching_enabled"] = "0"
        self.assertEqual(
            runtime_log.check_mixed_multilevel(rows)["elevated_matching_frames"],
            50)

    def test_mixed_checker_rejects_forty_nine_elevated_rows(self):
        rows = mixed_multilevel_rows()
        for item in rows[80:106]:
            item["matching_enabled"] = "0"
        with self.assertRaisesRegex(ValueError, "elevated matching"):
            runtime_log.check_mixed_multilevel(rows)

    def test_mixed_checker_decouples_endpoint_crossing_from_support_span(self):
        rows = mixed_multilevel_rows()
        rows[154]["runtime_support_root_height"] = ".40"
        report = runtime_log.check_mixed_multilevel(rows)
        self.assertGreaterEqual(report["elevated_span_m"], 2.95)

    def test_mixed_checker_enforces_exact_support_span_boundary(self):
        report = runtime_log.check_mixed_multilevel(
            mixed_support_span_rows(3.20))
        self.assertAlmostEqual(report["elevated_span_m"], 2.95, places=12)
        with self.assertRaisesRegex(ValueError, "support span"):
            runtime_log.check_mixed_multilevel(mixed_support_span_rows(3.21))

    def test_gate_d_accepts_safe_stop_and_rejects_blocked_footprint(self):
        self.assertGreaterEqual(
            runtime_log.check_gate_d(gate_d_rows())["stopped_frames"], 25)
        rows = gate_d_rows()
        rows[50]["walkability_class"] = "0"
        with self.assertRaisesRegex(ValueError, "entered blocked"):
            runtime_log.check_gate_d(rows)

    def test_gate_d_requires_twenty_pre_block_baseline_rows(self):
        self.assertGreaterEqual(
            runtime_log.check_gate_d(gate_d_rows())["stopped_frames"], 25)

        rows = gate_d_rows()
        rows[19]["blocked"] = "1"
        rows[19]["blocked_reason"] = "blocked-cell"
        rows[19]["blocked_distance"] = ".03"
        with self.assertRaisesRegex(ValueError, "20 pre-block"):
            runtime_log.check_gate_d(rows)

    def test_gate_d_binds_stop_hold_to_initial_stopped_blocked_event(self):
        rows = gate_d_rows()
        for item in rows[35:60]:
            item["applied_speed"] = ".1"
        with self.assertRaisesRegex(ValueError, "25 consecutive"):
            runtime_log.check_gate_d(rows)

    def test_gate_d_accepts_block_clear_during_uninterrupted_stop(self):
        rows = gate_d_rows()
        for item in rows[31:]:
            item["blocked"] = "0"
            item["blocked_reason"] = "clear"
        self.assertGreaterEqual(
            runtime_log.check_gate_d(rows)["stopped_frames"], 25)

    def test_gate_f_accepts_two_ordered_cycles_and_one_motion_load(self):
        ids = ["one", "two", "three"]
        self.assertEqual(
            runtime_log.check_gate_f(scene_cycle_rows(ids), ids), {
                "frames": 12,
                "generations": 6,
                "complete_cycles": 2,
                "motion_pack_loads": 1,
                "model_loads": 6,
                "model_unloads_before_final_cleanup": 5,
            })

    def test_gate_f_rejects_invalid_expected_scene_ids(self):
        rows = scene_cycle_rows(["one", "two"])
        for expected, diagnostic in (
                ([], "nonempty"),
                (["one", "one"], "unique"),
                (["one", "wrong"], "ordered"),
                (["one", ""], "nonempty")):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_f(rows, expected)

    def test_gate_f_rejects_generation_dwell_identity_and_order_drift(self):
        cases = []

        rows = scene_cycle_rows(["one", "two"])
        for item in rows:
            item["scene_generation"] = str(
                int(item["scene_generation"]) + 1)
            item["scene_reset_count"] = str(
                int(item["scene_reset_count"]) + 1)
        cases.append((rows, "generation 0"))

        rows = scene_cycle_rows(["one", "two"])
        for item in rows[2:]:
            item["scene_generation"] = str(
                int(item["scene_generation"]) + 1)
            item["scene_reset_count"] = str(
                int(item["scene_reset_count"]) + 1)
        cases.append((rows, "scene_generation"))

        rows = scene_segment_rows([
            ("one", 1), ("two", 2), ("one", 1), ("two", 2),
        ])
        cases.append((rows, "dwell"))

        rows = scene_cycle_rows(["one", "two"])
        rows[1]["scene_id"] = "different"
        cases.append((rows, "constant scene"))

        rows = scene_cycle_rows(["one", "two"])
        for item in rows[2:4]:
            item["scene_id"] = "one"
        cases.append((rows, "ordered"))

        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_requires_two_complete_cycles(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            runtime_log.check_gate_f(
                scene_cycle_rows(["one", "two"], cycles=1),
                ["one", "two"])

        rows = scene_segment_rows([
            ("one", 2), ("two", 2), ("one", 2), ("two", 2),
            ("one", 2),
        ])
        with self.assertRaisesRegex(ValueError, "complete ordered cycles"):
            runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_reset_and_model_counter_drift(self):
        cases = []

        rows = scene_cycle_rows(["one", "two"])
        for item in rows:
            item["scene_reset_count"] = str(
                int(item["scene_reset_count"]) + 1)
        cases.append((rows, "scene_reset_count"))

        rows = scene_cycle_rows(["one", "two"])
        rows[0]["model_load_count"] = "2"
        cases.append((rows, "model_load_count"))

        rows = scene_cycle_rows(["one", "two"])
        rows[0]["model_unload_count"] = "1"
        cases.append((rows, "model_unload_count"))

        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_interior_model_counter_spike(self):
        rows = scene_cycle_rows(["one", "two"])
        rows[1].update({
            "model_load_count": "2",
            "model_unload_count": "1",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(ValueError, "generation 0.*model_load_count"):
            runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_model_counter_decrease(self):
        rows = scene_cycle_rows(["one", "two"])
        rows[3].update({
            "model_load_count": "1",
            "model_unload_count": "0",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_final_row_model_counter_corruption(self):
        rows = scene_cycle_rows(["one", "two"])
        rows[-1].update({
            "model_load_count": "5",
            "model_unload_count": "4",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(ValueError, "generation 3.*model_load_count"):
            runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_live_model_count_difference_mismatch(self):
        rows = scene_cycle_rows(["one", "two"])
        rows[1].update({
            "model_load_count": "2",
            "model_unload_count": "0",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(
                ValueError,
                "live_model_count.*model_load_count.*model_unload_count"):
            runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_rejects_first_row_and_activation_contract_drift(self):
        for name, value, diagnostic in (
                ("route_waypoint", 1, "route_waypoint"),
                ("blocked", 1, "blocked"),
                ("airborne_frames", 2, "airborne_frames"),
                ("mode", "terrain", "scene-cycle mode"),
                ("scene_switch_failed", 1, "switch-failure")):
            rows = scene_cycle_rows(["one", "two"])
            rows[0][name] = str(value)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_f(rows, ["one", "two"])

    def test_gate_f_preserves_schema_25hz_and_ik_off_contracts(self):
        rows = scene_cycle_rows(["one", "two"])
        rows[0]["unexpected"] = "1"
        with self.assertRaisesRegex(ValueError, "exact runtime header"):
            runtime_log.check_gate_f(rows, ["one", "two"])

        for name, value, diagnostic in (
                ("fixed_dt", ".05", "fixed_dt"),
                ("ik_enabled", "1", "IK must be disabled"),
                ("adjustment_y", ".01", "adjustment_y must be zero"),
                ("support_height", "nan", "non-finite support_height")):
            rows = scene_cycle_rows(["one", "two"])
            rows[0][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_f(rows, ["one", "two"])

    def test_failed_switch_requires_preserved_identity_generation_and_model(self):
        rows = scene_cycle_rows(["one", "two"], dwell=3, cycles=2)
        rows[4]["scene_switch_failed"] = "1"
        self.assertEqual(runtime_log.check_failed_switch(rows), {
            "switch_failures": 1,
            "preserved_scene": "two",
            "preserved_generation": 1,
        })

        for name, value, diagnostic in (
                ("scene_id", "one", "preserve scene"),
                ("scene_generation", "9", "preserve generation"),
                ("scene_frame", "0", "continue scene frame"),
                ("scene_reset_count", "9", "preserve reset count"),
                ("motion_pack_load_count", "2", "motion pack"),
                ("live_model_count", "0", "live model")):
            changed = [dict(item) for item in rows]
            changed[4][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_failed_switch(changed)

    def test_failed_switch_rejects_missing_and_first_row_pulses(self):
        with self.assertRaisesRegex(ValueError, "failure pulse"):
            runtime_log.check_failed_switch(
                scene_cycle_rows(["one"], dwell=3, cycles=1))

        rows = scene_cycle_rows(["one"], dwell=3, cycles=1)
        rows[0]["scene_switch_failed"] = "1"
        with self.assertRaisesRegex(ValueError, "preceding row"):
            runtime_log.check_failed_switch(rows)

    def test_failed_switch_validates_model_counters_on_every_row(self):
        rows = scene_cycle_rows(["one", "two"], dwell=3, cycles=2)
        rows[4]["scene_switch_failed"] = "1"
        rows[5].update({
            "model_load_count": "3",
            "model_unload_count": "1",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(ValueError, "live_model_count"):
            runtime_log.check_failed_switch(rows)

        rows = scene_cycle_rows(["one", "two"], dwell=3, cycles=2)
        rows[4]["scene_switch_failed"] = "1"
        rows[5].update({
            "model_load_count": "1",
            "model_unload_count": "0",
            "live_model_count": "1",
        })
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            runtime_log.check_failed_switch(rows)

    def test_failed_switch_rejects_end_and_short_truncated_tails(self):
        rows = scene_cycle_rows(["one"], dwell=3, cycles=1)
        rows[-1]["scene_switch_failed"] = "1"
        with self.assertRaisesRegex(ValueError, "ten subsequent"):
            runtime_log.check_failed_switch(rows)

        rows = scene_cycle_rows(["one"], dwell=5, cycles=1)
        rows[1]["scene_switch_failed"] = "1"
        with self.assertRaisesRegex(ValueError, "ten subsequent"):
            runtime_log.check_failed_switch(rows)

    def test_failed_switch_checks_all_pulses(self):
        rows = scene_cycle_rows(["one", "two"], dwell=15, cycles=1)
        rows[2]["scene_switch_failed"] = "1"
        rows[17]["scene_switch_failed"] = "1"
        report = runtime_log.check_failed_switch(rows)
        self.assertEqual(report["switch_failures"], 2)
        self.assertEqual(report["preserved_scene"], "one")
        self.assertEqual(report["preserved_generation"], 0)

        rows[20]["scene_id"] = "different"
        with self.assertRaisesRegex(ValueError, "preserve scene"):
            runtime_log.check_failed_switch(rows)

    def test_failed_switch_enforces_ten_row_preservation_window(self):
        rows = scene_cycle_rows(["one"], dwell=13, cycles=1)
        rows[1]["scene_switch_failed"] = "1"
        rows[11]["scene_id"] = "different"
        with self.assertRaisesRegex(ValueError, "ten-row.*scene"):
            runtime_log.check_failed_switch(rows)

        rows = scene_cycle_rows(["one"], dwell=13, cycles=1)
        rows[1]["scene_switch_failed"] = "1"
        rows[11]["support_height"] = "nan"
        with self.assertRaisesRegex(ValueError, "non-finite support_height"):
            runtime_log.check_failed_switch(rows)

    def test_failed_switch_allows_later_successful_generation(self):
        rows = scene_cycle_rows(["one", "two"], dwell=5, cycles=1)
        rows[1]["scene_switch_failed"] = "1"
        self.assertEqual(
            runtime_log.check_failed_switch(rows)["switch_failures"], 1)

    def test_gate_f_cli_locks_catalog_and_prints_sorted_report(self):
        self.assertEqual(runtime_log.LOCKED_SCENE_IDS, TASK11_SCENE_IDS)
        rows = scene_cycle_rows(list(TASK11_SCENE_IDS))
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            write_runtime_rows(stream, rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runtime_log.main([
                    stream.name,
                    "--gate-f",
                    "--expected-scenes", ",".join(TASK11_SCENE_IDS),
                ]), 0)
        self.assertEqual(output.getvalue().splitlines()[0],
            "VALID gate-f complete_cycles=2 frames=56 generations=28 "
            "model_loads=28 model_unloads_before_final_cleanup=27 "
            "motion_pack_loads=1")

    def test_failed_switch_cli_prints_sorted_report(self):
        rows = scene_cycle_rows(["one", "two"], dwell=3, cycles=1)
        rows[1]["scene_switch_failed"] = "1"
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            write_runtime_rows(stream, rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runtime_log.main([
                    stream.name, "--expect-switch-failure",
                ]), 0)
        self.assertEqual(output.getvalue().splitlines()[0],
            "VALID switch-failure preserved_generation=0 "
            "preserved_scene=one switch_failures=1")

    def test_gate_f_cli_rejects_invalid_catalog_and_flag_combinations(self):
        catalog = list(TASK11_SCENE_IDS)
        duplicate = catalog[:-1] + [catalog[0]]
        wrong = catalog[:-1] + ["wrong"]
        reordered = catalog[:]
        reordered[0], reordered[1] = reordered[1], reordered[0]
        cases = (
            (["--gate-f"], "expected-scenes"),
            (["--expected-scenes", ",".join(catalog)], "requires --gate-f"),
            (["--gate-f", "--expected-scenes", ""], "locked 14-scene"),
            (["--gate-f", "--expected-scenes", ",".join(duplicate)],
             "locked 14-scene"),
            (["--gate-f", "--expected-scenes", ",".join(wrong)],
             "locked 14-scene"),
            (["--gate-f", "--expected-scenes", ",".join(reordered)],
             "locked 14-scene"),
            (["--expect-switch-failure", "--gate-a"], "may not combine"),
            (["--expect-switch-failure", "--gate-c"], "may not combine"),
            (["--expect-switch-failure", "--gate-d"], "may not combine"),
            (["--expect-switch-failure", "--gate-f", "--expected-scenes",
              ",".join(catalog)], "may not combine"),
        )
        for extra, diagnostic in cases:
            stderr = io.StringIO()
            with self.subTest(extra=extra):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        runtime_log.main(["unused.csv", *extra])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(diagnostic, stderr.getvalue())

    def test_ab_requires_identical_script(self):
        treatment = gate_c_rows()
        control = gate_c_rows()
        for treatment_row, control_row in zip(treatment, control):
            treatment_row["selected_terrain_error"] = ".1"
            treatment_row["effective_terrain_weight"] = "4"
            control_row["selected_terrain_error"] = ".4"
            control_row["effective_terrain_weight"] = "0"
        treatment[20]["route_waypoint"] = "9"
        with self.assertRaisesRegex(ValueError, "scripted input"):
            compare_control(treatment, control)

    def test_runtime_suffix_validation_is_exact_and_reset_aware(self):
        rows = [runtime_row(frame) for frame in range(24)]
        for frame in range(8, 24):
            generation = (frame - 8) // 8 + 1
            scene_frame = (frame - 8) % 8
            current = 200 + scene_frame
            query = current if scene_frame == 0 else current - 1
            rows[frame].update({
                "scene_generation": str(generation),
                "scene_frame": str(scene_frame),
                "scene_reset_count": str(generation + 1),
                "database_frame": str(current),
                "query_database_frame": str(query),
                "selected_database_frame": str(query),
                "left_stance_slip_reset": str(int(scene_frame == 0)),
                "right_stance_slip_reset": str(int(scene_frame == 0)),
            })
        self.assertEqual(check_rows(rows)["frames"], 24)

        cases = (
            ("fixed_dt", 4, ".05", "fixed_dt"),
            ("support_height", 4, "nan", "non-finite support_height"),
            ("left_contact", 4, "2", "left_contact must be 0 or 1"),
            ("source_name", 4, "", "empty source_name"),
            ("ik_enabled", 4, "1", "IK must be disabled"),
            ("adjustment_y", 4, ".01", "adjustment_y must be zero"),
            ("scene_frame", 4, "9", "scene_frame"),
            ("scene_generation", 8, "3", "scene_generation"),
            ("scene_frame", 8, "1", "scene_frame"),
            ("scene_reset_count", 8, "9", "scene_reset_count"),
            ("motion_pack_load_count", 4, "2", "motion pack"),
            ("live_model_count", 4, "0", "live model"),
        )
        for name, index, value, diagnostic in cases:
            changed = [dict(item) for item in rows]
            changed[index][name] = value
            with self.subTest(name=name, index=index):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    check_rows(changed)

    def test_runtime_gate_requires_exact_header(self):
        rows = gate_c_rows()
        rows[0].pop("source_name")
        with self.assertRaisesRegex(ValueError, "exact runtime header"):
            runtime_log.check_gate_c(rows)
        rows = gate_c_rows()
        rows[0]["unexpected"] = "1"
        with self.assertRaisesRegex(ValueError, "exact runtime header"):
            runtime_log.check_gate_c(rows)

    def test_gate_c_rejects_contract_activation_and_source_failures(self):
        cases = []
        rows = gate_c_rows()
        rows[0]["mode"] = "terrain"
        cases.append((rows, "route mode"))
        rows = gate_c_rows()
        rows[50]["scene_id"] = "other"
        cases.append((rows, "one scene and route"))
        rows = gate_c_rows()
        rows[50]["matching_enabled"] = "0"
        cases.append((rows, "matching"))
        rows = gate_c_rows()
        rows[50]["support_retargeting_enabled"] = "0"
        cases.append((rows, "support retargeting"))
        rows = gate_c_rows()
        for item in rows[:30]:
            set_terrain(item, 0, 0)
        cases.append((rows, "terrain activation"))
        rows = gate_c_rows()
        for item in rows:
            item["source_index"] = "0"
            item["source_terrain"] = "flat"
        cases.append((rows, "terrain source"))
        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_c(rows)

    def test_gate_c_rejects_landing_alignment_and_return_failures(self):
        cases = []
        rows = gate_c_rows()
        rows[80]["runtime_support_root_height"] = ".30"
        cases.append((rows, "landing block"))
        rows = gate_c_rows()
        for item in rows[113:]:
            item["runtime_support_root_height"] = ".10"
        cases.append((rows, "return to baseline"))
        rows = gate_c_rows()
        rows[80]["rendered_hips_y"] = "1.17"
        cases.append((rows, "Hips stage agreement"))
        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_c(rows)

    def test_mixed_checker_rejects_span_order_and_nonfinite_geometry(self):
        cases = []
        rows = mixed_multilevel_rows()
        for item in rows:
            if abs(float(item["simulation_z"]) - 6.20) <= 0.05:
                item["simulation_x"] = ".10"
        cases.append((rows, "endpoint crossing"))
        rows = mixed_multilevel_rows()
        for item in rows:
            z = float(item["simulation_z"])
            if 6.85 <= z <= 7.35:
                item["runtime_support_root_height"] = ".40"
            elif 6.25 <= z <= 6.75:
                item["runtime_support_root_height"] = ".28"
        cases.append((rows, "block plateaus"))
        rows = mixed_multilevel_rows()
        rows[210]["simulation_z"] = "nan"
        cases.append((rows, "non-finite"))
        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_mixed_multilevel(rows)

    def test_gate_d_rejects_missing_stop_clearance_and_support_rise(self):
        rows = gate_d_rows()
        for item in rows[31:]:
            item["blocked"] = "0"
            item["blocked_reason"] = "clear"
        self.assertGreaterEqual(
            runtime_log.check_gate_d(rows)["stopped_frames"], 25)

        cases = []
        rows = gate_d_rows()
        for item in rows:
            item["blocked"] = "0"
            item["blocked_reason"] = "clear"
        cases.append((rows, "never reported blocked"))
        rows = gate_d_rows()
        for item in rows:
            item["applied_speed"] = ".5"
        cases.append((rows, "never stopped"))
        rows = gate_d_rows()
        rows[50]["blocked_distance"] = ".019"
        cases.append((rows, "clearance"))
        rows = gate_d_rows()
        for item in rows[30:76]:
            item["applied_speed"] = ".1"
        cases.append((rows, "25 consecutive"))
        rows = gate_d_rows()
        rows[50]["source_root_height"] = ".03"
        rows[50]["blocked"] = "0"
        rows[50]["blocked_reason"] = "clear"
        cases.append((rows, "blocked support rise"))
        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_d(rows)

    def test_ab_requires_every_exact_runtime_script_field(self):
        fields = (
            "scene_id", "mode", "route", "scene_generation", "scene_frame",
            "route_waypoint", "route_complete", "commanded_speed",
        )
        for field in fields:
            treatment = gate_c_rows()
            control = gate_c_rows()
            for treatment_row, control_row in zip(treatment, control):
                treatment_row["selected_terrain_error"] = ".1"
                control_row["selected_terrain_error"] = ".4"
                control_row["effective_terrain_weight"] = "0"
            if field in ("scene_generation", "scene_frame"):
                for index, treatment_row in enumerate(treatment[40:], 40):
                    treatment_row["scene_generation"] = "1"
                    treatment_row["scene_frame"] = str(index - 40)
                    treatment_row["scene_reset_count"] = "2"
                    treatment_row["left_stance_slip_reset"] = str(
                        int(index == 40))
                    treatment_row["right_stance_slip_reset"] = str(
                        int(index == 40))
            elif field in ("route_waypoint", "route_complete"):
                treatment[40][field] = "9" if field == "route_waypoint" else "1"
            else:
                treatment[40][field] = (
                    ".50" if field == "commanded_speed" else "different")
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "scripted input"):
                    compare_control(treatment, control)

    def test_rejects_selected_frame_inconsistent_with_transition(self):
        with self.assertRaisesRegex(ValueError, "selected frame"):
            check_rows([row(
                0, 20, searched=1, transitioned=1,
                query_database_frame=20, selected_database_frame=20,
                incumbent_cost=2.0, selected_cost=1.0,
            )])

    def test_rejects_query_frame_or_range_not_from_prior_pose(self):
        with self.assertRaisesRegex(ValueError, "query frame"):
            check_rows([
                row(0, 10),
                row(
                    1, 21, query_database_frame=20,
                    selected_database_frame=20,
                ),
            ])
        with self.assertRaisesRegex(ValueError, "query range"):
            check_rows([
                row(0, 10, range=0, source_range=0, query_range=0),
                row(
                    1, 11, query_database_frame=10,
                    selected_database_frame=10,
                    query_range=1, source_range=1, range=1,
                ),
            ])

    def test_rejects_negative_database_frame_fields(self):
        cases = (
            ("query_database_frame", row(
                0, 21, searched=1, transitioned=1,
                query_database_frame=-1, selected_database_frame=20,
                incumbent_cost=2.0, selected_cost=1.0,
            )),
            ("selected_database_frame", row(
                0, 0, searched=1, transitioned=1,
                query_database_frame=10, selected_database_frame=-1,
                incumbent_cost=2.0, selected_cost=1.0,
            )),
            ("database_frame", row(
                0, -1,
                query_database_frame=0, selected_database_frame=0,
            )),
        )
        for name, values in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                        ValueError, f"{name} must be nonnegative"):
                    check_rows([values])

    def test_rejects_integer_outside_int32_range(self):
        too_large = 2 ** 31
        with self.assertRaisesRegex(ValueError, "outside signed int32"):
            check_rows([row(0, too_large)])

    def test_rejects_negative_range_fields(self):
        cases = (
            ("query_range", row(
                0, 21, searched=1, transitioned=1,
                query_database_frame=10, selected_database_frame=20,
                query_range=-1, source_range=0, range=0,
                incumbent_cost=2.0, selected_cost=1.0,
            )),
            ("range", row(
                0, 21, searched=1, transitioned=1,
                query_database_frame=10, selected_database_frame=20,
                query_range=0, source_range=-1, range=-1,
                incumbent_cost=2.0, selected_cost=1.0,
            )),
            ("source_range", row(
                0, 21, searched=1, transitioned=1,
                query_database_frame=10, selected_database_frame=20,
                query_range=0, source_range=-1, range=0,
                incumbent_cost=2.0, selected_cost=1.0,
            )),
        )
        for name, values in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                        ValueError, f"{name} must be nonnegative"):
                    check_rows([values])

    def test_rejects_nonpositive_fixed_dt(self):
        for fixed_dt in (0.0, -0.04):
            with self.subTest(fixed_dt=fixed_dt):
                with self.assertRaisesRegex(ValueError, "fixed_dt must be positive"):
                    check_rows([row(0, 10, fixed_dt=fixed_dt)])

    def test_rejects_terrain_weight_outside_parser_domain(self):
        for weight in (-0.1, 10.1):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                        ValueError, "effective_terrain_weight must be in"):
                    check_rows([row(
                        0, 10, effective_terrain_weight=weight,
                    )])

    def test_rejects_negative_xz_displacement_lengths(self):
        for name in ("adjustment_xz", "clamp_xz"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                        ValueError, f"{name} must be nonnegative"):
                    check_rows([row(0, 10, **{name: -0.1})])

    def test_accepts_signed_terrain_height_clearance_and_y_fields(self):
        summary = check_rows([row(
            0, 10,
            terrain0=-0.1,
            terrain_point0_y=-0.2,
            raw_selected_hips_y=-0.3,
            inertialized_hips_y=-0.3,
            rendered_hips_y=-0.3,
            runtime_root_surface_height=-0.4,
            runtime_left_toe_surface_height=-0.4,
            runtime_right_toe_surface_height=-0.4,
            raw_selected_min_clearance=-0.1,
            inertialized_min_clearance=-0.1,
            rendered_min_clearance=-0.1,
            adjustment_y=-0.2,
            clamp_y=-0.2,
        )])
        self.assertEqual(summary["frames"], 1)

    def test_rejects_transition_marked_repeated_substride(self):
        database_frames = list(range(10, 18)) * 3
        rows = []
        for index, current in enumerate(database_frames):
            if index == 0:
                query = current
                selected = current
                changes = {}
            elif index in (8, 16):
                query = database_frames[index - 1]
                selected = current - 1
                changes = {
                    "searched": 1,
                    "transitioned": 1,
                    "incumbent_cost": 2.0,
                    "selected_cost": 1.0,
                }
            else:
                query = database_frames[index - 1]
                selected = query
                changes = {}
            rows.append(row(
                index, current,
                query_database_frame=query,
                selected_database_frame=selected,
                **changes,
            ))
        with self.assertRaisesRegex(ValueError, "repeated sub-stride period 8"):
            check_rows(rows)
        with self.assertRaisesRegex(ValueError, "selected frame"):
            check_rows([row(
                0, 20, searched=1, transitioned=0,
                query_database_frame=10, selected_database_frame=20,
            )])

    def test_accepts_selected_frame_and_source_range_for_transition(self):
        summary = check_rows([row(
            0, 21, searched=1, transitioned=1,
            query_database_frame=10, query_range=0,
            selected_database_frame=20, source_range=1, range=1,
            incumbent_cost=2.0, selected_cost=1.0,
        )])
        self.assertEqual(summary["transitions"], 1)

    def test_rejects_range_change_without_transition(self):
        rows = [
            row(0, 10, range=0, source_range=0),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                query_range=0, source_range=1, range=1,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "range change"):
            check_rows(rows)

    def test_rejects_pose_range_different_from_selected_source_range(self):
        with self.assertRaisesRegex(ValueError, "source range"):
            check_rows([row(0, 10, range=1, source_range=0)])

    def test_rejects_nonsequential_advance_without_transition(self):
        rows = [
            row(0, 10),
            row(
                1, 10, query_database_frame=10,
                selected_database_frame=10,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "nonsequential"):
            check_rows(rows)

    def test_rejects_transition_that_does_not_beat_incumbent(self):
        rows = [row(
            0, 20, transitioned=1, searched=1,
            query_database_frame=10, selected_database_frame=20,
            incumbent_cost=1.0, selected_cost=1.5,
        )]
        with self.assertRaisesRegex(ValueError, "beat incumbent"):
            check_rows(rows)

    def test_accepts_transition_cost_within_four_ulps_of_incumbent(self):
        summary = check_rows([row(
            0, 20, transitioned=1, searched=1,
            query_database_frame=10, selected_database_frame=20,
            incumbent_cost=1.0, selected_cost=float32_offset(1.0, 3),
        )])
        self.assertEqual(summary["transitions"], 1)

    def test_rejects_unsearched_no_transition_selected_cost_drift(self):
        with self.assertRaisesRegex(ValueError, "no-transition selected cost"):
            check_rows([row(
                0, 10, searched=0, transitioned=0,
                incumbent_cost=1.0, selected_cost=0.5,
            )])

    def test_rejects_searched_no_transition_selected_cost_drift(self):
        with self.assertRaisesRegex(ValueError, "no-transition selected cost"):
            check_rows([row(
                0, 10, searched=1, transitioned=0,
                incumbent_cost=1.0, selected_cost=0.5,
            )])

    def test_accepts_searched_no_transition_cost_within_four_ulps(self):
        summary = check_rows([row(
            0, 10, searched=1, transitioned=0,
            incumbent_cost=1.0, selected_cost=float32_offset(1.0, 3),
        )])
        self.assertEqual(summary["frames"], 1)

    def test_rejects_searched_no_transition_cost_over_four_ulps(self):
        with self.assertRaisesRegex(ValueError, "5 float32 ULPs"):
            check_rows([row(
                0, 10, searched=1, transitioned=0,
                incumbent_cost=1.0, selected_cost=float32_offset(1.0, 5),
            )])

    def test_rejects_negative_incumbent_cost(self):
        with self.assertRaisesRegex(ValueError, "negative incumbent_cost"):
            check_rows([row(
                0, 10, incumbent_cost=-1.0, selected_cost=-1.0,
            )])

    def test_rejects_negative_selected_cost(self):
        with self.assertRaisesRegex(ValueError, "negative selected_cost"):
            check_rows([row(
                0, 20, searched=1, transitioned=1,
                query_database_frame=10, selected_database_frame=20,
                incumbent_cost=1.0, selected_cost=-0.5,
            )])

    def test_rejects_negative_selected_terrain_error(self):
        with self.assertRaisesRegex(
                ValueError, "negative selected_terrain_error"):
            check_rows([row(0, 10, selected_terrain_error=-0.5)])

    def test_rejects_cost_outside_float32_range(self):
        with self.assertRaisesRegex(
                ValueError, r"^row 0: cost is not float32$"):
            check_rows([row(
                0, 10, searched=1,
                incumbent_cost=1e100, selected_cost=1e100,
            )])

    def test_rejects_unsearched_cost_outside_float32_range(self):
        with self.assertRaisesRegex(
                ValueError, r"^row 0: cost is not float32$"):
            check_rows([row(
                0, 10, searched=0,
                incumbent_cost=1e100, selected_cost=1e100,
            )])

    def test_rejects_terrain_error_outside_float32_range(self):
        with self.assertRaisesRegex(
                ValueError, r"^row 0: cost is not float32$"):
            check_rows([row(0, 10, selected_terrain_error=1e100)])

    def test_rejects_non_cost_float_outside_float32_range(self):
        with self.assertRaisesRegex(
                ValueError,
                "raw_selected_min_clearance is not float32"):
            check_rows([row(
                0, 10, raw_selected_min_clearance=1e100,
            )])

    def test_rejects_incomplete_query_bit_snapshot(self):
        with self.assertRaisesRegex(ValueError, "39 float bit patterns"):
            check_rows([row(0, 10, query_bits_hex="0" * 311)])

    def test_rejects_nonfinite_query_bit_snapshot(self):
        with self.assertRaisesRegex(ValueError, "non-finite query"):
            check_rows([row(
                0, 10,
                query_bits_hex="7f800000" + "00000000" * 38,
            )])

    def test_rejects_query_terrain_bit_disagreement(self):
        with self.assertRaisesRegex(ValueError, "terrain query bits"):
            check_rows([row(
                0, 10, terrain0=0.125,
                query_bits_hex="00000000" * 39,
            )])

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

    def test_gate_a_classifies_the_earliest_penetration_stage(self):
        rows = [
            row(
                0, 10, terrain0=0.1,
                inertialized_min_clearance=-0.01,
                rendered_min_clearance=-0.02,
            ),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                raw_selected_min_clearance=-0.03,
            ),
        ]
        report = diagnose_gate_a(rows)
        self.assertEqual(report["first_penetration_frame"], 0)
        self.assertEqual(report["penetration_class"], "blended-rendered")

    def test_gate_a_raw_stage_wins_on_the_same_earliest_frame(self):
        report = diagnose_gate_a([row(
            0, 10, terrain0=0.1,
            raw_selected_min_clearance=-0.01,
            inertialized_min_clearance=-0.02,
        )])
        self.assertEqual(report["penetration_class"], "raw-selected")

    def test_terrain_treatment_must_improve_raw_terrain_error(self):
        treatment = [row(0, 10, terrain0=0.1, selected_terrain_error=0.5)]
        control = [row(
            0, 10, terrain0=0.1, effective_terrain_weight=0,
            selected_terrain_error=2.0,
        )]
        self.assertEqual(compare_control(treatment, control), (0.5, 2.0))

    def test_compare_control_requires_active_terrain_in_both_runs(self):
        treatment = [row(
            0, 10, terrain0=0.1, selected_terrain_error=0.5,
        )]
        control = [row(
            0, 10, effective_terrain_weight=0,
            selected_terrain_error=2.0,
        )]
        with self.assertRaisesRegex(ValueError, "control terrain query"):
            compare_control(treatment, control)

    def test_compare_control_accepts_independent_active_positions(self):
        treatment = [
            row(0, 10, terrain0=-0.1, selected_terrain_error=0.5),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                selected_terrain_error=9.0,
            ),
        ]
        control = [
            row(
                0, 10, effective_terrain_weight=0,
                selected_terrain_error=0.25,
            ),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10, terrain1=0.1,
                effective_terrain_weight=0,
                selected_terrain_error=2.0,
            ),
        ]
        self.assertEqual(compare_control(treatment, control), (0.5, 2.0))

    def test_compare_control_validates_both_logs(self):
        treatment = [
            row(0, 10, terrain0=0.1, selected_terrain_error=0.1),
            row(
                1, 11, query_database_frame=10,
                selected_database_frame=10,
                terrain0=0.1, selected_terrain_error=0.1,
            ),
        ]
        control = [
            row(
                0, 10, terrain0=0.1, effective_terrain_weight=0,
                selected_terrain_error=0.4,
            ),
            row(
                1, 10, query_database_frame=10,
                selected_database_frame=10,
                terrain0=0.1, effective_terrain_weight=0,
                selected_terrain_error=0.4,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "nonsequential"):
            compare_control(treatment, control)

    def test_compare_control_requires_exact_weights(self):
        treatment = [row(
            0, 10, terrain0=0.1, effective_terrain_weight=3,
            selected_terrain_error=0.1,
        )]
        control = [row(
            0, 10, terrain0=0.1, effective_terrain_weight=0,
            selected_terrain_error=0.4,
        )]
        with self.assertRaisesRegex(ValueError, "treatment weight"):
            compare_control(treatment, control)
        treatment[0]["effective_terrain_weight"] = "4"
        control[0]["effective_terrain_weight"] = "1"
        with self.assertRaisesRegex(ValueError, "control weight"):
            compare_control(treatment, control)

    def test_compare_control_requires_identical_script_metadata(self):
        for name, value in (
                ("fixed_dt", 0.05),
                ("scene_id", "different-scene"),
                ("mode", "flat"),
                ("route", "different-route")):
            treatment = [row(
                0, 10, terrain0=0.1, selected_terrain_error=0.1,
            )]
            control = [row(
                0, 10, terrain0=0.1, effective_terrain_weight=0,
                selected_terrain_error=0.4, **{name: value},
            )]
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "script metadata"):
                    compare_control(treatment, control)

    def test_gate_a_contract_accepts_only_the_frozen_fixture(self):
        rows = sequential_rows(375)
        self.assertEqual(
            runtime_log.check_gate_a_contract(rows)["frames"], 375)

    def test_gate_a_contract_rejects_fixture_drift(self):
        fixed_dt_bits = int.from_bytes(struct.pack(">f", 0.04), "big")
        next_fixed_dt = struct.unpack(
            ">f", (fixed_dt_bits + 1).to_bytes(4, "big"))[0]
        cases = (
            ("rows", None, None, "exactly 375"),
            ("fixed_dt", 0, next_fixed_dt, "fixed_dt"),
            ("scene_id", 0, "different-scene", "scene"),
            ("mode", 0, "flat", "mode"),
            ("route", 0, "different-route", "route"),
            ("effective_terrain_weight", 0, 0, "weight"),
            ("matching_enabled", 0, 0, "matching"),
            ("adjustment_enabled", 0, 0, "adjustment"),
            ("clamping_enabled", 0, 0, "clamping"),
            ("support_retargeting_enabled", 0, 1, "support retargeting"),
            ("ik_enabled", 0, 1, "IK"),
        )
        for name, index, value, diagnostic in cases:
            rows = sequential_rows(375)
            if name == "rows":
                rows.pop()
            else:
                rows[index][name] = str(value)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_a_contract(rows)

    def test_read_rows_rejects_duplicate_csv_columns(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            stream.write("frame,frame\n0,0\n")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "duplicate CSV column"):
                read_rows(stream.name)

    def test_read_rows_rejects_data_wider_than_header(self):
        values = row(0, 10)
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as stream:
            stream.write(",".join(CSV_COLUMNS) + "\n")
            stream.write(
                ",".join(values[name] for name in CSV_COLUMNS) +
                ",unexpected\n")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "wider than header"):
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
