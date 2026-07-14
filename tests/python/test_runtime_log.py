import tempfile
import struct
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


RUNTIME_SUFFIX = (
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
        query_values = [0.0] * 27 + [
            float(values[f"terrain{sample}"]) for sample in range(4)]
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
    if "query_bits_hex" not in changes:
        query_values = [0.0] * 27 + [
            float(values[f"terrain{sample}"]) for sample in range(4)]
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


def set_terrain(values, sample, value):
    values[f"terrain{sample}"] = str(value)
    query_values = [0.0] * 27 + [
        float(values[f"terrain{query_sample}"])
        for query_sample in range(4)
    ]
    values["query_bits_hex"] = "".join(
        struct.pack(">f", query_value).hex() for query_value in query_values)


class RuntimeLogTests(unittest.TestCase):
    def test_runtime_columns_append_after_gate_a(self):
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[-len(RUNTIME_SUFFIX):]), RUNTIME_SUFFIX)
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[:-len(RUNTIME_SUFFIX)]), GATE_A_COLUMNS)

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
        with self.assertRaisesRegex(ValueError, "31 float bit patterns"):
            check_rows([row(0, 10, query_bits_hex="0" * 247)])

    def test_rejects_nonfinite_query_bit_snapshot(self):
        with self.assertRaisesRegex(ValueError, "non-finite query"):
            check_rows([row(
                0, 10,
                query_bits_hex="7f800000" + "00000000" * 30,
            )])

    def test_rejects_query_terrain_bit_disagreement(self):
        with self.assertRaisesRegex(ValueError, "terrain query bits"):
            check_rows([row(
                0, 10, terrain0=0.125,
                query_bits_hex="00000000" * 31,
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
