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
