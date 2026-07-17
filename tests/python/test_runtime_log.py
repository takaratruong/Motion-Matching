import contextlib
import hashlib
import io
import math
import os
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


LEGACY_RUNTIME_SUFFIX = (
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


def _columns(text):
    return tuple(name.strip() for name in text.split(",") if name.strip())


IK_SUFFIX = _columns("""
ik_applied,ik_safe_stop_requested,ik_stop_reason,max_ik_correction,
actual_simulation_speed,ik_candidate_rejected,ik_candidate_clearance_status,
left_candidate_toe_clearance,left_candidate_foot_clearance,
right_candidate_toe_clearance,right_candidate_foot_clearance,
ik_candidate_minimum_clearance,
left_recorded_contact,right_recorded_contact,left_locked,right_locked,
left_observed_lock_drift,right_observed_lock_drift,
left_lock_drift,right_lock_drift,
left_sole_normal_alignment,right_sole_normal_alignment,
left_contact_residual,right_contact_residual,
left_target_height,right_target_height,
left_target_normal_x,left_target_normal_y,left_target_normal_z,
right_target_normal_x,right_target_normal_y,right_target_normal_z,
left_swing_candidates_evaluated,left_swing_selected_index,
left_swing_selected_lift_bits,left_swing_materialized_command_y_bits,
left_swing_actual_sphere_center_bits_hex,
left_swing_selected_clearance_status,
left_swing_selected_controller_constraints_passed,
left_swing_selected_clearance_certified,
left_swing_lower_margin,left_swing_witness_upper_margin,
left_swing_selected_work_point_queries,left_swing_selected_work_cells_visited,
left_swing_selected_work_primitive_triangle_pairs,
left_swing_selected_work_face_patches,left_swing_selected_work_candidate_tests,
left_swing_selected_work_subdivision_nodes,
left_swing_total_work_point_queries,left_swing_total_work_cells_visited,
left_swing_total_work_primitive_triangle_pairs,
left_swing_total_work_face_patches,left_swing_total_work_candidate_tests,
left_swing_total_work_subdivision_nodes,
right_swing_candidates_evaluated,right_swing_selected_index,
right_swing_selected_lift_bits,right_swing_materialized_command_y_bits,
right_swing_actual_sphere_center_bits_hex,
right_swing_selected_clearance_status,
right_swing_selected_controller_constraints_passed,
right_swing_selected_clearance_certified,
right_swing_lower_margin,right_swing_witness_upper_margin,
right_swing_selected_work_point_queries,right_swing_selected_work_cells_visited,
right_swing_selected_work_primitive_triangle_pairs,
right_swing_selected_work_face_patches,right_swing_selected_work_candidate_tests,
right_swing_selected_work_subdivision_nodes,
right_swing_total_work_point_queries,right_swing_total_work_cells_visited,
right_swing_total_work_primitive_triangle_pairs,
right_swing_total_work_face_patches,right_swing_total_work_candidate_tests,
right_swing_total_work_subdivision_nodes,
left_reachable,right_reachable,left_knee_clearance,left_ankle_clearance,
left_toe_clearance,left_foot_clearance,left_shin_clearance,
left_thigh_clearance,right_knee_clearance,right_ankle_clearance,
right_toe_clearance,right_foot_clearance,right_shin_clearance,
right_thigh_clearance,ik_hips_clearance,ik_minimum_clearance
""")

DIRECTIONAL_SUFFIX = _columns("""
requested_velocity_x,requested_velocity_y,requested_velocity_z,
applied_velocity_x,applied_velocity_y,applied_velocity_z,
desired_heading_bits_hex,predicted_heading_bits_hex,
simulation_heading_error_deg,rendered_heading_error_deg,
footprint_status,footprint_blocked,footprint_blocked_reason,
footprint_root_height,
left_footprint_min_height,left_footprint_max_height,
right_footprint_min_height,right_footprint_max_height,
left_maximum_root_split,right_maximum_root_split,
left_footprint_multilevel,right_footprint_multilevel,
left_landing_expected,left_landing_patch_ready,left_landing_sample,
left_landing_surface_status,left_landing_walkability_class,
left_predicted_landing_center_x,left_predicted_landing_center_y,
left_predicted_landing_center_z,left_predicted_landing_height,
left_predicted_landing_normal_x,left_predicted_landing_normal_y,
left_predicted_landing_normal_z,left_landing_patch_maximum_residual,
right_landing_expected,right_landing_patch_ready,right_landing_sample,
right_landing_surface_status,right_landing_walkability_class,
right_predicted_landing_center_x,right_predicted_landing_center_y,
right_predicted_landing_center_z,right_predicted_landing_height,
right_predicted_landing_normal_x,right_predicted_landing_normal_y,
right_predicted_landing_normal_z,right_landing_patch_maximum_residual,
footprint_sweeps,footprint_surface_queries,footprint_node_visits,
frame_rejected,frame_rejection_stage,ik_safe_stop_latched,
rejected_attempted_footprint_available,rejected_attempted_ik_available,
rejected_stop_reason,rejected_attempted_pose_available,rejected_pose_status,
rejected_pose_minimum_clearance,
rejected_left_landing_expected,rejected_left_landing_patch_ready,
rejected_left_landing_sample,rejected_left_landing_center_x,
rejected_left_landing_center_y,rejected_left_landing_center_z,
rejected_left_landing_surface_status,rejected_left_landing_surface_height,
rejected_left_landing_surface_normal_x,
rejected_left_landing_surface_normal_y,
rejected_left_landing_surface_normal_z,
rejected_left_landing_walkability_class,
rejected_left_landing_patch_maximum_residual,
rejected_left_target_x,rejected_left_target_y,rejected_left_target_z,
rejected_left_target_normal_x,rejected_left_target_normal_y,
rejected_left_target_normal_z,rejected_left_reachable,
rejected_left_correction_limited,rejected_left_selected_clearance_status,
rejected_left_selected_lower_margin,rejected_left_selected_witness_upper,
rejected_right_landing_expected,rejected_right_landing_patch_ready,
rejected_right_landing_sample,rejected_right_landing_center_x,
rejected_right_landing_center_y,rejected_right_landing_center_z,
rejected_right_landing_surface_status,rejected_right_landing_surface_height,
rejected_right_landing_surface_normal_x,
rejected_right_landing_surface_normal_y,
rejected_right_landing_surface_normal_z,
rejected_right_landing_walkability_class,
rejected_right_landing_patch_maximum_residual,
rejected_right_target_x,rejected_right_target_y,rejected_right_target_z,
rejected_right_target_normal_x,rejected_right_target_normal_y,
rejected_right_target_normal_z,rejected_right_reachable,
rejected_right_correction_limited,rejected_right_selected_clearance_status,
rejected_right_selected_lower_margin,rejected_right_selected_witness_upper,
accepted_state_digest_hex
""")

RUNTIME_SUFFIX = LEGACY_RUNTIME_SUFFIX + IK_SUFFIX + DIRECTIONAL_SUFFIX

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
    for name in IK_SUFFIX + DIRECTIONAL_SUFFIX:
        values.setdefault(name, "0")
    values.update({
        "ik_stop_reason": "none",
        "ik_candidate_clearance_status": "ok",
        "left_swing_actual_sphere_center_bits_hex": "00000000" * 12,
        "left_swing_selected_clearance_status": "invalid-input",
        "right_swing_actual_sphere_center_bits_hex": "00000000" * 12,
        "right_swing_selected_clearance_status": "invalid-input",
        "left_swing_selected_index": str(2 ** 32 - 1),
        "right_swing_selected_index": str(2 ** 32 - 1),
        "left_sole_normal_alignment": "1",
        "right_sole_normal_alignment": "1",
        "left_target_normal_y": "1",
        "right_target_normal_y": "1",
        "left_candidate_toe_clearance": ".03",
        "left_candidate_foot_clearance": ".03",
        "right_candidate_toe_clearance": ".03",
        "right_candidate_foot_clearance": ".03",
        "ik_candidate_minimum_clearance": ".03",
        "left_knee_clearance": ".03",
        "left_ankle_clearance": ".03",
        "left_toe_clearance": ".03",
        "left_foot_clearance": ".03",
        "left_shin_clearance": ".03",
        "left_thigh_clearance": ".03",
        "right_knee_clearance": ".03",
        "right_ankle_clearance": ".03",
        "right_toe_clearance": ".03",
        "right_foot_clearance": ".03",
        "right_shin_clearance": ".03",
        "right_thigh_clearance": ".03",
        "ik_hips_clearance": ".03",
        "ik_minimum_clearance": ".03",
        "requested_velocity_z": ".5",
        "applied_velocity_z": ".5",
        "desired_heading_bits_hex": "3f800000000000000000000000000000",
        "predicted_heading_bits_hex":
            "3f800000000000000000000000000000" * 4,
        "footprint_status": "ok",
        "footprint_blocked_reason": "clear",
        "left_landing_sample": str(2 ** 32 - 1),
        "left_landing_surface_status": "invalid",
        "left_predicted_landing_normal_y": "0",
        "right_landing_sample": str(2 ** 32 - 1),
        "right_landing_surface_status": "invalid",
        "right_predicted_landing_normal_y": "0",
        "frame_rejection_stage": "none",
        "rejected_stop_reason": "none",
        "rejected_pose_status": "invalid-input",
        "rejected_left_landing_sample": str(2 ** 32 - 1),
        "rejected_left_landing_surface_status": "invalid",
        "rejected_left_landing_surface_normal_y": "0",
        "rejected_left_target_normal_y": "0",
        "rejected_left_selected_clearance_status": "invalid-input",
        "rejected_right_landing_sample": str(2 ** 32 - 1),
        "rejected_right_landing_surface_status": "invalid",
        "rejected_right_landing_surface_normal_y": "0",
        "rejected_right_target_normal_y": "0",
        "rejected_right_selected_clearance_status": "invalid-input",
        "accepted_state_digest_hex": "0000000000000000",
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


def legacy_runtime_rows(rows):
    columns = tuple(GATE_A_COLUMNS) + LEGACY_RUNTIME_SUFFIX
    return [{name: item[name] for name in columns} for item in rows]


def write_columns(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        stream.write(",".join(columns) + "\n")
        for item in rows:
            stream.write(",".join(item[name] for name in columns) + "\n")


def make_oracle_tree(directory, basename, rows):
    legacy_columns = tuple(GATE_A_COLUMNS) + LEGACY_RUNTIME_SUFFIX
    names = ("controller",) + tuple(runtime_log.FLAT_ORACLE_CASES) + tuple(
        runtime_log.FORWARD_ORACLE_CASES)
    for name in names:
        path = os.path.join(directory, name)
        if name == "controller":
            with open(path, "wb") as stream:
                stream.write(b"test-controller")
        elif name == basename:
            write_columns(path, rows, legacy_columns)
        else:
            with open(path, "wb") as stream:
                stream.write(name.encode("ascii"))
    record = os.path.join(directory, "SHA256SUMS")
    with open(record, "w", encoding="ascii", newline="") as stream:
        for name in names:
            with open(os.path.join(directory, name), "rb") as payload:
                digest = hashlib.file_digest(payload, "sha256").hexdigest()
            stream.write(f"{digest}  {name}\n")
    return os.path.join(directory, basename)


def set_terrain(values, sample, value):
    values[f"terrain{sample}"] = str(value)
    query_values = [0.0] * 27 + [
        float(values[f"terrain{query_sample}"])
        for query_sample in range(4)
    ]
    values["query_bits_hex"] = "".join(
        struct.pack(">f", query_value).hex() for query_value in query_values)


HEADING_COMPONENTS = {
    "forward": (1.0, 0.0, 0.0, 0.0),
    "backward": (0.0, 0.0, 1.0, 0.0),
    "positive-x": (0.707106769, 0.0, 0.707106769, 0.0),
    "negative-x": (0.707106769, 0.0, -0.707106769, 0.0),
    "diagonal-positive-x": (0.923879504, 0.0, 0.382683426, 0.0),
    "diagonal-negative-x": (0.923879504, 0.0, -0.382683426, 0.0),
}


def heading_bits(name):
    return "".join(struct.pack(">f", value).hex()
                   for value in HEADING_COMPONENTS[name])


def gate_l_rows(
        heading="positive-x", count=120, ik=1,
        scene="stairs-standard", route="ascent-landing-descent",
        end_x=1.2, end_z=2.4, multilevel=True):
    rows = []
    bits = heading_bits(heading)
    for frame in range(count):
        phase = frame / (count - 1)
        hips_y = .8 + .02 * math.sin(frame * .1)
        rows.append(runtime_row(
            frame,
            scene_id=scene,
            route=route,
            ik_enabled=ik,
            ik_applied=ik,
            actual_simulation_speed=.5,
            simulation_x=end_x * phase,
            simulation_z=end_z * phase,
            route_complete=int(frame == count - 1),
            route_target_height=.36,
            desired_heading_bits_hex=bits,
            predicted_heading_bits_hex=bits * 4,
            simulation_heading_error_deg=3,
            rendered_heading_error_deg=5,
            max_ik_correction=.1,
            left_recorded_contact=1,
            right_recorded_contact=1,
            left_locked=1,
            right_locked=1,
            left_target_height=0,
            right_target_height=0,
            rendered_hips_y=hips_y,
            support_retargeted_hips_y=hips_y,
            ik_adjusted_hips_y=hips_y,
            footprint_status=("ok" if ik else "invalid-input"),
            footprint_blocked=0,
            footprint_blocked_reason="clear",
            footprint_root_height=0,
            left_footprint_min_height=0,
            left_footprint_max_height=(.08 if ik and multilevel else 0),
            right_footprint_min_height=0,
            right_footprint_max_height=0,
            left_maximum_root_split=(.08 if ik and multilevel else 0),
            left_footprint_multilevel=int(ik and multilevel and frame == 40),
            runtime_root_surface_height=0,
            runtime_left_toe_surface_height=(
                .08 if multilevel and frame == 41 else 0),
            runtime_right_toe_surface_height=0,
            requested_velocity_x=.25,
            requested_velocity_z=.5,
            applied_velocity_x=.25,
            applied_velocity_z=.5,
            walkability_class=1,
            blocked=0,
            blocked_reason="clear",
            accepted_state_digest_hex=f"{frame + 1:016x}",
        ))
    if not ik:
        for item in rows:
            for name in IK_SUFFIX:
                item[name] = "0"
            item.update({
                "ik_stop_reason": "none",
                "ik_candidate_clearance_status": "invalid-input",
                "left_swing_selected_index": str(2 ** 32 - 1),
                "left_swing_actual_sphere_center_bits_hex": "00000000" * 12,
                "left_swing_selected_clearance_status": "invalid-input",
                "right_swing_selected_index": str(2 ** 32 - 1),
                "right_swing_actual_sphere_center_bits_hex": "00000000" * 12,
                "right_swing_selected_clearance_status": "invalid-input",
            })
    return rows


def gate_l_check(rows, **changes):
    options = {
        "expected_end_x": 1.2,
        "expected_end_z": 2.4,
        "expected_heading": "positive-x",
        "require_multilevel": True,
    }
    options.update(changes)
    return runtime_log.check_gate_l(rows, **options)


def rejection_rows(stage="landing-patch"):
    accepted = runtime_row(
        0, ik_enabled=1, accepted_state_digest_hex="1" * 16)
    rejected_copy = dict(accepted)
    rejected_copy["frame"] = "1"
    rows = [accepted, rejected_copy]
    rejected = rows[1]
    rejected.update({
        "frame_rejected": "1",
        "ik_safe_stop_latched": "1",
        "frame_rejection_stage": stage,
        "rejected_attempted_footprint_available": "1",
    })
    if stage == "landing-patch":
        rejected.update({
            "rejected_attempted_ik_available": "1",
            "rejected_stop_reason": "landing-patch-unavailable",
            "rejected_left_landing_expected": "1",
            "rejected_left_landing_patch_ready": "0",
            "rejected_left_landing_sample": "1",
            "rejected_left_landing_surface_status": "valid",
            "rejected_left_landing_walkability_class": "1",
            "rejected_left_landing_patch_maximum_residual": ".006",
        })
    elif stage == "pose-certificate":
        rejected.update({
            "rejected_stop_reason": "pose-clearance-rejected",
            "rejected_attempted_pose_available": "0",
            "rejected_pose_status": "outside-domain",
        })
    return rows


def transition_rejection_rows():
    accepted = runtime_row(
        0, ik_enabled=1, accepted_state_digest_hex="1" * 16)
    transitioned = runtime_row(
        1, ik_enabled=1, accepted_state_digest_hex="2" * 16,
        searched=1, transitioned=1,
        query_database_frame=100, query_range=0,
        selected_database_frame=500, database_frame=501,
        range=2, source_range=2,
        incumbent_cost=2, selected_cost=1)
    rejected = dict(transitioned)
    rejected.update({
        "frame": "2",
        "frame_rejected": "1",
        "ik_safe_stop_latched": "1",
        "frame_rejection_stage": "landing-patch",
        "rejected_attempted_footprint_available": "1",
        "rejected_attempted_ik_available": "1",
        "rejected_stop_reason": "landing-patch-unavailable",
        "rejected_left_landing_expected": "1",
        "rejected_left_landing_patch_ready": "0",
        "rejected_left_landing_sample": "1",
        "rejected_left_landing_surface_status": "valid",
        "rejected_left_landing_walkability_class": "1",
        "rejected_left_landing_patch_maximum_residual": ".006",
    })
    return [accepted, transitioned, rejected]


def gate_l2_pair_rows():
    positive = gate_l_rows(heading="positive-x")
    negative = gate_l_rows(heading="negative-x")
    for item in positive[20:25]:
        item["left_target_height"] = ".10"
        item["right_target_height"] = "0"
    for item in negative[60:65]:
        item["left_target_height"] = "0"
        item["right_target_height"] = ".10"
    return positive, negative


def exit_safe_stop_rows(ik=1):
    rows = gate_l_rows(
        heading="positive-x", count=72, ik=ik,
        scene="stairs-standard", route="landing-side-exit-stress",
        end_x=.8, end_z=1.2, multilevel=False)
    for item in rows:
        item["route_complete"] = "0"
    if not ik:
        rows[-1]["route_complete"] = "1"
        return rows
    for item in rows:
        item["left_recorded_contact"] = "0"
    baseline = rows[69]
    held_digest = baseline["accepted_state_digest_hex"]
    held_fields = (
        "scene_generation", "scene_frame", "scene_reset_count",
        "query_database_frame", "query_range", "selected_database_frame",
        "database_frame", "range", "source_range", "route_waypoint",
        "left_recorded_contact", "right_recorded_contact",
        "left_target_height", "right_target_height",
        "simulation_x", "simulation_z", "rendered_hips_y",
        "ik_minimum_clearance", "rendered_min_clearance",
    )
    for index in (70, 71):
        item = rows[index]
        for name in held_fields:
            item[name] = baseline[name]
        item.update({
            "accepted_state_digest_hex": held_digest,
            "frame_rejected": "1",
            "frame_rejection_stage": "landing-patch",
            "ik_safe_stop_latched": "1",
            "rejected_attempted_footprint_available": "1",
            "rejected_attempted_ik_available": "1",
            "rejected_stop_reason": "landing-patch-unavailable",
            "rejected_left_landing_expected": "1",
            "rejected_left_landing_patch_ready": "0",
            "rejected_left_landing_sample": "1",
            "rejected_left_landing_surface_status": "valid",
            "rejected_left_landing_walkability_class": "1",
            "rejected_left_landing_patch_maximum_residual": ".006",
            "rejected_left_target_normal_y": "1",
            "rejected_left_reachable": "1",
            "rejected_right_target_normal_y": "1",
            "rejected_right_reachable": "1",
        })
    return rows


GATE_E_NORMAL_CASES = (
    ("grail-curb-low", "curb-forward"),
    ("stairs-shallow", "ascent-landing-descent"),
    ("stairs-standard", "ascent-landing-descent"),
    ("stairs-unseen-variable", "ascent-landing-descent"),
    ("ramp-05-up-down", "up-landing-down"),
    ("ramp-10-up-down", "up-landing-down"),
    ("cross-slope-05", "forward-cross-slope"),
    ("cross-slope-10", "forward-cross-slope"),
    ("mixed-multilevel", "full-course"),
)

GATE_E_STRESS_CASES = (
    ("grail-curb-default", "curb-forward"),
    ("grail-curb-medium", "curb-forward"),
    ("grail-curb-high", "curb-forward"),
    ("ramp-15-stress", "up-landing-down"),
)

GATE_D_IK_CASES = (
    ("blocked-course", "wall-safe-stop"),
    ("blocked-course", "ramp-safe-stop"),
)

SWING_LIFT_BITS = (
    0x00000000, 0x3b03126f, 0x3b83126f, 0x3bc49ba6,
    0x3c03126f, 0x3c23d70a, 0x3c449ba6, 0x3c656042,
    0x3c83126f, 0x3c9374bc, 0x3ca3d70a, 0x3cb43958,
    0x3cc49ba6, 0x3cd4fdf4, 0x3ce56042, 0x3cf5c28f,
    0x3d03126f, 0x3d0b4396, 0x3d1374bc, 0x3d1ba5e3,
    0x3d23d70a, 0x3d2c0831, 0x3d343958, 0x3d3c6a7f,
    0x3d449ba6, 0x3d4ccccd, 0x3d54fdf4, 0x3d5d2f1b,
    0x3d656042, 0x3d6d9168, 0x3d75c28f, 0x3d7df3b6,
    0x3d83126f, 0x3d872b02, 0x3d8b4396, 0x3d8f5c29,
    0x3d9374bc, 0x3d978d50, 0x3d9ba5e3, 0x3d9fbe77,
    0x3da3d70a,
)


def _set_gate_e_contact(item, foot, planted):
    item[f"{foot}_contact"] = str(int(planted))
    item[f"{foot}_recorded_contact"] = str(int(planted))
    item[f"{foot}_locked"] = str(int(planted))
    item[f"{foot}_reachable"] = "1"
    item[f"{foot}_observed_lock_drift"] = ".02" if planted else "0"
    item[f"{foot}_lock_drift"] = ".01" if planted else "0"
    item[f"{foot}_contact_residual"] = ".001"
    item[f"{foot}_sole_normal_alignment"] = "1"
    item[f"{foot}_target_normal_x"] = "0"
    item[f"{foot}_target_normal_y"] = "1"
    item[f"{foot}_target_normal_z"] = "0"


def _set_gate_e_contact_bypass(item, foot):
    item[f"{foot}_swing_candidates_evaluated"] = "0"
    item[f"{foot}_swing_selected_index"] = str(2 ** 32 - 1)
    item[f"{foot}_swing_selected_lift_bits"] = "0"
    item[f"{foot}_swing_materialized_command_y_bits"] = "0"
    item[f"{foot}_swing_actual_sphere_center_bits_hex"] = "00000000" * 12
    item[f"{foot}_swing_selected_clearance_status"] = "invalid-input"
    item[f"{foot}_swing_selected_controller_constraints_passed"] = "0"
    item[f"{foot}_swing_selected_clearance_certified"] = "0"
    item[f"{foot}_swing_lower_margin"] = "0"
    item[f"{foot}_swing_witness_upper_margin"] = "0"
    for scope in ("selected", "total"):
        for suffix in (
                "point_queries", "cells_visited",
                "primitive_triangle_pairs", "face_patches",
                "candidate_tests", "subdivision_nodes"):
            item[f"{foot}_swing_{scope}_work_{suffix}"] = "0"


def _set_gate_e_swing(item, foot, selected_index=1):
    _set_gate_e_contact(item, foot, False)
    evaluated = selected_index + 1
    item[f"{foot}_swing_candidates_evaluated"] = str(evaluated)
    item[f"{foot}_swing_selected_index"] = str(selected_index)
    item[f"{foot}_swing_selected_lift_bits"] = str(
        SWING_LIFT_BITS[selected_index])
    target_height = struct.unpack(
        ">f", struct.pack(">f", float(item[f"{foot}_target_height"])))[0]
    lift = struct.unpack(
        ">f", SWING_LIFT_BITS[selected_index].to_bytes(4, "big"))[0]
    item[f"{foot}_swing_materialized_command_y_bits"] = str(
        int.from_bytes(struct.pack(">f", target_height + lift), "big"))
    sphere = "".join(
        struct.pack(">f", value).hex()
        for value in (.1, .8, .2) * 4)
    item[f"{foot}_swing_actual_sphere_center_bits_hex"] = sphere
    item[f"{foot}_swing_selected_clearance_status"] = "ok"
    item[f"{foot}_swing_selected_controller_constraints_passed"] = "1"
    item[f"{foot}_swing_selected_clearance_certified"] = "1"
    item[f"{foot}_swing_lower_margin"] = ".01"
    item[f"{foot}_swing_witness_upper_margin"] = ".0100005"
    selected = (0, 1, 2, 3, 4, 5)
    total = tuple(value * evaluated for value in selected)
    suffixes = (
        "point_queries", "cells_visited", "primitive_triangle_pairs",
        "face_patches", "candidate_tests", "subdivision_nodes",
    )
    for suffix, value in zip(suffixes, selected):
        item[f"{foot}_swing_selected_work_{suffix}"] = str(value)
    for suffix, value in zip(suffixes, total):
        item[f"{foot}_swing_total_work_{suffix}"] = str(value)


def gate_e_rows(
        count=800, scene="grail-curb-low", route="curb-forward"):
    treatment = gate_l_rows(
        heading="forward", count=count, ik=1, scene=scene, route=route,
        end_x=0, end_z=6, multilevel=False)
    control = gate_l_rows(
        heading="forward", count=count, ik=0, scene=scene, route=route,
        end_x=0, end_z=6, multilevel=False)
    for index, (on, off) in enumerate(zip(treatment, control)):
        left_planted = (index // 10) % 2 == 0
        for foot, planted in (
                ("left", left_planted), ("right", not left_planted)):
            _set_gate_e_contact(on, foot, planted)
            if planted:
                _set_gate_e_contact_bypass(on, foot)
            else:
                _set_gate_e_swing(on, foot)
            off[f"{foot}_contact"] = str(int(planted))
        on["accepted_state_digest_hex"] = f"{index + 1:016x}"
        off["accepted_state_digest_hex"] = f"{index + 1001:016x}"
    return treatment, control


def gate_e_safe_stop_rows(stop=700):
    treatment, control = gate_e_rows(
        scene="grail-curb-medium", route="curb-forward")
    for item in treatment + control:
        item["walkability_class"] = "2"
    held = dict(treatment[stop - 1])
    for index in range(stop, len(treatment)):
        item = dict(held)
        item.update({
            "frame": str(index),
            "actual_simulation_speed": "0",
            "applied_speed": "0",
            "applied_velocity_x": "0",
            "applied_velocity_y": "0",
            "applied_velocity_z": "0",
            "route_complete": "0",
            "frame_rejected": "1",
            "frame_rejection_stage": "footprint",
            "ik_safe_stop_latched": "1",
            "rejected_attempted_footprint_available": "1",
            "rejected_stop_reason": "footprint-blocked",
        })
        treatment[index] = item
    return treatment, control


def gate_d_ik_rows(route="wall-safe-stop"):
    treatment, control = gate_e_rows(
        count=600, scene="blocked-course", route=route)
    for index, (on, off) in enumerate(zip(treatment, control)):
        blocked = index >= 100
        speed = max(0.0, .5 - max(0, index - 100) * .05)
        position = min(index * .02, 2.2)
        for item in (on, off):
            item.update({
                "blocked": str(int(blocked)),
                "blocked_reason": "blocked-cell" if blocked else "clear",
                "blocked_distance": ".03" if blocked else "3.4e38",
                "applied_speed": str(speed),
                "simulation_x": "0",
                "simulation_z": str(position),
                "applied_velocity_x": "0",
                "applied_velocity_y": "0",
                "applied_velocity_z": str(speed),
                "route_complete": str(int(index >= 200)),
            })
        on["actual_simulation_speed"] = str(speed)
    stop = 111
    held = dict(treatment[stop - 1])
    for index in range(stop, len(treatment)):
        item = dict(held)
        item.update({
            "frame": str(index),
            "actual_simulation_speed": "0",
            "applied_speed": "0",
            "applied_velocity_x": "0",
            "applied_velocity_y": "0",
            "applied_velocity_z": "0",
            "route_complete": "0",
            "frame_rejected": "1",
            "frame_rejection_stage": "footprint",
            "ik_safe_stop_latched": "1",
            "rejected_attempted_footprint_available": "1",
            "rejected_stop_reason": "footprint-blocked",
        })
        treatment[index] = item
    return treatment, control


class RuntimeLogTests(unittest.TestCase):
    def test_runtime_columns_append_after_gate_a(self):
        self.assertEqual(len(GATE_A_COLUMNS), 62)
        self.assertEqual(len(IK_SUFFIX), 92)
        self.assertEqual(len(DIRECTIONAL_SUFFIX), 109)
        self.assertEqual(len(RUNTIME_COLUMNS), 305)
        self.assertEqual(len(set(RUNTIME_COLUMNS)), 305)
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[-len(RUNTIME_SUFFIX):]), RUNTIME_SUFFIX)
        self.assertEqual(
            tuple(RUNTIME_COLUMNS[:-len(RUNTIME_SUFFIX)]), GATE_A_COLUMNS)

    def test_directional_columns_append_after_exact_ik_suffix(self):
        start = len(GATE_A_COLUMNS) + len(LEGACY_RUNTIME_SUFFIX)
        middle = start + len(IK_SUFFIX)
        self.assertEqual(tuple(RUNTIME_COLUMNS[start:middle]), IK_SUFFIX)
        self.assertEqual(tuple(RUNTIME_COLUMNS[middle:]), DIRECTIONAL_SUFFIX)

    def test_gate_e_accepts_only_exact_normal_matrix_and_800_forward_rows(self):
        self.assertEqual(
            runtime_log.GATE_E_ROUTES, frozenset(GATE_E_NORMAL_CASES))
        treatment, control = gate_e_rows()
        report = runtime_log.check_gate_e_pair(
            treatment, control,
            expected_scene="grail-curb-low",
            expected_route="curb-forward")
        self.assertEqual(report["frames"], 800)
        self.assertEqual(report["scene"], "grail-curb-low")
        self.assertEqual(report["route"], "curb-forward")
        self.assertGreaterEqual(report["left_planted_samples"], 10)
        self.assertGreaterEqual(report["right_planted_samples"], 10)
        self.assertLess(
            report["mean_corrected_lock_drift"],
            report["mean_observed_lock_drift"])

        for changed_rows, scene, route, diagnostic in (
                (treatment[:-1], "grail-curb-low", "curb-forward",
                 "exactly 800"),
                (treatment, "grail-curb-default", "curb-forward",
                 "allowlist"),
                (treatment, "grail-curb-low", "wrong-route",
                 "allowlist")):
            with self.subTest(scene=scene, route=route, diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_pair(
                        changed_rows, control,
                        expected_scene=scene, expected_route=route)

        changed = [dict(item) for item in treatment]
        changed[200]["scene_id"] = "grail-curb-default"
        with self.assertRaisesRegex(ValueError, "scene/route"):
            runtime_log.check_gate_e_pair(
                changed, control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

        changed = [dict(item) for item in treatment]
        changed[200]["desired_heading_bits_hex"] = heading_bits("positive-x")
        with self.assertRaisesRegex(ValueError, "forward heading"):
            runtime_log.check_gate_e_pair(
                changed, control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

    def test_gate_e_locks_treatment_control_and_pair_invariant_groups(self):
        treatment, control = gate_e_rows()
        changed = [dict(item) for item in treatment]
        for item in changed:
            item["rendered_hips_y"] = str(
                float(item["rendered_hips_y"]) + .01)
            item["ik_adjusted_hips_y"] = str(
                float(item["ik_adjusted_hips_y"]) + .01)
        self.assertEqual(runtime_log.check_gate_e_pair(
            changed, control,
            expected_scene="grail-curb-low",
            expected_route="curb-forward")["frames"], 800)

        treatment_mutations = (
            ("ik_enabled", "0", "IK enabled"),
            ("ik_applied", "0", "IK applied"),
            ("matching_enabled", "0", "matching"),
            ("support_retargeting_enabled", "0", "support retargeting"),
            ("effective_terrain_weight", "3", "weight"),
            ("route_complete", "0", "complete"),
            ("frame_rejected", "1", "rejection|scene_frame"),
        )
        for name, value, diagnostic in treatment_mutations:
            changed = [dict(item) for item in treatment]
            index = 799 if name == "route_complete" else 200
            changed[index][name] = value
            with self.subTest(treatment=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_pair(
                        changed, control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

        for name in ("adjustment_enabled", "clamping_enabled"):
            changed = [dict(item) for item in treatment]
            changed_control = [dict(item) for item in control]
            changed[200][name] = "0"
            changed_control[200][name] = "0"
            with self.subTest(treatment_configuration=name):
                with self.assertRaisesRegex(ValueError, name.split("_")[0]):
                    runtime_log.check_gate_e_pair(
                        changed, changed_control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

        changed_control = [dict(item) for item in control]
        changed_control[200]["ik_enabled"] = "1"
        with self.assertRaisesRegex(ValueError, "IK (?:must be|disabled)"):
            runtime_log.check_gate_e_pair(
                treatment, changed_control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

        invariant_mutations = (
            ("source_name", "different-source", "matching"),
            ("support_height", ".001", "support"),
            ("simulation_x", ".001", "simulation"),
            ("requested_velocity_z", ".5000001", "intent"),
        )
        for name, value, group in invariant_mutations:
            changed_control = [dict(item) for item in control]
            changed_control[200][name] = value
            with self.subTest(pair_group=group):
                with self.assertRaisesRegex(
                        ValueError, f"pair invariant.*{name}"):
                    runtime_log.check_gate_e_pair(
                        treatment, changed_control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

    def test_gate_e_binds_contact_lock_reach_residual_normal_and_clearance(self):
        treatment, control = gate_e_rows()
        residual_limit = struct.unpack(">f", bytes.fromhex("3ba3d70a"))[0]
        mutations = (
            (20, "left_recorded_contact", "0", "recorded contact"),
            (20, "left_locked", "0", "lock"),
            (20, "left_reachable", "0", "reachable"),
            (20, "left_contact_residual",
             str(math.nextafter(residual_limit, math.inf)), "residual"),
            (20, "left_sole_normal_alignment", ".9989", "sole alignment"),
            (20, "left_target_normal_y", ".9", "unit normal"),
            (20, "left_target_normal_y", "-1", "upward normal"),
            (20, "left_candidate_toe_clearance", ".029", "candidate.*accepted"),
            (20, "left_toe_clearance", "-.00501", "planted"),
            (20, "left_knee_clearance", "-.01001", "clearance"),
            (20, "ik_minimum_clearance", "-.01001", "clearance"),
            (20, "max_ik_correction", ".35001", "correction"),
        )
        for index, name, value, diagnostic in mutations:
            changed = [dict(item) for item in treatment]
            changed[index][name] = value
            if name == "left_toe_clearance":
                changed[index]["left_candidate_toe_clearance"] = value
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_pair(
                        changed, control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

    def test_gate_e_owns_canonical_contact_bypass(self):
        treatment, control = gate_e_rows()
        mutations = (
            ("left_swing_candidates_evaluated", "1",
             "contact bypass|no-candidate"),
            ("left_swing_selected_index", "0",
             "contact bypass|selected swing"),
            ("left_swing_selected_lift_bits", "1",
             "contact bypass|no-candidate"),
            ("left_swing_materialized_command_y_bits", "1",
             "contact bypass|no-candidate"),
            ("left_swing_actual_sphere_center_bits_hex",
             "3f800000" + "00000000" * 11,
             "contact bypass|no-candidate"),
            ("left_swing_selected_clearance_status", "ok",
             "contact bypass|no-candidate"),
            ("left_swing_selected_controller_constraints_passed", "1",
             "contact bypass|no-candidate"),
            ("left_swing_selected_clearance_certified", "1",
             "contact bypass|no-candidate"),
            ("left_swing_lower_margin", ".001",
             "contact bypass|no-candidate"),
            ("left_swing_selected_work_cells_visited", "1",
             "contact bypass|no-candidate"),
            ("left_swing_total_work_cells_visited", "1",
             "contact bypass"),
        )
        for name, value, diagnostic in mutations:
            changed = [dict(item) for item in treatment]
            changed[20][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_pair(
                        changed, control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

    def test_gate_e_owns_full_selected_swing_ladder_and_work(self):
        treatment, control = gate_e_rows()
        self.assertEqual(runtime_log.check_gate_e_pair(
            treatment, control,
            expected_scene="grail-curb-low",
            expected_route="curb-forward")["frames"], 800)
        mutations = (
            ("left_swing_candidates_evaluated", "3",
             "evaluated|selected swing index"),
            ("left_swing_selected_index", "0",
             "selected index|selected swing index"),
            ("left_swing_selected_lift_bits", "0", "lift bits"),
            ("left_swing_materialized_command_y_bits", str(0x7f800000),
             "materialized"),
            ("left_swing_materialized_command_y_bits",
             str(SWING_LIFT_BITS[2]), "materialized.*bits"),
            ("left_swing_actual_sphere_center_bits_hex",
             "7f800000" + "00000000" * 11, "non-finite"),
            ("left_swing_selected_clearance_status", "uncertified", "status"),
            ("left_swing_selected_controller_constraints_passed", "0",
             "constraints"),
            ("left_swing_selected_clearance_certified", "0", "certified"),
            ("left_swing_lower_margin", "-.001", "margin"),
            ("left_swing_witness_upper_margin", ".010002", "width"),
            ("left_swing_selected_work_point_queries", "1", "work"),
            ("left_swing_selected_work_cells_visited", "257", "work"),
            ("left_swing_total_work_cells_visited", "0", "total work"),
        )
        for name, value, diagnostic in mutations:
            changed = [dict(item) for item in treatment]
            changed[10][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_pair(
                        changed, control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

    def test_gate_e_requires_ten_samples_per_foot_and_strict_mean_improvement(self):
        treatment, control = gate_e_rows()
        for index, (on, off) in enumerate(zip(treatment, control)):
            if index < 9:
                _set_gate_e_contact(on, "left", True)
                _set_gate_e_contact_bypass(on, "left")
                off["left_contact"] = "1"
            else:
                _set_gate_e_swing(on, "left")
                off["left_contact"] = "0"
        with self.assertRaisesRegex(ValueError, "left.*10 planted"):
            runtime_log.check_gate_e_pair(
                treatment, control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

        treatment, control = gate_e_rows()
        for item in treatment:
            if item["left_locked"] == "1":
                item["left_lock_drift"] = item["left_observed_lock_drift"]
            if item["right_locked"] == "1":
                item["right_lock_drift"] = item["right_observed_lock_drift"]
        with self.assertRaisesRegex(ValueError, "strictly improve"):
            runtime_log.check_gate_e_pair(
                treatment, control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

        treatment, control = gate_e_rows()
        for item in treatment:
            if item["left_locked"] == "1":
                item["left_observed_lock_drift"] = ".01"
                item["left_lock_drift"] = ".02"
            if item["right_locked"] == "1":
                item["right_observed_lock_drift"] = ".1"
                item["right_lock_drift"] = "0"
        with self.assertRaisesRegex(ValueError, "left.*strictly improve"):
            runtime_log.check_gate_e_pair(
                treatment, control,
                expected_scene="grail-curb-low",
                expected_route="curb-forward")

        for name in ("left_observed_lock_drift", "left_lock_drift"):
            treatment, control = gate_e_rows()
            treatment[20][name] = "-.001"
            with self.subTest(nonnegative=name):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    runtime_log.check_gate_e_pair(
                        treatment, control,
                        expected_scene="grail-curb-low",
                        expected_route="curb-forward")

    def test_gate_e_stress_accepts_exact_traverse_or_atomic_safe_stop(self):
        self.assertEqual(
            runtime_log.GATE_E_STRESS_ROUTES,
            frozenset(GATE_E_STRESS_CASES))
        treatment, control = gate_e_rows(
            scene="grail-curb-medium", route="curb-forward")
        for item in treatment + control:
            item["walkability_class"] = "2"
        report = runtime_log.check_gate_e_stress_pair(
            treatment, control,
            expected_scene="grail-curb-medium",
            expected_route="curb-forward")
        self.assertEqual(report["branch"], "traverse")

        treatment, control = gate_e_safe_stop_rows()
        report = runtime_log.check_gate_e_stress_pair(
            treatment, control,
            expected_scene="grail-curb-medium",
            expected_route="curb-forward")
        self.assertEqual(report["branch"], "safe-stop")
        self.assertEqual(report["first_rejected_frame"], 700)

        mutations = (
            (750, "actual_simulation_speed", ".001", "stopped"),
            (750, "simulation_z", "6.021", "displacement"),
            (750, "support_height", ".021", "support rise"),
            (750, "accepted_state_digest_hex", "f" * 16, "digest"),
            (750, "ik_adjusted_hips_y", ".9", "atomic"),
            (750, "scene_frame", "700", "atomic|scene_frame"),
            (100, "source_name", "different", "pair invariant"),
        )
        for index, name, value, diagnostic in mutations:
            treatment, control = gate_e_safe_stop_rows()
            treatment[index][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_e_stress_pair(
                        treatment, control,
                        expected_scene="grail-curb-medium",
                        expected_route="curb-forward")

        treatment, control = gate_e_safe_stop_rows()
        treatment[750]["frame_rejected"] = "0"
        with self.assertRaisesRegex(ValueError, "tail|rejection|scene_frame"):
            runtime_log.check_gate_e_stress_pair(
                treatment, control,
                expected_scene="grail-curb-medium",
                expected_route="curb-forward")

        treatment, control = gate_e_safe_stop_rows()
        treatment[750].update({
            "frame_rejection_stage": "ik-candidate",
            "rejected_stop_reason": "target-unreachable",
            "rejected_attempted_ik_available": "1",
        })
        with self.assertRaisesRegex(ValueError, "stage/reason/availability"):
            runtime_log.check_gate_e_stress_pair(
                treatment, control,
                expected_scene="grail-curb-medium",
                expected_route="curb-forward")

        for name, value in (
                ("frame_rejection_stage", "ik-candidate"),
                ("rejected_stop_reason", "target-unreachable"),
                ("rejected_attempted_footprint_available", "0"),
                ("rejected_attempted_ik_available", "1"),
                ("rejected_attempted_pose_available", "1"),
                ("route_complete", "1")):
            treatment, control = gate_e_safe_stop_rows()
            treatment[750][name] = value
            with self.subTest(stress_rejection_owner=name):
                with self.assertRaises(ValueError):
                    runtime_log.check_gate_e_stress_pair(
                        treatment, control,
                        expected_scene="grail-curb-medium",
                        expected_route="curb-forward")

    def test_gate_d_ik_pair_reuses_control_gate_and_proves_atomic_hold(self):
        self.assertEqual(
            runtime_log.GATE_D_IK_ROUTES, frozenset(GATE_D_IK_CASES))
        treatment, control = gate_d_ik_rows()
        report = runtime_log.check_gate_d_ik_pair(
            treatment, control,
            expected_scene="blocked-course",
            expected_route="wall-safe-stop")
        self.assertEqual(report["frames"], 600)
        self.assertGreaterEqual(report["stopped_frames"], 25)
        self.assertGreaterEqual(report["minimum_blocked_distance"], .0199)

        mutations = (
            (112, "walkability_class", "0", "class-0"),
            (112, "blocked_distance", ".0198", "distance"),
            (112, "rendered_min_clearance", "-.01001", "clearance"),
            (112, "support_height", ".021", "support rise"),
            (112, "accepted_state_digest_hex", "f" * 16, "digest"),
            (112, "ik_adjusted_hips_y", ".9", "root-reach|atomic"),
        )
        for index, name, value, diagnostic in mutations:
            treatment, control = gate_d_ik_rows()
            treatment[index][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_d_ik_pair(
                        treatment, control,
                        expected_scene="blocked-course",
                        expected_route="wall-safe-stop")

        treatment, control = gate_d_ik_rows()
        for item in treatment[112:590]:
            item["applied_speed"] = ".1"
            item["actual_simulation_speed"] = ".1"
        with self.assertRaisesRegex(ValueError, "25.*stopped"):
            runtime_log.check_gate_d_ik_pair(
                treatment, control,
                expected_scene="blocked-course",
                expected_route="wall-safe-stop")

        treatment, control = gate_d_ik_rows()
        treatment[200]["applied_speed"] = ".1"
        treatment[200]["actual_simulation_speed"] = ".1"
        with self.subTest(blocked_tail="late-motion"):
            with self.assertRaisesRegex(ValueError, "tail.*stopped"):
                runtime_log.check_gate_d_ik_pair(
                    treatment, control,
                    expected_scene="blocked-course",
                    expected_route="wall-safe-stop")

        treatment, control = gate_d_ik_rows()
        treatment[200].update({
            "frame_rejection_stage": "ik-candidate",
            "rejected_stop_reason": "target-unreachable",
            "rejected_attempted_ik_available": "1",
        })
        with self.subTest(blocked_tail="alternate-rejection"):
            with self.assertRaisesRegex(
                    ValueError, "stage/reason/availability"):
                runtime_log.check_gate_d_ik_pair(
                    treatment, control,
                    expected_scene="blocked-course",
                    expected_route="wall-safe-stop")

        for name, value in (
                ("frame_rejection_stage", "ik-candidate"),
                ("rejected_stop_reason", "target-unreachable"),
                ("rejected_attempted_footprint_available", "0"),
                ("rejected_attempted_ik_available", "1"),
                ("rejected_attempted_pose_available", "1"),
                ("route_complete", "1")):
            treatment, control = gate_d_ik_rows()
            treatment[200][name] = value
            with self.subTest(blocked_rejection_owner=name):
                with self.assertRaises(ValueError):
                    runtime_log.check_gate_d_ik_pair(
                        treatment, control,
                        expected_scene="blocked-course",
                        expected_route="wall-safe-stop")

        treatment, control = gate_d_ik_rows()
        control[120]["walkability_class"] = "0"
        with self.assertRaisesRegex(ValueError, "Gate D entered blocked"):
            runtime_log.check_gate_d_ik_pair(
                treatment, control,
                expected_scene="blocked-course",
                expected_route="wall-safe-stop")

    def test_gate_d_ik_pair_causally_binds_control_at_rejection(self):
        treatment, control = gate_d_ik_rows()
        first_rejected = next(
            index for index, item in enumerate(treatment)
            if int(item["frame_rejected"]) == 1)
        self.assertEqual(first_rejected, 111)
        self.assertEqual(control[first_rejected - 1]["blocked"], "1")
        control[first_rejected].update({
            "blocked": "0",
            "blocked_reason": "clear",
        })
        with self.assertRaisesRegex(ValueError, "paired control.*rejection"):
            runtime_log.check_gate_d_ik_pair(
                treatment, control,
                expected_scene="blocked-course",
                expected_route="wall-safe-stop")

    def test_rejected_tail_freezes_accepted_matching_provenance(self):
        for name, mutate in (
                ("source_name", lambda item: "forged-source"),
                ("query_bits_hex", lambda item: "3f800000" + item[8:]),
                ("source_index", lambda item: str(int(item) + 1))):
            treatment, control = gate_d_ik_rows()
            treatment[112][name] = mutate(treatment[112][name])
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "root-reach|atomic"):
                    runtime_log.check_gate_d_ik_pair(
                        treatment, control,
                        expected_scene="blocked-course",
                        expected_route="wall-safe-stop")

    def test_gate_e_cli_dispatches_all_modes_and_locks_composition(self):
        normal, normal_off = gate_e_rows()
        stress, stress_off = gate_e_safe_stop_rows()
        blocked, blocked_off = gate_d_ik_rows()
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, values in (
                    ("normal", normal), ("normal-off", normal_off),
                    ("stress", stress), ("stress-off", stress_off),
                    ("blocked", blocked), ("blocked-off", blocked_off)):
                paths[name] = os.path.join(directory, name + ".csv")
                write_columns(paths[name], values, RUNTIME_COLUMNS)
            cases = (
                ("gate-e", paths["normal"], paths["normal-off"],
                 "grail-curb-low", "curb-forward"),
                ("gate-e-stress", paths["stress"], paths["stress-off"],
                 "grail-curb-medium", "curb-forward"),
                ("gate-d-ik", paths["blocked"], paths["blocked-off"],
                 "blocked-course", "wall-safe-stop"),
            )
            for flag, primary, control_path, scene, route in cases:
                output = io.StringIO()
                with self.subTest(flag=flag), contextlib.redirect_stdout(output):
                    self.assertEqual(runtime_log.main([
                        primary, f"--{flag}",
                        "--compare-ik-off", control_path,
                        "--expected-scene", scene,
                        "--expected-route", route,
                    ]), 0)
                self.assertTrue(
                    output.getvalue().startswith(f"VALID {flag} "))

        for extra, diagnostic in (
                (["--gate-e"], "compare-ik-off.*expected-scene.*expected-route"),
                (["--gate-e", "--gate-e-stress"], "not allowed"),
                (["--expected-scene", "x"], "requires"),
                (["--gate-e", "--gate-a"], "may not combine")):
            stderr = io.StringIO()
            with self.subTest(extra=extra), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    runtime_log.main(["unused.csv", *extra])
            self.assertEqual(raised.exception.code, 2)
            self.assertRegex(stderr.getvalue(), diagnostic)

    def test_legacy_runtime_schema_remains_readable_for_earlier_gates(self):
        legacy_c = legacy_runtime_rows(gate_c_rows())
        self.assertEqual(check_rows(legacy_c)["frames"], len(legacy_c))
        self.assertEqual(
            runtime_log.check_gate_c(legacy_c)["frames"], len(legacy_c))

        legacy_d = legacy_runtime_rows(gate_d_rows())
        report = runtime_log.check_gate_d(legacy_d)
        self.assertEqual(report["frames"], len(legacy_d))
        self.assertIn("minimum_certified_clearance", report)

        legacy_f = legacy_runtime_rows(
            scene_cycle_rows(list(TASK11_SCENE_IDS)))
        self.assertEqual(runtime_log.check_gate_f(
            legacy_f, TASK11_SCENE_IDS)["complete_cycles"], 2)

    def test_gate_l_accepts_all_exact_heading_codes_on_one_physical_path(self):
        for heading in HEADING_COMPONENTS:
            rows = gate_l_rows(heading=heading)
            report = gate_l_check(rows, expected_heading=heading)
            with self.subTest(heading=heading):
                self.assertEqual(report["frames"], 120)
                self.assertEqual(report["heading"], heading)
                self.assertLessEqual(report["endpoint_error_m"], .25)

    def test_gate_l_rejects_one_bit_desired_or_predicted_heading_change(self):
        for name, offset in (
                ("desired_heading_bits_hex", 31),
                ("predicted_heading_bits_hex", 127)):
            rows = gate_l_rows()
            word = rows[33][name]
            rows[33][name] = word[:offset] + (
                "1" if word[offset] != "1" else "2") + word[offset + 1:]
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "heading bits"):
                    gate_l_check(rows)
                with self.assertRaisesRegex(ValueError, "heading bits"):
                    gate_l_check(rows, safety_only=True)

    def test_gate_l_safety_only_defers_only_rendered_quality_thresholds(self):
        rows = gate_l_rows()
        for item in rows[50:]:
            item["rendered_heading_error_deg"] = "25"
        self.assertEqual(
            gate_l_check(rows, safety_only=True)["frames"], 120)
        with self.assertRaisesRegex(ValueError, "heading error"):
            gate_l_check(rows)

        safety_mutations = (
            ("route_complete", -1, "0", "complete"),
            ("ik_safe_stop_requested", 70, "1", "safe-stop"),
            ("ik_enabled", 70, "0", "IK enabled"),
            ("ik_applied", 70, "0", "IK applied"),
            ("walkability_class", 70, "2", "class-1"),
            ("frame_rejected", 70, "1", "rejection|scene_frame"),
            ("left_toe_clearance", 70, "-.006", "planted"),
            ("left_foot_clearance", 70, "-.006", "planted"),
            ("left_candidate_toe_clearance", 70, "-.006", "planted"),
            ("rendered_min_clearance", 70, "-.011", "clearance"),
            ("ik_minimum_clearance", 70, "-.011", "clearance"),
            ("ik_candidate_minimum_clearance", 70, "-.011", "clearance"),
            ("max_ik_correction", 70, ".351", "correction"),
        )
        for name, index, value, diagnostic in safety_mutations:
            changed = gate_l_rows()
            changed[index][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    gate_l_check(changed, safety_only=True)

        endpoint = gate_l_rows(end_x=.8, end_z=1.8)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            gate_l_check(endpoint, safety_only=True)

        hips = gate_l_rows()
        hips[75]["rendered_hips_y"] = str(
            float(hips[74]["rendered_hips_y"]) + .05001)
        with self.assertRaisesRegex(ValueError, "Hips"):
            gate_l_check(hips, safety_only=True)

        without_multilevel = gate_l_rows(multilevel=False)
        with self.assertRaisesRegex(ValueError, "multilevel"):
            gate_l_check(without_multilevel, safety_only=True)

        excessive_support = gate_l_rows(
            scene="mixed-multilevel", route="tangent-level-boundary")
        excessive_support[60]["support_velocity"] = "1.51"
        with self.assertRaisesRegex(ValueError, "support velocity"):
            gate_l_check(excessive_support, safety_only=True)

        treatment = gate_l_rows()
        mismatched_control = gate_l_rows(ik=0)
        mismatched_control[25]["requested_velocity_x"] = ".25000003"
        with self.assertRaisesRegex(ValueError, "IK-off invariant"):
            gate_l_check(
                treatment, compare_ik_off=mismatched_control,
                safety_only=True)

        changed_digest = gate_l_rows()
        changed_digest[70].update({
            "left_recorded_contact": "0",
            "frame_rejected": "1",
            "ik_safe_stop_latched": "1",
            "frame_rejection_stage": "landing-patch",
            "rejected_attempted_footprint_available": "1",
            "rejected_attempted_ik_available": "1",
            "rejected_stop_reason": "landing-patch-unavailable",
            "rejected_left_landing_expected": "1",
            "rejected_left_landing_patch_ready": "0",
            "rejected_left_landing_sample": "1",
            "rejected_left_landing_surface_status": "valid",
            "rejected_left_landing_walkability_class": "1",
            "rejected_left_landing_patch_maximum_residual": ".006",
        })
        with self.assertRaisesRegex(
                ValueError, "accepted-state digest|scene_frame"):
            gate_l_check(changed_digest, safety_only=True)

    def test_gate_l_enforces_ik_route_completion_endpoint_and_no_stop(self):
        cases = (
            ("ik_enabled", 0, "0", "IK enabled"),
            ("ik_applied", 10, "0", "IK applied"),
            ("mode", 10, "terrain", "route mode"),
            ("scene_generation", 60, "1", "one generation"),
            ("walkability_class", 60, "2", "class-1"),
            ("route_complete", -1, "0", "complete"),
            ("footprint_blocked", 60, "1",
             "safe-stop|accepted footprint.*blocked"),
            ("ik_safe_stop_latched", 60, "1", "safe-stop"),
            ("frame_rejected", 60, "1", "rejection|scene_frame"),
        )
        for name, index, value, diagnostic in cases:
            rows = gate_l_rows()
            rows[index][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    gate_l_check(rows)

        rows = gate_l_rows(end_x=.8, end_z=1.8)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            gate_l_check(rows)

    def test_gate_l_enforces_physical_clearance_hips_and_correction_limits(self):
        cases = (
            ("left_toe_clearance", "-.00501", "planted toe/foot"),
            ("left_foot_clearance", "-.00501", "planted toe/foot"),
            ("right_toe_clearance", "-.00501", "planted toe/foot"),
            ("right_foot_clearance", "-.00501", "planted toe/foot"),
            ("left_candidate_toe_clearance", "-.00501",
             "planted toe/foot"),
            ("left_candidate_foot_clearance", "-.00501",
             "planted toe/foot"),
            ("right_candidate_toe_clearance", "-.00501",
             "planted toe/foot"),
            ("right_candidate_foot_clearance", "-.00501",
             "planted toe/foot"),
            ("rendered_min_clearance", "-.01001", "physical clearance"),
            ("ik_minimum_clearance", "-.01001", "physical clearance"),
            ("ik_candidate_minimum_clearance", "-.01001",
             "physical clearance"),
            ("max_ik_correction", ".35001", "correction"),
        )
        for name, value, diagnostic in cases:
            rows = gate_l_rows()
            rows[75][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    gate_l_check(rows)

        rows = gate_l_rows()
        rows[75]["rendered_hips_y"] = str(
            float(rows[74]["rendered_hips_y"]) + .05001)
        with self.assertRaisesRegex(ValueError, "Hips"):
            gate_l_check(rows)

        rows = gate_l_rows()
        rows[75]["left_recorded_contact"] = "0"
        rows[75]["left_toe_clearance"] = "-.5"
        self.assertEqual(gate_l_check(rows)["frames"], 120)

    def test_gate_l_locks_median_p95_and_maximum_rendered_heading_error(self):
        rows = gate_l_rows()
        for item in rows[50:86]:
            item["rendered_heading_error_deg"] = "10.01"
        with self.assertRaisesRegex(ValueError, "median.*heading error"):
            gate_l_check(rows)

        rows = gate_l_rows()
        for item in rows[50:54]:
            item["rendered_heading_error_deg"] = "20.01"
        with self.assertRaisesRegex(ValueError, "95th.*heading error"):
            gate_l_check(rows)

        rows = gate_l_rows()
        rows[75]["rendered_heading_error_deg"] = "35.01"
        with self.assertRaisesRegex(ValueError, "maximum.*heading error"):
            gate_l_check(rows)

    def test_gate_l_requires_multilevel_report_no_later_than_surface_split(self):
        rows = gate_l_rows()
        self.assertGreaterEqual(
            gate_l_check(rows)["maximum_surface_split_m"], .04)

        rows = gate_l_rows(multilevel=False)
        with self.assertRaisesRegex(ValueError, "multilevel"):
            gate_l_check(rows)

        rows = gate_l_rows()
        rows[40]["left_footprint_multilevel"] = "0"
        rows[42]["left_footprint_multilevel"] = "1"
        with self.assertRaisesRegex(ValueError, "same-or-earlier"):
            gate_l_check(rows)

        rows = gate_l_rows(multilevel=False)
        self.assertEqual(
            gate_l_check(rows, require_multilevel=False)["frames"], 120)

    def test_gate_l_ik_off_pair_locks_matcher_support_simulation_and_intent(self):
        treatment = gate_l_rows(ik=1)
        control = gate_l_rows(ik=0)
        report = gate_l_check(treatment, compare_ik_off=control)
        self.assertEqual(report["paired_ik_off_frames"], 120)

        for name, value in (
                ("query_bits_hex", "00000001" + "00000000" * 30),
                ("selected_database_frame", "999"),
                ("selected_cost", "1.00000012"),
                ("terrain0", ".01"),
                ("runtime_support_root_height", ".01"),
                ("runtime_support_left_toe_height", ".01"),
                ("support_left_toe_delta", ".01"),
                ("simulation_x", ".01"),
                ("matching_enabled", "0"),
                ("requested_velocity_x", ".25000003")):
            changed = gate_l_rows(ik=0)
            changed[25][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "IK-off invariant"):
                    gate_l_check(treatment, compare_ik_off=changed)

        unsafe_control = gate_l_rows(ik=0)
        unsafe_control[25]["footprint_blocked"] = "1"
        with self.assertRaisesRegex(
                ValueError, "IK-off control.*safe-stop|accepted footprint.*blocked"):
            gate_l_check(treatment, compare_ik_off=unsafe_control)

        noncanonical_control = gate_l_rows(ik=0)
        noncanonical_control[25]["footprint_status"] = "ok"
        with self.assertRaisesRegex(ValueError, "IK-off control.*footprint"):
            gate_l_check(treatment, compare_ik_off=noncanonical_control)

        for name, value in (
                ("left_footprint_multilevel", "1"),
                ("footprint_sweeps", "123"),
                ("footprint_root_height", "9"),
                ("ik_candidate_rejected", "1"),
                ("ik_candidate_clearance_status", "outside-domain"),
                ("ik_minimum_clearance", "-100")):
            noncanonical_control = gate_l_rows(ik=0)
            noncanonical_control[25][name] = value
            with self.subTest(canonical_field=name):
                with self.assertRaisesRegex(
                        ValueError,
                        "invalid-input footprint.*canonical|"
                        "accepted IK candidate|disabled IK.*canonical"):
                    gate_l_check(
                        treatment, compare_ik_off=noncanonical_control)

    def test_gate_l_support_bound_uses_forward_pair_or_tangential_cap(self):
        rows = gate_l_rows(
            scene="mixed-multilevel", route="tangent-level-boundary")
        rows[60]["support_velocity"] = "1.51"
        with self.assertRaisesRegex(ValueError, "support velocity"):
            gate_l_check(rows)

        unpaired_forward = gate_l_rows(heading="forward")
        unpaired_forward[60]["support_velocity"] = "2"
        self.assertEqual(gate_l_check(
            unpaired_forward, expected_heading="forward")["frames"], 120)

        rows = gate_l_rows()
        forward = gate_l_rows(heading="forward")
        for item in forward:
            item["support_velocity"] = ".8"
        rows[60]["support_velocity"] = "1.05"
        self.assertEqual(gate_l_check(
            rows, compare_forward=forward)["frames"], 120)
        rows[60]["support_velocity"] = "1.06"
        with self.assertRaisesRegex(ValueError, "support velocity"):
            gate_l_check(rows, compare_forward=forward)

        wrong_forward = gate_l_rows(
            heading="forward", scene="ramp-05-up-down",
            route="up-landing-down")
        with self.assertRaisesRegex(ValueError, "stairs-standard"):
            gate_l_check(gate_l_rows(), compare_forward=wrong_forward)

    def test_gate_l2_pair_requires_both_legs_in_both_three_frame_roles(self):
        positive, negative = gate_l2_pair_rows()
        report = runtime_log.check_gate_l2_pair(positive, negative)
        self.assertEqual(report["role_coverage"],
                         "left:downhill+uphill,right:downhill+uphill")

        missing = [dict(item) for item in negative]
        for item in missing:
            item["left_target_height"] = "0"
            item["right_target_height"] = "0"
        with self.assertRaisesRegex(ValueError, "leg-role coverage"):
            runtime_log.check_gate_l2_pair(positive, missing)

        short_positive = [dict(item) for item in positive]
        for item in short_positive[22:25]:
            item["left_target_height"] = "0"
            item["right_target_height"] = "0"
        with self.assertRaisesRegex(ValueError, "leg-role coverage"):
            runtime_log.check_gate_l2_pair(short_positive, negative)

        flat_positive = [dict(item) for item in positive]
        for item in flat_positive:
            item["left_footprint_multilevel"] = "0"
            item["right_footprint_multilevel"] = "0"
            item["runtime_left_toe_surface_height"] = "0"
            item["runtime_right_toe_surface_height"] = "0"
        with self.assertRaisesRegex(ValueError, "multilevel"):
            runtime_log.check_gate_l2_pair(flat_positive, negative)

        fast_positive = [dict(item) for item in positive]
        fast_positive[30]["support_velocity"] = "1.51"
        with self.assertRaisesRegex(ValueError, "support velocity"):
            runtime_log.check_gate_l2_pair(fast_positive, negative)

    def test_gate_l2_pair_locks_route_headings_completion_and_safety(self):
        positive, negative = gate_l2_pair_rows()
        cases = (
            (positive, 10, "route", "different", "one scene"),
            (positive, -1, "route_complete", "0", "complete"),
            (positive, 10, "ik_safe_stop_requested", "1", "safe-stop"),
            (negative, 10, "desired_heading_bits_hex",
             heading_bits("positive-x"), "heading bits"),
        )
        for target, index, name, value, diagnostic in cases:
            left = [dict(item) for item in positive]
            right = [dict(item) for item in negative]
            changed = left if target is positive else right
            changed[index][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_l2_pair(left, right)

    def test_gate_l2_exit_stress_accepts_only_completed_or_proven_rollback(self):
        treatment = exit_safe_stop_rows(ik=1)
        control = exit_safe_stop_rows(ik=0)
        report = runtime_log.check_gate_l2_exit_stress(treatment, control)
        self.assertEqual(report["branch"], "safe-stop")
        self.assertEqual(report["rejected_frames"], 2)
        self.assertEqual(report["paired_ik_off_frames"], 70)

        unreachable = [dict(item) for item in treatment]
        for item in unreachable:
            item["left_recorded_contact"] = "1"
        for item in unreachable[70:]:
            item.update({
                "frame_rejection_stage": "ik-candidate",
                "rejected_stop_reason": "target-unreachable",
                "rejected_left_landing_expected": "0",
                "rejected_left_landing_patch_ready": "0",
                "rejected_left_landing_sample": str(2 ** 32 - 1),
                "rejected_left_landing_surface_status": "invalid",
                "rejected_left_landing_walkability_class": "0",
                "rejected_left_landing_patch_maximum_residual": "0",
                "rejected_left_target_normal_y": "1",
                "rejected_left_reachable": "0",
                "rejected_right_reachable": "1",
            })
        self.assertEqual(runtime_log.check_gate_l2_exit_stress(
            unreachable, control)["branch"], "safe-stop")

        right_unreachable = [dict(item) for item in unreachable]
        for item in right_unreachable[70:]:
            item.update({
                "rejected_left_reachable": "1",
                "rejected_right_target_normal_y": "1",
                "rejected_right_reachable": "0",
            })
        self.assertEqual(runtime_log.check_gate_l2_exit_stress(
            right_unreachable, control)["branch"], "safe-stop")

        canonical_other_foot = [dict(item) for item in unreachable]
        for item in canonical_other_foot[70:]:
            item["rejected_left_reachable"] = "1"
            item["rejected_right_reachable"] = "1"
        with self.assertRaisesRegex(ValueError, "outside.*contract"):
            runtime_log.check_gate_l2_exit_stress(
                canonical_other_foot, control)

        no_swing = [dict(item) for item in treatment]
        for item in no_swing:
            item["left_recorded_contact"] = "0"
        for item in no_swing[70:]:
            item.update({
                "frame_rejection_stage": "ik-candidate",
                "rejected_stop_reason": "no-swing-candidate",
                "rejected_left_landing_expected": "0",
                "rejected_left_landing_patch_ready": "0",
                "rejected_left_landing_sample": str(2 ** 32 - 1),
                "rejected_left_landing_surface_status": "invalid",
                "rejected_left_landing_walkability_class": "0",
                "rejected_left_landing_patch_maximum_residual": "0",
                "rejected_left_target_normal_y": "1",
                "rejected_left_reachable": "1",
                "rejected_left_selected_clearance_status": "invalid-input",
                "rejected_right_reachable": "1",
            })
        self.assertEqual(runtime_log.check_gate_l2_exit_stress(
            no_swing, control)["branch"], "safe-stop")

        right_no_swing = [dict(item) for item in treatment]
        for item in right_no_swing:
            item["right_recorded_contact"] = "0"
        for item in right_no_swing[70:]:
            item.update({
                "frame_rejection_stage": "ik-candidate",
                "rejected_stop_reason": "no-swing-candidate",
                "rejected_left_landing_expected": "0",
                "rejected_left_landing_patch_ready": "0",
                "rejected_left_landing_sample": str(2 ** 32 - 1),
                "rejected_left_landing_surface_status": "invalid",
                "rejected_left_landing_walkability_class": "0",
                "rejected_left_landing_patch_maximum_residual": "0",
                "rejected_left_target_normal_y": "1",
                "rejected_left_reachable": "1",
                "rejected_left_selected_clearance_status": "ok",
                "rejected_right_target_normal_y": "1",
                "rejected_right_reachable": "1",
                "rejected_right_selected_clearance_status": "invalid-input",
            })
        self.assertEqual(runtime_log.check_gate_l2_exit_stress(
            right_no_swing, control)["branch"], "safe-stop")

        canonical_other_swing = [dict(item) for item in no_swing]
        for item in canonical_other_swing[70:]:
            item["rejected_left_selected_clearance_status"] = "ok"
            item["rejected_right_reachable"] = "1"
        with self.assertRaisesRegex(ValueError, "outside.*contract"):
            runtime_log.check_gate_l2_exit_stress(
                canonical_other_swing, control)

        completed = gate_l_rows(
            heading="positive-x", count=72, ik=1,
            scene="stairs-standard", route="landing-side-exit-stress",
            end_x=0, end_z=5.53, multilevel=False)
        completed_control = gate_l_rows(
            heading="positive-x", count=72, ik=0,
            scene="stairs-standard", route="landing-side-exit-stress",
            end_x=0, end_z=5.53, multilevel=False)
        self.assertEqual(runtime_log.check_gate_l2_exit_stress(
            completed, completed_control)["branch"], "complete")

        for name, value, diagnostic in (
                ("rejected_left_landing_patch_maximum_residual",
                 str(struct.unpack(">f", struct.pack(">f", .005))[0]),
                 "outside.*contract"),
                ("left_target_height", ".2", "committed target"),
                ("rejected_stop_reason", "target-unreachable", "rejection"),
                ("walkability_class", "2", "class-1"),
                ("ik_applied", "0", "IK applied"),
                ("footprint_blocked", "1", "block")):
            changed = [dict(item) for item in treatment]
            target_index = 10 if name in {
                "walkability_class", "ik_applied", "footprint_blocked"} else 70
            changed[target_index][name] = value
            if name == "left_target_height":
                changed[71][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    runtime_log.check_gate_l2_exit_stress(changed, control)

    def test_gate_l_cli_wires_safety_pairing_and_exact_requirements(self):
        treatment = gate_l_rows()
        control = gate_l_rows(ik=0)
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as on_stream, \
                tempfile.NamedTemporaryFile("w+", suffix=".csv") as off_stream:
            write_runtime_rows(on_stream, treatment)
            write_runtime_rows(off_stream, control)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runtime_log.main([
                    on_stream.name, "--gate-l-safety-only",
                    "--compare-ik-off", off_stream.name,
                    "--expected-end-x", "1.2",
                    "--expected-end-z", "2.4",
                    "--expected-heading", "positive-x",
                    "--require-multilevel",
                ]), 0)
        self.assertTrue(output.getvalue().startswith("VALID gate-l-safety "))

        for extra, diagnostic in (
                (["--gate-l"], "expected-end-x"),
                (["--gate-l", "--expected-end-x", "1",
                  "--expected-end-z", "2"], "expected-heading"),
                (["--gate-l", "--expected-end-x", "1",
                  "--expected-end-z", "2",
                  "--expected-heading", "forward"], "compare-ik-off"),
                (["--gate-l2-exit-stress", "--expected-end-x", "1"],
                 "may not combine")):
            stderr = io.StringIO()
            with self.subTest(extra=extra):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        runtime_log.main(["unused.csv", *extra])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(diagnostic, stderr.getvalue())

    def test_gate_l4_authenticates_flat_oracle_and_relative_direction(self):
        rows = gate_l_rows(
            heading="forward", count=100, ik=0,
            scene="stairs-shallow", route="flat-positive-z",
            end_x=0, end_z=1, multilevel=False)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "flat-flat-positive-z__forward__forward.csv", rows)
            report = runtime_log.check_gate_l4(
                rows, baseline,
                expected_heading="forward",
                expected_relative_direction="forward")
            self.assertEqual(report["relative_direction"], "forward")
            self.assertEqual(report["frames"], 100)

    def test_gate_l4_accepts_exact_eight_absolute_relative_pairs(self):
        for basename, (scene, route, heading, relative) in \
                runtime_log.FLAT_ORACLE_CASES.items():
            end_x, end_z = (0, 1) if route == "flat-positive-z" else (1, 0)
            rows = gate_l_rows(
                heading=heading, count=100, ik=0, scene=scene, route=route,
                end_x=end_x, end_z=end_z, multilevel=False)
            with self.subTest(basename=basename), \
                    tempfile.TemporaryDirectory() as directory:
                baseline = make_oracle_tree(directory, basename, rows)
                report = runtime_log.check_gate_l4(
                    rows, baseline, expected_heading=heading,
                    expected_relative_direction=relative)
                self.assertEqual(report["relative_direction"], relative)

    def test_gate_l4_rejects_oracle_invariant_sha_and_pairing_drift(self):
        rows = gate_l_rows(
            heading="forward", count=100, ik=0,
            scene="stairs-shallow", route="flat-positive-z",
            end_x=0, end_z=1, multilevel=False)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "flat-flat-positive-z__forward__forward.csv", rows)
            changed = [dict(item) for item in rows]
            changed[25]["simulation_x"] = ".001"
            with self.assertRaisesRegex(ValueError, "flat oracle invariant"):
                runtime_log.check_gate_l4(
                    changed, baseline, expected_heading="forward",
                    expected_relative_direction="forward")

            changed = [dict(item) for item in rows]
            changed[25]["runtime_root_surface_height"] = ".01"
            with self.assertRaisesRegex(ValueError, "flat oracle invariant"):
                runtime_log.check_gate_l4(
                    changed, baseline, expected_heading="forward",
                    expected_relative_direction="forward")

            with self.assertRaisesRegex(ValueError, "relative direction"):
                runtime_log.check_gate_l4(
                    rows, baseline, expected_heading="forward",
                    expected_relative_direction="left")

            with open(os.path.join(directory, "SHA256SUMS"),
                      "r+", encoding="ascii") as stream:
                record = stream.read()
                stream.seek(0)
                stream.write(record.replace(
                    "flat-flat-positive-x__positive-x__forward.csv\n",
                    "missing-paired-flat-oracle.csv\n", 1))
                stream.truncate()
            with self.assertRaisesRegex(ValueError, "SHA256|oracle record"):
                runtime_log.check_gate_l4(
                    rows, baseline, expected_heading="forward",
                    expected_relative_direction="forward")

    def test_gate_l4_rejects_ik_surface_split_stop_and_wrong_length(self):
        baseline_rows = gate_l_rows(
            heading="forward", count=100, ik=0,
            scene="stairs-shallow", route="flat-positive-z",
            end_x=0, end_z=1, multilevel=False)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "flat-flat-positive-z__forward__forward.csv", baseline_rows)
            for name, index, value, diagnostic in (
                    ("ik_enabled", 10, "1", "IK disabled"),
                    ("footprint_blocked", 10, "1",
                     "block/stop|accepted footprint.*blocked"),
                    ("runtime_left_toe_surface_height", 10, ".04",
                     "surface split"),
                    ("left_footprint_multilevel", 10, "1",
                     "invalid-input footprint.*canonical"),
                    ("footprint_sweeps", 10, "123",
                     "invalid-input footprint.*canonical"),
                    ("footprint_root_height", 10, "9",
                     "invalid-input footprint.*canonical"),
                    ("ik_candidate_rejected", 10, "1",
                     "accepted IK candidate|disabled IK.*canonical"),
                    ("ik_candidate_clearance_status", 10, "outside-domain",
                     "disabled IK.*canonical"),
                    ("ik_minimum_clearance", 10, "-100",
                     "disabled IK.*canonical")):
                changed = [dict(item) for item in baseline_rows]
                changed[index][name] = value
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        runtime_log.check_gate_l4(
                            changed, baseline, expected_heading="forward",
                            expected_relative_direction="forward")
            with self.assertRaisesRegex(ValueError, "exactly 100"):
                runtime_log.check_gate_l4(
                    baseline_rows[:-1], baseline,
                    expected_heading="forward",
                    expected_relative_direction="forward")

    def test_forward_baseline_authenticates_invariants_and_quality(self):
        rows = gate_l_rows(
            heading="forward", count=800, ik=0,
            scene="stairs-standard", route="ascent-landing-descent",
            end_x=0, end_z=4.8, multilevel=True)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "forward-stairs-standard__ascent-landing-descent.csv", rows)
            report = runtime_log.check_gate_l_forward_baseline(rows, baseline)
            self.assertEqual(report["frames"], 800)

            changed = [dict(item) for item in rows]
            changed[100]["simulation_z"] = "9"
            with self.assertRaisesRegex(ValueError, "forward oracle invariant"):
                runtime_log.check_gate_l_forward_baseline(changed, baseline)

            changed = [dict(item) for item in rows]
            changed[100].update({
                "footprint_status": "ok",
                "footprint_sweeps": "123",
                "footprint_root_height": "9",
            })
            with self.assertRaisesRegex(
                    ValueError, "forward.*footprint|invalid-input footprint"):
                runtime_log.check_gate_l_forward_baseline(changed, baseline)

            changed = [dict(item) for item in rows]
            changed[200]["rendered_min_clearance"] = "-.001"
            with self.assertRaisesRegex(ValueError, "clearance regressed"):
                runtime_log.check_gate_l_forward_baseline(changed, baseline)

            changed = [dict(item) for item in rows]
            changed[300]["rendered_hips_y"] = str(
                float(changed[299]["rendered_hips_y"]) + .049)
            with self.assertRaisesRegex(ValueError, "Hips step regressed"):
                runtime_log.check_gate_l_forward_baseline(changed, baseline)

    def test_gate_l_consumes_forward_baseline_through_paired_ik_off(self):
        treatment = gate_l_rows(
            heading="forward", count=800, ik=1,
            scene="stairs-standard", route="ascent-landing-descent",
            end_x=0, end_z=4.8, multilevel=True)
        control = gate_l_rows(
            heading="forward", count=800, ik=0,
            scene="stairs-standard", route="ascent-landing-descent",
            end_x=0, end_z=4.8, multilevel=True)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "forward-stairs-standard__ascent-landing-descent.csv",
                control)
            report = runtime_log.check_gate_l(
                treatment, expected_end_x=0, expected_end_z=4.8,
                expected_heading="forward", require_multilevel=True,
                compare_ik_off=control,
                compare_forward_baseline=baseline)
        self.assertEqual(report["forward_baseline_frames"], 800)

    def test_gate_l4_cli_dispatches_without_endpoint_options(self):
        rows = gate_l_rows(
            heading="forward", count=100, ik=0,
            scene="stairs-shallow", route="flat-positive-z",
            end_x=0, end_z=1, multilevel=False)
        with tempfile.TemporaryDirectory() as directory:
            baseline = make_oracle_tree(
                directory,
                "flat-flat-positive-z__forward__forward.csv", rows)
            log = os.path.join(directory, "current.csv")
            write_columns(log, rows, RUNTIME_COLUMNS)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runtime_log.main([
                    log, "--gate-l", "--compare-flat-baseline", baseline,
                    "--expected-heading", "forward",
                    "--expected-relative-direction", "forward",
                ]), 0)
        self.assertTrue(output.getvalue().startswith("VALID gate-l4 "))

    def test_gate_l2_cli_dispatches_pair_and_exit_stress(self):
        positive, negative = gate_l2_pair_rows()
        treatment = exit_safe_stop_rows(ik=1)
        control = exit_safe_stop_rows(ik=0)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, values in (
                    ("positive", positive), ("negative", negative),
                    ("exit-on", treatment), ("exit-off", control)):
                paths[name] = os.path.join(directory, name + ".csv")
                write_columns(paths[name], values, RUNTIME_COLUMNS)
            pair_output = io.StringIO()
            with contextlib.redirect_stdout(pair_output):
                self.assertEqual(runtime_log.main([
                    paths["positive"], "--gate-l2-pair", paths["negative"],
                ]), 0)
            exit_output = io.StringIO()
            with contextlib.redirect_stdout(exit_output):
                self.assertEqual(runtime_log.main([
                    paths["exit-on"], "--gate-l2-exit-stress",
                    "--compare-ik-off", paths["exit-off"],
                ]), 0)
        self.assertTrue(
            pair_output.getvalue().startswith("VALID gate-l2-pair "))
        self.assertTrue(
            exit_output.getvalue().startswith("VALID gate-l2-exit-stress "))

    def test_full_suffix_rejects_malformed_bit_words_statuses_and_landings(self):
        mutations = (
            ("desired_heading_bits_hex", "0" * 31, "float words"),
            ("predicted_heading_bits_hex",
             "7f800000" + "00000000" * 15, "non-finite"),
            ("left_swing_actual_sphere_center_bits_hex",
             "00000000" * 11, "float words"),
            ("accepted_state_digest_hex", "ABCDEF0123456789", "uint64"),
            ("footprint_status", "unknown", "unknown value"),
            ("left_landing_patch_ready", "1", "absent.*landing"),
            ("left_predicted_landing_center_x", ".1", "absent.*landing"),
            ("left_swing_selected_lift_bits", "1", "no-candidate.*canonical"),
            ("right_swing_selected_work_point_queries", "1",
             "no-candidate.*canonical"),
            ("left_footprint_min_height", ".2", "envelope"),
            ("max_ik_correction", "-.1", "nonnegative"),
        )
        for name, value, diagnostic in mutations:
            rows = [runtime_row(0)]
            rows[0][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    check_rows(rows)

    def test_full_suffix_binds_accepted_status_reasons_and_landing_readiness(self):
        simple_mutations = (
            ("ik_stop_reason", "no-swing-candidate", "accepted.*stop reason"),
            ("ik_safe_stop_requested", "1", "accepted.*safe-stop"),
            ("ik_candidate_rejected", "1", "accepted IK candidate"),
            ("ik_candidate_clearance_status", "outside-domain",
             "candidate clearance status"),
            ("footprint_blocked_reason", "blocked-cell",
             "footprint.*blocked reason"),
            ("footprint_blocked", "1", "footprint.*blocked"),
        )
        for name, value, diagnostic in simple_mutations:
            rows = [runtime_row(0, ik_enabled=1, ik_applied=1)]
            rows[0][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    check_rows(rows, allow_ik=True)

        unready = [runtime_row(
            0, ik_enabled=1, ik_applied=1, left_recorded_contact=0,
            left_landing_expected=1, left_landing_patch_ready=0,
            left_landing_sample=1, left_landing_surface_status="valid",
            left_landing_walkability_class=1,
            left_predicted_landing_normal_y=1,
            left_landing_patch_maximum_residual=.006)]
        with self.assertRaisesRegex(ValueError, "accepted.*landing.*ready"):
            check_rows(unready, allow_ik=True)

    def test_landing_readiness_uses_exact_float32_residual_limit(self):
        limit = struct.unpack(">f", bytes.fromhex("3ba3d70a"))[0]
        just_above = math.nextafter(limit, math.inf)
        self.assertLess(just_above, .005)

        accepted = [runtime_row(
            0, left_landing_expected=1, left_landing_patch_ready=1,
            left_landing_sample=1, left_landing_surface_status="valid",
            left_landing_walkability_class=1,
            left_predicted_landing_normal_y=1,
            left_landing_patch_maximum_residual=just_above)]
        with self.assertRaisesRegex(ValueError, "ready.*landing.*inconsistent"):
            check_rows(accepted)

        rejected = rejection_rows()
        rejected[1].update({
            "frame_rejection_stage": "ik-candidate",
            "rejected_stop_reason": "target-unreachable",
            "rejected_left_landing_patch_ready": "1",
            "rejected_left_landing_patch_maximum_residual": str(just_above),
        })
        with self.assertRaisesRegex(
                ValueError, "rejected left ready patch is inconsistent"):
            check_rows(rejected, allow_ik=True)

    def test_expected_landing_requires_valid_future_swing_sample(self):
        valid = [runtime_row(
            0, left_recorded_contact=0, left_landing_expected=1,
            left_landing_patch_ready=1, left_landing_sample=1,
            left_landing_surface_status="valid",
            left_landing_walkability_class=1,
            left_predicted_landing_normal_y=1,
            left_landing_patch_maximum_residual=0)]
        self.assertEqual(check_rows(valid)["frames"], 1)

        mutations = (
            ({"left_landing_sample": "0"}, "expected left landing.*malformed"),
            ({"left_landing_patch_ready": "0",
              "left_landing_surface_status": "outside"},
             "expected left landing.*malformed"),
            ({"left_landing_patch_ready": "0",
              "left_landing_walkability_class": "0"},
             "expected left landing.*malformed"),
            ({"left_recorded_contact": "1"}, "expected left landing.*swing"),
        )
        for changes, diagnostic in mutations:
            changed = [dict(valid[0])]
            changed[0].update(changes)
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    check_rows(changed)

        rejected = rejection_rows()
        rejected[1].update({
            "frame_rejection_stage": "ik-candidate",
            "rejected_stop_reason": "target-unreachable",
            "rejected_left_landing_patch_ready": "1",
            "rejected_left_landing_patch_maximum_residual": "0",
        })
        self.assertEqual(check_rows(rejected, allow_ik=True)["frames"], 2)
        rejected[1]["rejected_left_landing_sample"] = "0"
        with self.assertRaisesRegex(
                ValueError, "rejected left landing.*malformed"):
            check_rows(rejected, allow_ik=True)

    def test_finite_rejection_preserves_digest_and_locks_stage_availability(self):
        rows = rejection_rows()
        self.assertEqual(check_rows(rows, allow_ik=True)["frames"], 2)

        changed = [dict(item) for item in rows]
        changed[1]["accepted_state_digest_hex"] = "2" * 16
        with self.assertRaisesRegex(ValueError, "accepted-state digest"):
            check_rows(changed, allow_ik=True)

        for name, value in (
                ("rejected_attempted_footprint_available", "0"),
                ("rejected_attempted_ik_available", "0"),
                ("rejected_attempted_pose_available", "1"),
                ("rejected_stop_reason", "no-swing-candidate"),
                ("rejected_pose_status", "ok")):
            changed = [dict(item) for item in rows]
            changed[1][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "availability|rejection"):
                    check_rows(changed, allow_ik=True)

    def test_rejected_transition_freezes_accepted_query_cursor(self):
        rows = transition_rejection_rows()
        self.assertEqual(check_rows(rows, allow_ik=True)["frames"], 3)
        self.assertEqual(
            check_rows(rows, allow_ik=True)["transitions"], 1)

        unscheduled = [dict(item) for item in rows]
        for item in unscheduled[1:]:
            item["matching_enabled"] = "1"
            item["searched"] = "0"
        self.assertEqual(
            check_rows(unscheduled, allow_ik=True),
            {"frames": 3, "transitions": 1})

        for name, value, diagnostic in (
                ("query_database_frame", "101", "accepted query.*frame"),
                ("query_range", "1", "accepted query.*range")):
            changed = [dict(item) for item in rows]
            changed[2][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    check_rows(changed, allow_ik=True)

        changed = [dict(item) for item in rows]
        changed[1]["query_range"] = "1"
        with self.assertRaisesRegex(ValueError, "prior pose range"):
            check_rows(changed, allow_ik=True)

    def test_landing_patch_rejection_requires_attempted_unready_patch_evidence(self):
        rows = rejection_rows()
        rows[1].update({
            "rejected_left_landing_expected": "0",
            "rejected_left_landing_sample": str(2 ** 32 - 1),
            "rejected_left_landing_surface_status": "invalid",
            "rejected_left_landing_walkability_class": "0",
            "rejected_left_landing_patch_maximum_residual": "0",
        })
        with self.assertRaisesRegex(ValueError, "unready patch evidence"):
            check_rows(rows, allow_ik=True)

    def test_pose_rejection_ignores_unavailable_default_but_authenticates_available(self):
        rows = rejection_rows("pose-certificate")
        rows[1]["rejected_pose_minimum_clearance"] = "0"
        self.assertEqual(check_rows(rows, allow_ik=True)["frames"], 2)

        changed = [dict(item) for item in rows]
        changed[1]["rejected_attempted_pose_available"] = "1"
        with self.assertRaisesRegex(ValueError, "available rejected pose"):
            check_rows(changed, allow_ik=True)

        available = rejection_rows("pose-certificate")
        available[1].update({
            "rejected_attempted_pose_available": "1",
            "rejected_pose_status": "ok",
            "rejected_pose_minimum_clearance": "-.006",
        })
        self.assertEqual(check_rows(available, allow_ik=True)["frames"], 2)

        available[1]["rejected_pose_minimum_clearance"] = "0"
        with self.assertRaisesRegex(ValueError, "does not violate"):
            check_rows(available, allow_ik=True)

    def test_accepted_rows_require_canonical_rejection_defaults(self):
        for name, value in (
                ("frame_rejection_stage", "footprint"),
                ("rejected_attempted_footprint_available", "1"),
                ("rejected_stop_reason", "target-unreachable"),
                ("rejected_pose_status", "outside-domain")):
            rows = [runtime_row(0)]
            rows[0][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "accepted row"):
                    check_rows(rows)

        for name, value in (
                ("rejected_left_target_x", "1"),
                ("rejected_right_reachable", "1"),
                ("rejected_left_landing_surface_normal_y", "1"),
                ("rejected_right_selected_lower_margin", ".1")):
            rows = [runtime_row(0)]
            rows[0][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "canonical"):
                    check_rows(rows)

    def test_rejection_without_attempted_ik_requires_canonical_foot_fields(self):
        rows = rejection_rows("pose-certificate")
        rows[1]["rejected_left_target_x"] = "1"
        with self.assertRaisesRegex(ValueError, "canonical"):
            check_rows(rows, allow_ik=True)

    def test_pose_rejection_keeps_footprint_landing_without_attempted_ik(self):
        rows = rejection_rows("pose-certificate")
        rows[1].update({
            "rejected_left_landing_expected": "1",
            "rejected_left_landing_patch_ready": "0",
            "rejected_left_landing_sample": "1",
            "rejected_left_landing_center_x": "-.0123839173",
            "rejected_left_landing_center_y": ".0927663371",
            "rejected_left_landing_center_z": "-.0442949049",
            "rejected_left_landing_surface_status": "valid",
            "rejected_left_landing_surface_height": "0",
            "rejected_left_landing_surface_normal_x": "0",
            "rejected_left_landing_surface_normal_y": "1",
            "rejected_left_landing_surface_normal_z": "0",
            "rejected_left_landing_walkability_class": "1",
            "rejected_left_landing_patch_maximum_residual":
                ".089818030595779419",
        })
        self.assertEqual(check_rows(rows, allow_ik=True)["frames"], 2)

        changed = [dict(item) for item in rows]
        changed[1]["rejected_attempted_footprint_available"] = "0"
        with self.assertRaisesRegex(ValueError, "availability|footprint"):
            check_rows(changed, allow_ik=True)

        changed = [dict(item) for item in rows]
        changed[1]["rejected_left_target_x"] = ".01"
        with self.assertRaisesRegex(ValueError, "attempted IK|canonical"):
            check_rows(changed, allow_ik=True)

    def test_rejected_landing_does_not_use_republished_accepted_contact(self):
        rows = rejection_rows()
        rows[1].update({
            "frame_rejection_stage": "ik-candidate",
            "rejected_stop_reason": "no-swing-candidate",
            "left_recorded_contact": "1",
            "rejected_left_landing_expected": "1",
            "rejected_left_landing_patch_ready": "1",
            "rejected_left_landing_sample": "3",
            "rejected_left_landing_surface_status": "valid",
            "rejected_left_landing_walkability_class": "1",
            "rejected_left_landing_patch_maximum_residual": "0",
        })
        self.assertEqual(check_rows(rows, allow_ik=True)["frames"], 2)

        for name, value in (
                ("rejected_left_landing_sample", "0"),
                ("rejected_left_landing_surface_status", "outside"),
                ("rejected_left_landing_walkability_class", "0"),
                ("rejected_left_landing_patch_maximum_residual", ".006")):
            changed = [dict(item) for item in rows]
            changed[1][name] = value
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                        ValueError, "landing.*malformed|ready.*inconsistent"):
                    check_rows(changed, allow_ik=True)

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

    def test_gate_d_rejects_physical_penetration_even_with_traversal_reserve(self):
        rows = gate_d_rows()
        rows[50]["blocked_distance"] = ".04"
        rows[50]["rendered_min_clearance"] = "-.40"
        with self.assertRaisesRegex(ValueError, "physical clearance"):
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
        cases.append((rows, "distance"))
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

        unscheduled = check_rows([row(
            0, 21, matching_enabled=1, searched=0, transitioned=1,
            query_database_frame=10, query_range=0,
            selected_database_frame=20, source_range=1, range=1,
            incumbent_cost=2.0, selected_cost=1.0,
        )])
        self.assertEqual(unscheduled["transitions"], 1)

        for searched, transitioned in ((1, 0), (0, 1), (1, 1)):
            values = row(
                0, 21, matching_enabled=0,
                searched=searched, transitioned=transitioned,
                query_database_frame=10,
                selected_database_frame=(20 if transitioned else 10),
                source_range=(1 if transitioned else 0),
                range=(1 if transitioned else 0),
                incumbent_cost=2.0,
                selected_cost=(1.0 if transitioned else 2.0),
            )
            with self.subTest(
                    searched=searched, transitioned=transitioned):
                with self.assertRaisesRegex(
                        ValueError, "matching is disabled"):
                    check_rows([values])

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

        for selected in (
                1.0,
                float32_offset(1.0, 1),
                float32_offset(1.0, 4)):
            values = row(
                0, 21, matching_enabled=1, searched=0, transitioned=1,
                query_database_frame=10, query_range=0,
                selected_database_frame=20, source_range=1, range=1,
                incumbent_cost=1.0, selected_cost=selected,
            )
            with self.subTest(unscheduled_selected=selected):
                with self.assertRaisesRegex(
                        ValueError,
                        "unscheduled recovery did not strictly beat incumbent"):
                    check_rows([values])

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
