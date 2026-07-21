from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from jsonschema import Draft202012Validator
import numpy as np

import mm_sonic.metrics as metrics_module
from mm_sonic.artifacts import RunBundle
from mm_sonic.coordinator import (
    LOGGER_JOINT_PERMUTATION,
    DeliveryAuditEvidence,
    PreflightResult,
)
from mm_sonic.external import VerifiedGearCheckout
from mm_sonic.joints import ContractError
from mm_sonic.metrics import (
    PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256,
    ProductionDeliveryAuditor,
    SecondaryMetrics,
    aggregate_overall_hypothesis,
    aggregate_scene_hypothesis,
    attribute_failure,
    evaluate_dynamic_trial,
    evaluate_flat_trial,
    evaluate_terrain_trial,
    has_forbidden_contact,
    has_reference_penetration,
    minimum_local_pelvis_height,
    nearest_rank_summary,
    pelvis_up_dots,
    tracking_metrics,
    within_duration,
    within_known_good_ratio,
    within_target_radius,
)
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import encode_pose_v1


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64


def verdict_metrics() -> dict[str, object]:
    return {
        "joint_position_rmse_rad": 0.1,
        "pelvis_orientation_rms_rad": 0.01,
        "swing_foot_scuff_count": 0,
        "minimum_foot_clearance_m": 0.02,
        "horizontal_path_drift_m": 0.03,
        "joint_tracking_trace_rad": [0.1],
        "pelvis_tracking_trace_rad": [0.01],
        "contact_impulses_ns": [0.0],
        "policy_execution_timing_ns": {"p50": 1, "p95": 2, "p99": 3},
    }


def trial_verdict(**changes: object) -> dict[str, object]:
    document = {
        "schema": "mm-sonic-trial-verdict/v1",
        "stage": "C",
        "scene_id": "grail-curb-low",
        "terrain_weight": 4.0,
        "expected_frames": 21,
        "expected_sim_time_s": 0.4,
        "integration_pass": True,
        "kinematic_pass": True,
        "dynamic_pass": True,
        "failure_layer": None,
        "metrics": verdict_metrics(),
        "timings": {"generation": {"p50": 1, "p95": 2, "p99": 3}},
        "evidence_hashes": {"run": SHA_A},
    }
    document.update(changes)
    return document


def class_verdict(
    scene_id: str,
    *,
    aware_successes: int = 8,
    blind_successes: int = 5,
    aware_integration_failures: int = 0,
    blind_integration_failures: int = 0,
    kinematic_defects: int = 0,
    **changes: object,
) -> dict[str, object]:
    aggregate = aggregate_scene_hypothesis(aware_successes, blind_successes)
    document = {
        "scene_id": scene_id,
        "aware_successes": aware_successes,
        "blind_successes": blind_successes,
        "scored_trials_per_condition": 10,
        "aware_integration_failures": aware_integration_failures,
        "blind_integration_failures": blind_integration_failures,
        "kinematic_defects": kinematic_defects,
        "pass_margin": aggregate.pass_margin,
        "composition_verdict": (
            "supported" if aggregate.composition_feasible else "not_supported"
        ),
        "awareness_verdict": aggregate.awareness_effect,
        "evidence_hashes": {"class": SHA_A},
    }
    document.update(changes)
    return document


def hypothesis_verdict(
    terrain_classes: list[dict[str, object]],
    *,
    overall_hypothesis: str | None,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "schema": "mm-sonic-hypothesis-verdict/v1",
        "status": status,
        "terrain_classes": terrain_classes,
        "overall_hypothesis": overall_hypothesis,
        "evidence_hashes": {"aggregate": SHA_A},
    }


def schema_validator(name: str) -> Draft202012Validator:
    schema = json.loads((ROOT / "sonic/schemas" / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def canonical_buffer(count: int = 4) -> CanonicalTargetBuffer:
    position = (
        np.arange(count * 29, dtype=np.float32).reshape(count, 29)
        / np.float32(16.0)
    )
    velocity = -position.copy(order="C")
    yaw = np.linspace(0.0, 0.3, count, dtype=np.float32)
    quaternion = np.zeros((count, 4), dtype=np.float32)
    quaternion[:, 0] = np.cos(yaw / np.float32(2.0))
    quaternion[:, 3] = np.sin(yaw / np.float32(2.0))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return CanonicalTargetBuffer(
        joint_position=position,
        joint_velocity=velocity,
        body_quat_w=quaternion,
        frame_index=np.arange(count, dtype=np.int64),
    )


def slice_buffer(buffer: CanonicalTargetBuffer, begin: int, end: int) -> CanonicalTargetBuffer:
    return CanonicalTargetBuffer(
        joint_position=buffer.joint_position[begin:end].copy(order="C"),
        joint_velocity=buffer.joint_velocity[begin:end].copy(order="C"),
        body_quat_w=buffer.body_quat_w[begin:end].copy(order="C"),
        frame_index=buffer.frame_index[begin:end].copy(order="C"),
    )


def official_rows(buffer: CanonicalTargetBuffer, begin: int = 1) -> bytes:
    permutation = np.asarray(LOGGER_JOINT_PERMUTATION, dtype=np.int64)
    rows = []
    for index in range(begin, buffer.count):
        values = np.concatenate(
            (
                np.zeros(3, dtype=np.float32),
                buffer.body_quat_w[index],
                buffer.joint_position[index, permutation],
            )
        )
        rows.append(",".join(format(float(value), ".6g") for value in values) + ",\n")
    return "".join(rows).encode("ascii")


class DeliveryAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundles: list[RunBundle] = []
        self.run_number = 0

    def tearDown(self) -> None:
        for bundle in self.bundles:
            descriptor = getattr(bundle, "_directory_fd", -1)
            if descriptor >= 0:
                os.close(descriptor)
                bundle._directory_fd = -1
        self.temporary.cleanup()

    def verified(self, digest: str | None = PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256):
        hashes = {}
        if digest is not None:
            hashes["gear:current_frame_advancement_source"] = digest
        return VerifiedGearCheckout(
            gear_checkout=self.root,
            gear_commit="4a63412b034f1fef3e8e087adbb9be6cbeaaebdc",
            gear_dirty=False,
            hashes=MappingProxyType(hashes),
            known_good_reference=self.root,
        )

    def evidence(
        self,
        *,
        buffer: CanonicalTargetBuffer | None = None,
        official: bytes | None = None,
        split_timeline: bool = True,
        transmitted_buffer: CanonicalTargetBuffer | None = None,
        extra_timeline: CanonicalTargetBuffer | None = None,
        readiness_log_start_offset: int = 17,
        scoring_log_offset: int = 113,
    ) -> DeliveryAuditEvidence:
        canonical = canonical_buffer() if buffer is None else buffer
        transmitted = canonical if transmitted_buffer is None else transmitted_buffer
        self.run_number += 1
        bundle = RunBundle.create(self.root, "delivery", f"run-{self.run_number}")
        self.bundles.append(bundle)
        readiness = PreflightResult(
            session_id="session-a",
            readiness_log_start_offset=readiness_log_start_offset,
            scoring_log_offset=scoring_log_offset,
            readiness_attempts=1,
            official_log_path=str(self.root / "target.csv"),
            log_device=1,
            log_inode=2,
            expected_row_sha256=hashlib.sha256(official_rows(canonical, begin=0).splitlines(keepends=True)[0]).hexdigest(),
        )
        bundle.write_readiness(readiness)

        def archive(part: CanonicalTargetBuffer, *, phase: str, attempt=None):
            record = dict(
                bundle.archive_transmission(
                    encode_pose_v1(part),
                    first_frame_index=int(part.frame_index[0]),
                    last_frame_index=int(part.frame_index[-1]),
                    phase=phase,
                    attempt=attempt,
                )
            )
            record["local_send_completed"] = True
            return record

        readiness_publications = (archive(slice_buffer(transmitted, 0, 1), phase="readiness", attempt=1),)
        if split_timeline and transmitted.count > 2:
            timeline_publications = (
                archive(slice_buffer(transmitted, 1, transmitted.count - 1), phase="timeline"),
                archive(slice_buffer(transmitted, transmitted.count - 1, transmitted.count), phase="timeline"),
            )
        elif transmitted.count > 1:
            timeline_publications = (archive(slice_buffer(transmitted, 1, transmitted.count), phase="timeline"),)
        else:
            timeline_publications = ()
        if extra_timeline is not None:
            timeline_publications += (archive(extra_timeline, phase="timeline"),)
        return DeliveryAuditEvidence(
            session_id="session-a",
            canonical_buffer=canonical,
            readiness=readiness,
            official_log_slice=official_rows(canonical) if official is None else official,
            readiness_publications=readiness_publications,
            timeline_publications=timeline_publications,
            run=bundle,
        )

    def test_real_archived_pose_messages_and_official_rows_pass_exactly(self) -> None:
        evidence = self.evidence()
        audit = ProductionDeliveryAuditor(self.verified()).audit(evidence)
        self.assertTrue(audit.passed)
        self.assertEqual(audit.expected_rows, 4)
        self.assertEqual(audit.observed_rows, 4)
        self.assertTrue(audit.transmitted_indices_exact)
        self.assertTrue(audit.official_rows_exact)
        self.assertRegex(audit.evidence_sha256, r"^[0-9a-f]{64}$")

    def test_evidence_digest_binds_the_complete_readiness_boundary(self) -> None:
        first = ProductionDeliveryAuditor(self.verified()).audit(self.evidence())
        second = ProductionDeliveryAuditor(self.verified()).audit(
            self.evidence(
                readiness_log_start_offset=18,
                scoring_log_offset=114,
            )
        )
        self.assertTrue(first.passed)
        self.assertTrue(second.passed)
        self.assertNotEqual(first.evidence_sha256, second.evidence_sha256)

    def test_official_rows_fail_closed_on_missing_extra_reordered_or_unequal_data(self) -> None:
        canonical = canonical_buffer()
        rows = official_rows(canonical).splitlines(keepends=True)
        cases = {
            "missing": b"".join(rows[:-1]),
            "extra": b"".join(rows + [rows[-1]]),
            "reordered": b"".join((rows[1], rows[0], rows[2])),
            "unequal": rows[0].replace(b"0,", b"1,", 1) + b"".join(rows[1:]),
            "partial": official_rows(canonical)[:-1],
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                audit = ProductionDeliveryAuditor(self.verified()).audit(
                    self.evidence(buffer=canonical, official=value)
                )
                self.assertFalse(audit.passed)
                self.assertFalse(audit.official_rows_exact)

    def test_transmissions_fail_closed_on_missing_extra_reordered_or_unequal_rows(self) -> None:
        canonical = canonical_buffer()

        missing = self.evidence(buffer=canonical)
        missing = replace(missing, timeline_publications=missing.timeline_publications[:-1])

        extra_frame = CanonicalTargetBuffer(
            joint_position=canonical.joint_position[-1:].copy(),
            joint_velocity=canonical.joint_velocity[-1:].copy(),
            body_quat_w=canonical.body_quat_w[-1:].copy(),
            frame_index=np.array([4], dtype=np.int64),
        )
        extra = self.evidence(buffer=canonical, extra_timeline=extra_frame)

        reordered = self.evidence(buffer=canonical)
        reordered = replace(
            reordered,
            timeline_publications=tuple(reversed(reordered.timeline_publications)),
        )

        changed = canonical_buffer()
        changed_position = changed.joint_position.copy()
        changed_position[2, 0] += np.float32(1.0)
        changed = CanonicalTargetBuffer(
            joint_position=changed_position,
            joint_velocity=changed.joint_velocity,
            body_quat_w=changed.body_quat_w,
            frame_index=changed.frame_index,
        )
        unequal = self.evidence(buffer=canonical, transmitted_buffer=changed)

        for name, evidence in (
            ("missing", missing),
            ("extra", extra),
            ("reordered", reordered),
            ("unequal", unequal),
        ):
            with self.subTest(name=name):
                audit = ProductionDeliveryAuditor(self.verified()).audit(evidence)
                self.assertFalse(audit.passed)
                self.assertFalse(audit.transmitted_indices_exact)

    def test_session_canonical_and_readiness_evidence_are_bound_to_durable_records(self) -> None:
        base = self.evidence()
        wrong_readiness = replace(base.readiness, session_id="session-b")
        wrong_session = replace(base, session_id="session-b", readiness=wrong_readiness)
        wrong_offsets = replace(
            base,
            readiness=replace(base.readiness, scoring_log_offset=114),
        )
        changed = canonical_buffer()
        changed_position = changed.joint_position.copy()
        changed_position[3, 0] += np.float32(2.0)
        wrong_canonical = replace(
            base,
            canonical_buffer=CanonicalTargetBuffer(
                joint_position=changed_position,
                joint_velocity=changed.joint_velocity,
                body_quat_w=changed.body_quat_w,
                frame_index=changed.frame_index,
            ),
        )
        for name, evidence in (
            ("session", wrong_session),
            ("offsets", wrong_offsets),
            ("canonical", wrong_canonical),
        ):
            with self.subTest(name=name):
                audit = ProductionDeliveryAuditor(self.verified()).audit(evidence)
                self.assertFalse(audit.passed)

    def test_run_path_replacement_and_noncanonical_sidecars_fail_closed(self) -> None:
        sidecar_evidence = self.evidence()
        sidecar_path = (
            sidecar_evidence.run.path
            / sidecar_evidence.timeline_publications[0]["digest_path"]
        )
        sidecar_path.write_bytes(sidecar_path.read_bytes() + b"appended-data\n")
        sidecar_audit = ProductionDeliveryAuditor(self.verified()).audit(
            sidecar_evidence
        )
        self.assertFalse(sidecar_audit.passed)
        self.assertFalse(sidecar_audit.transmitted_indices_exact)

        replaced_evidence = self.evidence()
        original_path = replaced_evidence.run.path
        retained_path = original_path.with_name(original_path.name + "-retained")
        original_path.rename(retained_path)
        original_path.symlink_to(retained_path, target_is_directory=True)
        replaced_audit = ProductionDeliveryAuditor(self.verified()).audit(
            replaced_evidence
        )
        self.assertFalse(replaced_audit.passed)
        self.assertFalse(replaced_audit.transmitted_indices_exact)

    def test_durable_readiness_rejects_duplicate_keys_and_invalid_envelope(self) -> None:
        duplicate = self.evidence()
        duplicate_path = duplicate.run.path / "readiness.jsonl"
        duplicate_path.write_bytes(
            duplicate_path.read_bytes().replace(
                b'"kind":"readiness"',
                b'"kind":"readiness","kind":"readiness"',
            )
        )
        duplicate_audit = ProductionDeliveryAuditor(self.verified()).audit(duplicate)
        self.assertFalse(duplicate_audit.passed)
        self.assertFalse(duplicate_audit.official_rows_exact)

        invalid_envelope = self.evidence()
        envelope_path = invalid_envelope.run.path / "readiness.jsonl"
        envelope_path.write_bytes(
            envelope_path.read_bytes().replace(b'"sequence":1', b'"sequence":"1"')
        )
        envelope_audit = ProductionDeliveryAuditor(self.verified()).audit(
            invalid_envelope
        )
        self.assertFalse(envelope_audit.passed)
        self.assertFalse(envelope_audit.official_rows_exact)

    def test_missing_or_mismatched_current_frame_advancement_hash_is_rejected(self) -> None:
        for digest in (None, "0" * 64):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(ContractError, "CurrentFrameAdvancement"):
                    ProductionDeliveryAuditor(self.verified(digest))


class TrackingMetricTests(unittest.TestCase):
    def test_tracking_requires_aligned_unique_exact_50hz_rows(self) -> None:
        target_joint = np.zeros((2, 29), dtype=np.float64)
        actual_joint = np.ones((2, 29), dtype=np.float64)
        target_quat = np.array(((1.0, 0.0, 0.0, 0.0),) * 2)
        actual_quat = np.array(
            (
                (math.cos(math.pi / 6), 0.0, 0.0, math.sin(math.pi / 6)),
                (math.cos(math.pi / 3), 0.0, 0.0, math.sin(math.pi / 3)),
            )
        )
        result = tracking_metrics(
            np.array([0, 1]),
            target_joint,
            target_quat,
            np.array([0, 1]),
            actual_joint,
            actual_quat,
        )
        self.assertEqual(result.joint_position_rmse_rad, 1.0)
        self.assertEqual(result.joint_tracking_trace_rad, (1.0, 1.0))
        self.assertEqual(
            result.pelvis_tracking_trace_rad,
            (
                2.0 * math.acos(abs(math.cos(math.pi / 6))),
                2.0 * math.acos(abs(math.cos(math.pi / 3))),
            ),
        )
        self.assertAlmostEqual(
            result.pelvis_orientation_rms_rad,
            math.sqrt(((math.pi / 3) ** 2 + (2 * math.pi / 3) ** 2) / 2),
        )

        for target_index, actual_index in (
            (np.array([0]), np.array([0, 1])),
            (np.array([0, 0]), np.array([0, 1])),
            (np.array([0, 1]), np.array([1, 0])),
        ):
            with self.subTest(target=target_index, actual=actual_index):
                with self.assertRaisesRegex(ContractError, "frame"):
                    tracking_metrics(
                        target_index,
                        target_joint[: len(target_index)],
                        target_quat[: len(target_index)],
                        actual_index,
                        actual_joint[: len(actual_index)],
                        actual_quat[: len(actual_index)],
                    )

    def test_quaternion_sign_is_equivalent_and_nonunit_values_are_rejected(self) -> None:
        target = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = tracking_metrics(
            np.array([0]),
            np.zeros((1, 29)),
            target,
            np.array([0]),
            np.zeros((1, 29)),
            -target,
        )
        self.assertEqual(result.pelvis_orientation_rms_rad, 0.0)
        with self.assertRaisesRegex(ContractError, "unit quaternion"):
            tracking_metrics(
                np.array([0]),
                np.zeros((1, 29)),
                target,
                np.array([0]),
                np.zeros((1, 29)),
                np.array([[2.0, 0.0, 0.0, 0.0]]),
            )

    def test_local_clearance_up_dot_and_contact_classification(self) -> None:
        pelvis = np.array(((0.0, 0.0, 0.9), (1.0, 2.0, 0.7)))
        terrain = np.array((0.2, 0.3))
        self.assertAlmostEqual(minimum_local_pelvis_height(pelvis, terrain), 0.4)
        quaternions = np.array(
            (
                (1.0, 0.0, 0.0, 0.0),
                (math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0),
            )
        )
        np.testing.assert_allclose(pelvis_up_dots(quaternions), np.array([1.0, 0.5]))
        self.assertFalse(has_forbidden_contact(("left_foot", "right_foot")))
        self.assertTrue(has_forbidden_contact(("left_foot", "knees")))


class ThresholdAndVerdictTests(unittest.TestCase):
    def test_registered_thresholds_are_inclusive_only_at_the_boundary(self) -> None:
        self.assertTrue(within_target_radius(0.25))
        self.assertFalse(within_target_radius(math.nextafter(0.25, math.inf)))
        self.assertTrue(within_duration(12.5, 10.0))
        self.assertFalse(within_duration(math.nextafter(12.5, math.inf), 10.0))
        self.assertFalse(has_reference_penetration(("knees",), (0.005,)))
        self.assertTrue(
            has_reference_penetration(
                ("knees",), (math.nextafter(0.005, math.inf),)
            )
        )
        self.assertTrue(within_known_good_ratio(1.5, 1.0))
        self.assertFalse(within_known_good_ratio(math.nextafter(1.5, math.inf), 1.0))
        self.assertTrue(within_known_good_ratio(1.0e-8, 0.0))
        self.assertFalse(within_known_good_ratio(math.nextafter(1.0e-8, math.inf), 0.0))
        self.assertTrue(within_known_good_ratio(1.0e-8, 1.0e-8))
        self.assertFalse(
            within_known_good_ratio(
                math.nextafter(1.0e-8, math.inf),
                1.0e-8,
            )
        )

    def test_nearest_rank_timing_quantiles_are_deterministic(self) -> None:
        self.assertEqual(
            nearest_rank_summary((9, 1, 5, 3, 7)),
            {"p50": 5, "p95": 9, "p99": 9},
        )
        self.assertEqual(
            nearest_rank_summary(tuple(range(1, 101))),
            {"p50": 50, "p95": 95, "p99": 99},
        )

    def test_dynamic_trial_boundaries_and_secondary_outputs_do_not_override_success(self) -> None:
        secondary = SecondaryMetrics(
            swing_foot_scuff_count=999,
            minimum_foot_clearance_m=-1.0,
            horizontal_path_drift_m=100.0,
            joint_tracking_trace_rad=(10.0,),
            pelvis_tracking_trace_rad=(10.0,),
            contact_impulses_ns=(1000.0,),
            policy_execution_timing_ns=None,
        )
        passing = evaluate_terrain_trial(
            integration_pass=True,
            exact_frame_coverage=True,
            target_distance_m=0.25,
            reached_time_s=12.5,
            nominal_duration_s=10.0,
            minimum_pelvis_local_height_m=0.45,
            minimum_pelvis_up_dot=0.5,
            contact_groups=("left_foot",),
            secondary=secondary,
        )
        self.assertTrue(passing.dynamic_pass)
        self.assertIs(passing.secondary, secondary)
        failed = evaluate_terrain_trial(
            integration_pass=True,
            exact_frame_coverage=False,
            target_distance_m=0.0,
            reached_time_s=0.0,
            nominal_duration_s=10.0,
            minimum_pelvis_local_height_m=1.0,
            minimum_pelvis_up_dot=1.0,
            contact_groups=(),
            secondary=secondary,
        )
        self.assertFalse(failed.dynamic_pass)
        self.assertEqual(
            evaluate_dynamic_trial(
                integration_pass=True,
                exact_frame_coverage=True,
                target_distance_m=0.25,
                reached_time_s=12.5,
                nominal_duration_s=10.0,
                minimum_pelvis_local_height_m=0.45,
                minimum_pelvis_up_dot=0.5,
                contact_groups=("left_foot",),
                secondary=secondary,
            ),
            passing,
        )

    def test_flat_trial_enforces_every_stage_b_gate(self) -> None:
        secondary = SecondaryMetrics(
            swing_foot_scuff_count=0,
            minimum_foot_clearance_m=0.01,
            horizontal_path_drift_m=0.0,
            joint_tracking_trace_rad=(0.0,),
            pelvis_tracking_trace_rad=(0.0,),
            contact_impulses_ns=(),
            policy_execution_timing_ns=None,
        )
        fields = {
            "integration_pass": True,
            "exact_command_coverage": True,
            "exact_frame_coverage": True,
            "exact_control_duration": True,
            "exact_safety_log_coverage": True,
            "minimum_pelvis_local_height_m": 0.45,
            "minimum_pelvis_up_dot": 0.5,
            "contact_groups": ("left_foot",),
            "joint_position_rmse_rad": 1.5,
            "known_good_joint_position_rmse_rad": 1.0,
            "pelvis_orientation_rms_rad": 0.3,
            "known_good_pelvis_orientation_rms_rad": 0.2,
            "secondary": secondary,
        }
        passing = evaluate_flat_trial(**fields)
        self.assertTrue(passing.dynamic_pass)
        self.assertEqual(
            set(passing.checks),
            {
                "integration",
                "exact_command_coverage",
                "exact_frame_coverage",
                "exact_control_duration",
                "exact_safety_log_coverage",
                "pelvis_local_height",
                "pelvis_up_dot",
                "forbidden_contacts",
                "joint_tracking_ratio",
                "pelvis_tracking_ratio",
            },
        )
        for name, replacement in (
            ("integration_pass", False),
            ("exact_command_coverage", False),
            ("exact_frame_coverage", False),
            ("exact_control_duration", False),
            ("exact_safety_log_coverage", False),
            (
                "joint_position_rmse_rad",
                math.nextafter(1.5, math.inf),
            ),
            (
                "pelvis_orientation_rms_rad",
                math.nextafter(1.5 * 0.2, math.inf),
            ),
            ("minimum_pelvis_local_height_m", math.nextafter(0.45, -math.inf)),
            ("minimum_pelvis_up_dot", math.nextafter(0.5, -math.inf)),
            ("contact_groups", ("left_foot", "pelvis")),
        ):
            with self.subTest(name=name):
                changed = dict(fields)
                changed[name] = replacement
                self.assertFalse(evaluate_flat_trial(**changed).dynamic_pass)

    def test_secondary_metrics_reject_impossible_negative_outputs(self) -> None:
        fields = {
            "swing_foot_scuff_count": 0,
            "minimum_foot_clearance_m": -0.01,
            "horizontal_path_drift_m": 0.0,
            "joint_tracking_trace_rad": (0.0,),
            "pelvis_tracking_trace_rad": (0.0,),
            "contact_impulses_ns": (),
            "policy_execution_timing_ns": {"p50": 1, "p95": 2, "p99": 3},
        }
        for name, replacement in (
            ("horizontal_path_drift_m", -1.0e-9),
            ("joint_tracking_trace_rad", (-1.0e-9,)),
            ("pelvis_tracking_trace_rad", (-1.0e-9,)),
            ("contact_impulses_ns", (-1.0e-9,)),
            ("joint_tracking_trace_rad", ()),
            ("pelvis_tracking_trace_rad", ()),
            ("policy_execution_timing_ns", {"p50": 2, "p95": 1, "p99": 3}),
        ):
            with self.subTest(name=name, replacement=replacement):
                changed = dict(fields)
                changed[name] = replacement
                with self.assertRaises(ContractError):
                    SecondaryMetrics(**changed)

    def test_hypothesis_requires_eight_of_ten_and_aware_margin_three(self) -> None:
        supported = aggregate_scene_hypothesis(8, 5)
        self.assertTrue(supported.composition_feasible)
        self.assertEqual(supported.awareness_effect, "supported")
        self.assertTrue(supported.hypothesis_supported)

        too_few = aggregate_scene_hypothesis(7, 4)
        self.assertFalse(too_few.hypothesis_supported)
        margin_two = aggregate_scene_hypothesis(9, 7)
        self.assertFalse(margin_two.hypothesis_supported)
        self.assertEqual(margin_two.awareness_effect, "not_supported")
        both_pass = aggregate_scene_hypothesis(10, 8)
        self.assertTrue(both_pass.composition_feasible)
        self.assertEqual(both_pass.awareness_effect, "inconclusive")
        self.assertFalse(both_pass.hypothesis_supported)

    def test_overall_hypothesis_requires_all_three_registered_classes(self) -> None:
        supported = aggregate_scene_hypothesis(8, 5)
        both_pass = aggregate_scene_hypothesis(10, 8)
        unsupported = aggregate_scene_hypothesis(7, 4)
        registered = {
            "grail-curb-low": supported,
            "ramp-10-up-down": supported,
            "stairs-shallow": supported,
        }
        overall = aggregate_overall_hypothesis(registered)
        self.assertTrue(overall.composition_feasible)
        self.assertEqual(overall.awareness_effect, "supported")
        self.assertTrue(overall.hypothesis_supported)

        inconclusive = dict(registered)
        inconclusive["stairs-shallow"] = both_pass
        result = aggregate_overall_hypothesis(inconclusive)
        self.assertTrue(result.composition_feasible)
        self.assertEqual(result.awareness_effect, "inconclusive")
        self.assertFalse(result.hypothesis_supported)

        failed = dict(registered)
        failed["ramp-10-up-down"] = unsupported
        result = aggregate_overall_hypothesis(failed)
        self.assertFalse(result.composition_feasible)
        self.assertEqual(result.awareness_effect, "not_supported")

        for invalid in (
            {key: value for key, value in registered.items() if key != "stairs-shallow"},
            {**registered, "unregistered": supported},
        ):
            with self.subTest(classes=tuple(invalid)):
                with self.assertRaisesRegex(ContractError, "three registered"):
                    aggregate_overall_hypothesis(invalid)

    def test_failure_attribution_matches_approved_layers_and_integration_precedes_all(self) -> None:
        self.assertEqual(
            attribute_failure(
                stage="C",
                integration_pass=False,
                known_good_file_pass=False,
                known_good_stream_pass=False,
                kinematic_pass=False,
                dynamic_pass=False,
            ),
            "integration",
        )
        self.assertEqual(
            attribute_failure(
                stage="A",
                integration_pass=True,
                known_good_file_pass=True,
                known_good_stream_pass=False,
                kinematic_pass=True,
                dynamic_pass=False,
            ),
            "bridge_protocol",
        )
        self.assertEqual(
            attribute_failure(
                stage="B",
                integration_pass=True,
                known_good_file_pass=True,
                known_good_stream_pass=True,
                kinematic_pass=False,
                dynamic_pass=False,
            ),
            "joint_or_coordinate_conversion",
        )
        self.assertEqual(
            attribute_failure(
                stage="B",
                integration_pass=True,
                known_good_file_pass=True,
                known_good_stream_pass=True,
                kinematic_pass=True,
                dynamic_pass=False,
            ),
            "reference_distribution_or_sonic_tracking",
        )
        self.assertEqual(
            attribute_failure(
                stage="C",
                integration_pass=True,
                known_good_file_pass=True,
                known_good_stream_pass=True,
                kinematic_pass=False,
                dynamic_pass=True,
            ),
            "motion_matching_reference",
        )
        self.assertEqual(
            attribute_failure(
                stage="C",
                integration_pass=True,
                known_good_file_pass=True,
                known_good_stream_pass=True,
                kinematic_pass=True,
                dynamic_pass=False,
            ),
            "low_level_tracking_contact_domain_mismatch",
        )


class VerdictSchemaTests(unittest.TestCase):
    def test_draft_2020_12_validator_is_an_exact_test_dependency(self) -> None:
        pyproject = (ROOT / "sonic/pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'test = [\n  "jsonschema==4.25.1",\n]',
            pyproject,
        )

    def test_verdict_schemas_are_strict_draft_2020_12_contracts(self) -> None:
        for name, schema_name in (
            ("trial_verdict_v1.schema.json", "mm-sonic-trial-verdict/v1"),
            ("hypothesis_verdict_v1.schema.json", "mm-sonic-hypothesis-verdict/v1"),
        ):
            with self.subTest(name=name):
                document = json.loads((ROOT / "sonic/schemas" / name).read_text())
                self.assertEqual(document["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(document["properties"]["schema"]["const"], schema_name)
                self.assertFalse(document["additionalProperties"])
                self.assertIn("schema", document["required"])

    def test_verdict_schemas_require_complete_metrics_and_evidence(self) -> None:
        trial = json.loads(
            (ROOT / "sonic/schemas/trial_verdict_v1.schema.json").read_text()
        )
        metrics = trial["properties"]["metrics"]
        self.assertEqual(
            set(metrics["required"]),
            {
                "joint_position_rmse_rad",
                "pelvis_orientation_rms_rad",
                "swing_foot_scuff_count",
                "minimum_foot_clearance_m",
                "horizontal_path_drift_m",
                "joint_tracking_trace_rad",
                "pelvis_tracking_trace_rad",
                "contact_impulses_ns",
                "policy_execution_timing_ns",
            },
        )
        self.assertEqual(trial["properties"]["evidence_hashes"]["minProperties"], 1)
        self.assertEqual(metrics["properties"]["joint_tracking_trace_rad"]["minItems"], 1)
        self.assertEqual(metrics["properties"]["pelvis_tracking_trace_rad"]["minItems"], 1)
        state_constraints = trial["allOf"]
        self.assertEqual(
            state_constraints[0]["if"]["properties"]["dynamic_pass"]["const"],
            True,
        )
        self.assertEqual(
            state_constraints[0]["then"]["properties"]["integration_pass"]["const"],
            True,
        )
        self.assertEqual(
            state_constraints[1]["then"]["properties"]["failure_layer"]["const"],
            None,
        )

        hypothesis = json.loads(
            (ROOT / "sonic/schemas/hypothesis_verdict_v1.schema.json").read_text()
        )
        self.assertEqual(
            hypothesis["properties"]["evidence_hashes"]["minProperties"], 1
        )
        class_schema = hypothesis["$defs"]["classVerdict"]
        self.assertEqual(
            class_schema["properties"]["evidence_hashes"]["minProperties"], 1
        )
        complete = hypothesis["allOf"][0]
        self.assertEqual(
            complete["if"]["properties"]["status"]["const"],
            "complete",
        )
        complete_classes = complete["then"]["properties"]["terrain_classes"]
        self.assertEqual(complete_classes["minItems"], 3)
        self.assertEqual(complete_classes["maxItems"], 3)
        self.assertEqual(len(complete_classes["allOf"]), 3)

    def test_real_trial_instances_enforce_terminal_integration_attribution(
        self,
    ) -> None:
        validator = schema_validator("trial_verdict_v1.schema.json")
        valid_failure = trial_verdict(
            integration_pass=False,
            kinematic_pass=False,
            dynamic_pass=False,
            failure_layer="integration",
        )
        self.assertTrue(validator.is_valid(trial_verdict()))
        self.assertTrue(validator.is_valid(valid_failure))
        for wrong_layer in (None, "bridge_protocol"):
            contradictory = trial_verdict(
                integration_pass=False,
                kinematic_pass=False,
                dynamic_pass=False,
                failure_layer=wrong_layer,
            )
            with self.subTest(wrong_layer=wrong_layer):
                self.assertFalse(
                    validator.is_valid(contradictory),
                    "terminal integration failure accepted wrong attribution",
                )

        self.assertTrue(
            hasattr(metrics_module, "validate_trial_verdict_semantics"),
            "missing runtime trial semantic validator",
        )
        metrics_module.validate_trial_verdict_semantics(trial_verdict())
        metrics_module.validate_trial_verdict_semantics(valid_failure)
        for wrong_layer in (None, "bridge_protocol"):
            with self.subTest(runtime_wrong_layer=wrong_layer):
                with self.assertRaisesRegex(ContractError, "integration"):
                    metrics_module.validate_trial_verdict_semantics(
                        trial_verdict(
                            integration_pass=False,
                            kinematic_pass=False,
                            dynamic_pass=False,
                            failure_layer=wrong_layer,
                        )
                    )

        valid_stage_a = trial_verdict(
            stage="A",
            kinematic_pass=False,
            dynamic_pass=True,
            failure_layer=None,
        )
        unavailable_stage_a_evidence = trial_verdict(
            stage="A",
            kinematic_pass=False,
            dynamic_pass=True,
            failure_layer="bridge_protocol",
        )
        contradictory_stage_c = trial_verdict(
            stage="C",
            kinematic_pass=False,
            dynamic_pass=True,
            failure_layer="low_level_tracking_contact_domain_mismatch",
        )
        self.assertTrue(validator.is_valid(valid_stage_a))
        self.assertTrue(
            validator.is_valid(unavailable_stage_a_evidence),
            "schema inferred an unavailable Stage A bridge result",
        )
        self.assertFalse(validator.is_valid(contradictory_stage_c))
        metrics_module.validate_trial_verdict_semantics(valid_stage_a)
        metrics_module.validate_trial_verdict_semantics(
            unavailable_stage_a_evidence
        )
        with self.assertRaisesRegex(ContractError, "failure|pass"):
            metrics_module.validate_trial_verdict_semantics(
                contradictory_stage_c
            )

    def test_real_hypothesis_instances_cannot_contradict_aggregation(self) -> None:
        validator = schema_validator("hypothesis_verdict_v1.schema.json")
        unsupported_classes = [
            class_verdict(scene_id, aware_successes=7, blind_successes=4)
            for scene_id in (
                "grail-curb-low",
                "ramp-10-up-down",
                "stairs-shallow",
            )
        ]
        valid = hypothesis_verdict(
            unsupported_classes,
            overall_hypothesis="not_supported",
        )
        contradictory_overall = hypothesis_verdict(
            unsupported_classes,
            overall_hypothesis="supported",
        )
        self.assertTrue(validator.is_valid(valid))
        self.assertFalse(
            validator.is_valid(contradictory_overall),
            "three not-supported classes accepted an overall supported claim",
        )

        contradictory_class = hypothesis_verdict(
            [
                class_verdict(
                    scene_id,
                    composition_verdict="not_supported",
                    awareness_verdict="not_supported",
                )
                for scene_id in (
                    "grail-curb-low",
                    "ramp-10-up-down",
                    "stairs-shallow",
                )
            ],
            overall_hypothesis="not_supported",
        )
        self.assertFalse(
            validator.is_valid(contradictory_class),
            "per-class verdict accepted counts that imply support",
        )

        supported = hypothesis_verdict(
            [
                class_verdict(scene_id)
                for scene_id in (
                    "grail-curb-low",
                    "ramp-10-up-down",
                    "stairs-shallow",
                )
            ],
            overall_hypothesis="supported",
        )
        inconclusive = hypothesis_verdict(
            [
                class_verdict(scene_id, aware_successes=10, blind_successes=8)
                for scene_id in (
                    "grail-curb-low",
                    "ramp-10-up-down",
                    "stairs-shallow",
                )
            ],
            overall_hypothesis="inconclusive",
        )
        incomplete = hypothesis_verdict(
            [],
            status="incomplete",
            overall_hypothesis=None,
        )
        for representative in (supported, inconclusive, incomplete):
            self.assertTrue(validator.is_valid(representative))
            metrics_module.validate_hypothesis_verdict_semantics(representative)

        self.assertTrue(
            hasattr(metrics_module, "validate_hypothesis_verdict_semantics"),
            "missing runtime hypothesis semantic validator",
        )
        metrics_module.validate_hypothesis_verdict_semantics(valid)
        contradictions = (
            ("overall", contradictory_overall),
            ("composition", contradictory_class),
            (
                "pass_margin",
                hypothesis_verdict(
                    [
                        class_verdict(
                            scene_id,
                            aware_successes=8,
                            blind_successes=5,
                            pass_margin=2,
                        )
                        for scene_id in (
                            "grail-curb-low",
                            "ramp-10-up-down",
                            "stairs-shallow",
                        )
                    ],
                    overall_hypothesis="supported",
                ),
            ),
            (
                "counts",
                hypothesis_verdict(
                    [
                        class_verdict(
                            scene_id,
                            aware_successes=8,
                            blind_successes=5,
                            aware_integration_failures=3,
                        )
                        for scene_id in (
                            "grail-curb-low",
                            "ramp-10-up-down",
                            "stairs-shallow",
                        )
                    ],
                    overall_hypothesis="supported",
                ),
            ),
        )
        for expected, document in contradictions:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    metrics_module.validate_hypothesis_verdict_semantics(document)

    def test_not_run_forbids_results_but_incomplete_retains_partial_evidence(
        self,
    ) -> None:
        validator = schema_validator("hypothesis_verdict_v1.schema.json")
        partial_class = class_verdict("grail-curb-low")
        valid_not_run = hypothesis_verdict(
            [],
            status="not_run",
            overall_hypothesis=None,
        )
        incomplete = hypothesis_verdict(
            [partial_class],
            status="incomplete",
            overall_hypothesis=None,
        )
        not_run_with_class = hypothesis_verdict(
            [partial_class],
            status="not_run",
            overall_hypothesis=None,
        )
        not_run_with_claim = hypothesis_verdict(
            [],
            status="not_run",
            overall_hypothesis="supported",
        )

        for valid in (valid_not_run, incomplete):
            self.assertTrue(validator.is_valid(valid))
            metrics_module.validate_hypothesis_verdict_semantics(valid)
        for invalid in (not_run_with_class, not_run_with_claim):
            self.assertFalse(validator.is_valid(invalid))
            with self.assertRaisesRegex(ContractError, "not_run"):
                metrics_module.validate_hypothesis_verdict_semantics(invalid)


if __name__ == "__main__":
    unittest.main()
