from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from mm_sonic.full_walking_terrain_lmm_contracts import RangeRecord
from mm_sonic.full_walking_terrain_lmm_pfnn import (
    EXPECTED_EXCLUDED_COUNTS,
    PFNN_INVENTORY_SHA256,
    PFNN_PIPELINE_VERSION,
    admitted_pfnn_ranges,
    authenticate_pfnn_runtime,
    bind_pfnn_source_frame_count,
    classify_pfnn_terrain,
    discover_full_pfnn_inventory,
    excluded_pfnn_blocks,
    fit_pfnn_range_terrain,
    pfnn_identity_stage_name,
    pfnn_worker_start_method,
    range_local_pfnn_contacts,
    resolve_pfnn_work_root,
    retarget_subprocess_argv,
    reverse_mirror_terrain_lanes,
    safe_absolute_60hz_rows,
    summarize_pfnn_gait,
)

PFNN_ROOT = Path("/home/ubuntu/datasets/pfnn/pfnn")
GMR_ROOT = Path("/home/ubuntu/.cache/native-g1-pfnn/GMR")
RETARGET_ROOT = Path("/home/ubuntu/.cache/native-g1-pfnn/retargeting_project")
G1_XML = Path("/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml")
INVENTORY = Path(
    "/home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/"
    "g1-full-walking-terrain-lmm/inventory-v2.json"
)


def test_pipeline_version_invalidates_pre_contact_endpoint_stages() -> None:
    assert PFNN_PIPELINE_VERSION == "g1-full-walking-pfnn-pipeline/v3"


def test_retarget_runs_in_the_pinned_mink_environment() -> None:
    assert pfnn_worker_start_method() == "spawn"
    argv = retarget_subprocess_argv(
        source=Path("/source.bvh"),
        gmr_root=Path("/gmr"),
        retarget_root=Path("/retarget"),
        output=Path("/motion.npz"),
    )
    assert argv[0] == "/home/ubuntu/.cache/native-g1-pfnn/venv/bin/python"
    assert argv[1:3] == ("-m", "mm_sonic.retarget_pfnn_bvh_g1")
    assert argv[-2:] == ("--grounding", "source")
    authority = authenticate_pfnn_runtime(
        gmr_root=GMR_ROOT,
        retarget_root=RETARGET_ROOT,
        g1_xml=G1_XML,
    )
    assert authority == {
        "gmr_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
        "retarget_project_commit": "fb3433a6310ab4198102d3905e74b73944fc1f6b",
        "g1_xml_sha256": (
            "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
        ),
    }


def test_explicit_work_root_reuses_authenticated_stages(tmp_path: Path) -> None:
    output = tmp_path / "pfnn-60hz-v2"
    expected_default = tmp_path / ".pfnn-60hz-v2.work"
    assert resolve_pfnn_work_root(output=output) == expected_default
    existing = tmp_path / ".pfnn-v1.work"
    existing.mkdir()
    assert resolve_pfnn_work_root(output=output, work_root=existing) == existing
    link = tmp_path / "linked-work"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        resolve_pfnn_work_root(output=output, work_root=link)


def test_source_frame_count_binds_and_bounds_native_maps() -> None:
    record = RangeRecord(
        range_id="range",
        canonical_source_id="pfnn:source",
        terrain_id="pfnn-terrain:source",
        mirror_of=None,
        family="flat",
        split_group_id="group",
        split="train",
        start=0,
        stop=2,
        quality="usable",
        authority={"kind": "pfnn"},
    )
    receipt = {
        "schema": "g1-full-walking-pfnn-source/v1",
        "status": "accepted",
        "source_id": "pfnn:source",
        "native_rows": 6,
        "ranges": 1,
    }
    bound = bind_pfnn_source_frame_count(
        (record,),
        np.array([2, 4], dtype=np.int32),
        np.array([2, 5], dtype=np.int32),
        (receipt,),
        expected_source_id="pfnn:source",
    )
    assert bound[0].authority["source_frame_count"] == 6
    with pytest.raises(ValueError, match="source provenance"):
        bind_pfnn_source_frame_count(
            (record,),
            np.array([2, 4], dtype=np.int32),
            np.array([2, 6], dtype=np.int32),
            (receipt,),
            expected_source_id="pfnn:source",
        )


def _source_paths() -> tuple[Path, ...]:
    tree = ast.parse((PFNN_ROOT / "generate_database.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "data_terrain"
            for target in statement.targets
        ):
            values = ast.literal_eval(statement.value)
            return tuple(PFNN_ROOT / value for value in values)
    raise AssertionError("authenticated PFNN table has no data_terrain")


def test_discovers_exact_authenticated_sources_mirrors_and_sidecars() -> None:
    discovery = discover_full_pfnn_inventory(
        INVENTORY,
        pfnn_root=PFNN_ROOT,
        expected_inventory_sha256=PFNN_INVENTORY_SHA256,
    )
    assert len(discovery.sources) == 80
    assert sum(source.mirror_of is not None for source in discovery.sources) == 40
    assert discovery.rest_pose.path == "data/animations/rest.bvh"
    assert discovery.rest_pose.size_bytes == 5_461
    by_id = {source.source_id: source for source in discovery.sources}
    for source in discovery.sources:
        inputs = source.authority["inputs"]
        assert set(inputs) == {"motion", "gait", "phase", "footsteps"}
        if source.mirror_of is not None:
            assert source.mirror_of in by_id
            assert source.canonical_source_id == by_id[source.mirror_of].source_id
            assert (
                inputs["motion"]["sha256"]
                != by_id[source.mirror_of].authority["inputs"]["motion"]["sha256"]
            )
        stage_name = pfnn_identity_stage_name(source)
        assert stage_name == pfnn_identity_stage_name(source)
        assert "__stage-" in stage_name


def test_classification_matches_released_flat_rocky_beam_and_jumpy_table() -> None:
    assert classify_pfnn_terrain("LocomotionFlat12_000_mirror") == "jumpy"
    assert classify_pfnn_terrain("NewCaptures01_000") == "flat"
    assert classify_pfnn_terrain("NewCaptures03_002_mirror") == "jumpy"
    assert classify_pfnn_terrain("WalkingUpSteps06_000") == "beam"
    assert classify_pfnn_terrain("WalkingUpSteps10_000_mirror") == "flat"
    assert classify_pfnn_terrain("LocomotionFlat04_000") == "flat"
    assert classify_pfnn_terrain("WalkingUpSteps04_001") == "rocky"


def test_full_gait_admission_freezes_counts_runs_and_absolute_60hz_clock() -> None:
    native_rows = 0
    output_rows = 0
    run_count = 0
    exclusions: Counter[str] = Counter()
    for source in _source_paths():
        gait = np.loadtxt(source.with_suffix(".gait"), dtype=np.float64)
        summary = summarize_pfnn_gait(gait)
        exclusions.update(summary.excluded_counts)
        family = classify_pfnn_terrain(source.stem)
        ranges = admitted_pfnn_ranges(
            gait,
            np.zeros(len(gait) - 1, dtype=np.bool_),
            ((0, len(gait), family),),
        )
        native_rows += sum(stop - start for start, stop, _ in ranges)
        run_count += len(ranges)
        for start, stop, _ in ranges:
            rows, left, right, alpha = safe_absolute_60hz_rows(
                len(gait), ((start, stop),)
            )
            assert np.all(left >= start)
            assert np.all(right < stop)
            assert np.all(rows % 2 == 0)
            assert np.all(alpha == 0.0)
            output_rows += len(rows)
    assert native_rows == 493_214
    assert run_count == 458
    assert output_rows == 246_630
    assert dict(exclusions) == EXPECTED_EXCLUDED_COUNTS


def test_gait_rule_is_strict_and_boundaries_have_zero_halo() -> None:
    gait = np.zeros((8, 8), dtype=np.float64)
    gait[0, 0] = 1.0
    gait[1, 1] = 1.0
    gait[2, [0, 2]] = 0.5  # exact allowed/disallowed tie is admitted
    gait[3, 2] = 1.0
    gait[4, 1] = 1.0
    gait[4, 7] = 1.0  # bump is an independent overlay
    gait[5, 3] = 1.0
    gait[6, 0] = 1.0
    gait[7, 0] = 1.0
    ranges = admitted_pfnn_ranges(
        gait,
        np.array([False, True, False, False, False, False, False]),
        ((0, 7, "flat"), (7, 8, "rocky")),
    )
    assert ranges == (
        (0, 2, "flat"),
        (2, 3, "flat"),
        (4, 5, "flat"),
        (6, 7, "flat"),
        (7, 8, "rocky"),
    )
    summary = summarize_pfnn_gait(gait)
    assert summary.excluded_counts == {"jog": 1, "run": 1}
    assert summary.admitted_rows == 6
    assert excluded_pfnn_blocks(gait) == ((3, 4, "jog"), (5, 6, "run"))
    malformed = gait.copy()
    malformed[0, 0] = np.nan
    assert admitted_pfnn_ranges(malformed, np.zeros(7, bool), ((0, 8, "flat"),)) == (
        (1, 3, "flat"),
        (4, 5, "flat"),
        (6, 8, "flat"),
    )
    assert summarize_pfnn_gait(malformed).malformed_rows == 1
    assert excluded_pfnn_blocks(malformed)[0] == (0, 1, "malformed")
    malformed = gait.copy()
    malformed[0, :7] = 0.0
    assert admitted_pfnn_ranges(malformed, np.zeros(7, bool), ((0, 8, "flat"),)) == (
        (1, 3, "flat"),
        (4, 5, "flat"),
        (6, 8, "flat"),
    )


def test_safe_absolute_clock_never_restarts_at_an_odd_range_start() -> None:
    rows, left, right, alpha = safe_absolute_60hz_rows(12, ((3, 10),))
    np.testing.assert_array_equal(rows, [4, 6, 8])
    np.testing.assert_array_equal(left, rows)
    np.testing.assert_array_equal(right, rows)
    np.testing.assert_array_equal(alpha, 0.0)
    short = safe_absolute_60hz_rows(12, ((3, 10),), minimum_rows=4)
    assert all(values.size == 0 for values in short)


def test_pfnn_fit_contacts_do_not_read_past_the_safe_range() -> None:
    positions = np.zeros((5, 11, 3), dtype=np.float64)
    positions[3:, 4, 0] = 1.0
    local = range_local_pfnn_contacts(positions, np.array([1, 2], np.int32))
    assert local.shape == (2, 4)
    assert local[-1, 0]


def test_mirror_reverses_only_right_and_left_terrain_lanes() -> None:
    grid = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
    mirrored = reverse_mirror_terrain_lanes(grid)
    np.testing.assert_array_equal(mirrored[..., 0], grid[..., 2])
    np.testing.assert_array_equal(mirrored[..., 1], grid[..., 1])
    np.testing.assert_array_equal(mirrored[..., 2], grid[..., 0])
    np.testing.assert_array_equal(
        grid, np.arange(72, dtype=np.float32).reshape(2, 12, 3)
    )


def test_family_fitter_is_deterministic_and_preserves_exact_flat_modes() -> None:
    frames = 12
    positions = np.zeros((frames, 31, 3), dtype=np.float64)
    positions[:, :, 0] = np.linspace(-1.0, 1.0, frames)[:, None]
    positions[:, [4, 9], 2] = -0.1
    positions[:, [5, 10], 2] = 0.1
    positions[:, [4, 5, 9, 10], 1] = np.array([5.0, 4.0, 5.0, 4.0])
    contacts = np.ones((frames, 4), dtype=np.bool_)
    patches = np.stack((np.zeros((4, 4)), np.arange(4)[:, None] * np.ones((1, 4))))
    coords = np.zeros((2, 4), dtype=np.float64)
    flat = fit_pfnn_range_terrain(
        family="flat",
        global_positions=positions,
        contacts=contacts,
        patches=patches,
        patch_coords=coords,
        source_start=20,
        source_sha256="a" * 64,
        patches_sha256="b" * 64,
    )
    assert flat.mode == "flat-constant"
    assert flat.selected_patch_index == -1
    assert flat.height(np.array([[0.0, 0.0]])).item() == pytest.approx(0.0)
    no_contact = fit_pfnn_range_terrain(
        family="flat",
        global_positions=positions,
        contacts=np.zeros_like(contacts),
        patches=patches,
        patch_coords=coords,
        source_start=20,
        source_sha256="a" * 64,
        patches_sha256="b" * 64,
    )
    assert no_contact.mode == "no-contact-zero"
    assert no_contact.height(np.array([[1.0, 2.0]])).item() == 0.0
    for family in ("rocky", "jumpy", "beam"):
        first = fit_pfnn_range_terrain(
            family=family,
            global_positions=positions,
            contacts=contacts,
            patches=patches,
            patch_coords=coords,
            source_start=20,
            source_sha256="a" * 64,
            patches_sha256="b" * 64,
        )
        second = fit_pfnn_range_terrain(
            family=family,
            global_positions=positions,
            contacts=contacts,
            patches=patches,
            patch_coords=coords,
            source_start=20,
            source_sha256="a" * 64,
            patches_sha256="b" * 64,
        )
        assert first.fit_id == second.fit_id
        np.testing.assert_array_equal(
            first.height(np.array([[0.0, 0.0], [1.0, 1.0]])),
            second.height(np.array([[0.0, 0.0], [1.0, 1.0]])),
        )
    standing = positions.copy()
    standing[:, :, (0, 2)] = 0.0
    degenerate = fit_pfnn_range_terrain(
        family="rocky",
        global_positions=standing,
        contacts=contacts,
        patches=patches,
        patch_coords=coords,
        source_start=20,
        source_sha256="c" * 64,
        patches_sha256="b" * 64,
    )
    assert np.isfinite(degenerate.height(np.array([[0.0, 0.0]]))).all()
    long_positions = np.tile(positions, (9, 1, 1))
    long_contacts = np.ones((len(long_positions), 4), dtype=np.bool_)
    bounded = fit_pfnn_range_terrain(
        family="rocky",
        global_positions=long_positions,
        contacts=long_contacts,
        patches=patches,
        patch_coords=coords,
        source_start=20,
        source_sha256="d" * 64,
        patches_sha256="b" * 64,
    )
    assert bounded.receipt["stance_probe_count"] == 432
    assert bounded.receipt["objective_stance_probe_count"] == 256
    assert bounded.receipt["rbf_probe_count"] == 256
