from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.build_g1_pfnn_vertical_dataset import (
    VerticalSliceSource,
    VerticalSurfaceSegment,
    build_vertical_dataset,
    build_vertical_dataset_from_retarget,
    load_vertical_dataset,
    save_vertical_dataset,
)
from mm_sonic.pfnn_terrain_fit import PFNNTerrainFit
from mm_sonic.terrain_pfnn.layout import OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.phase import ContactPhaseTrack
from mm_sonic.terrain_pfnn.sources import PFNNSourceClip


def _clip(stem: str, *, joint_offset: float = 0.0) -> PFNNSourceClip:
    frames = 100
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 0] = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    root[:, 2] = 0.8
    root_quaternion = np.zeros((frames, 4), dtype=np.float32)
    root_quaternion[:, 0] = 1.0
    joint = np.full((frames, 29), joint_offset, dtype=np.float32)
    body = np.repeat(root[:, None, :], 30, axis=1)
    body[:, :, 2] += np.linspace(0.0, 0.4, 30, dtype=np.float32)
    body_quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quaternion[:, :, 0] = 1.0
    linear = np.zeros_like(body)
    linear[:, :, 0] = 30.0 / 99.0
    root_linear = np.zeros_like(root)
    root_linear[:, 0] = 30.0 / 99.0
    return PFNNSourceClip(
        clip_id=stem,
        terrain_id="pfnn_vertical",
        fps=30.0,
        root_position_world=root,
        root_quaternion_world_wxyz=root_quaternion,
        joint_position=joint,
        body_position_world=body,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=linear,
        body_angular_velocity_world=np.zeros_like(body),
        root_linear_velocity_world=root_linear,
        root_angular_velocity_world=np.zeros_like(root),
        joint_velocity=np.zeros_like(joint),
        terrain_path=None,
        terrain_position_world=np.zeros(3),
        terrain_quaternion_world_from_usd_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        motion_sha256=("a" if stem.endswith("train") else "b") * 64,
        terrain_sha256=None,
        source_license_id="PFNN-academic-noncommercial",
    )


def _phase(frames: int = 100) -> ContactPhaseTrack:
    phase = np.remainder(np.arange(frames) * (2.0 * np.pi / 30.0), 2.0 * np.pi)
    advance = np.full(frames, 2.0 * np.pi / 30.0, dtype=np.float32)
    advance[-1] = 0.0
    contact = np.zeros((frames, 4), dtype=np.bool_)
    contact[: frames // 2, :2] = True
    contact[frames // 2 :, 2:] = True
    return ContactPhaseTrack(
        contact=contact,
        confidence=contact.astype(np.float32),
        phase=phase.astype(np.float32),
        phase_advance=advance,
        valid=np.ones(frames, dtype=np.bool_),
    )


def _surface(points: np.ndarray) -> np.ndarray:
    return np.zeros(np.asarray(points).shape[:-1], dtype=np.float64)


def _source(role: str, stem: str, *, joint_offset: float = 0.0) -> VerticalSliceSource:
    return VerticalSliceSource(
        role=role,
        stem=stem,
        source_start_frame_120hz=400,
        clip=_clip(stem, joint_offset=joint_offset),
        phase_track=_phase(),
        segments=(
            VerticalSurfaceSegment(
                cycle_start_frame_120hz=520,
                cycle_stop_frame_120hz=664,
                terrain_sha256=("c" if role == "train" else "d") * 64,
                height_at=_surface,
            ),
        ),
        selection_sha256="e" * 64,
        retarget_manifest_sha256="f" * 64,
    )


class BuildG1PFNNVerticalDatasetTest(unittest.TestCase):
    def test_cycle_rows_stay_inside_surface_and_train_is_mirrored_once(self) -> None:
        dataset = build_vertical_dataset(
            (_source("train", "released_train"), _source("validation", "released_val"))
        )
        train = dataset.splits["train"]
        validation = dataset.splits["validation"]
        self.assertGreater(len(train.phase), 0)
        self.assertEqual(train.sequence_lane.shape, train.phase.shape)
        self.assertTrue(np.all(train.sequence_lane == "motion"))
        self.assertEqual(np.count_nonzero(train.mirrored), len(train.phase) // 2)
        self.assertFalse(np.any(validation.mirrored))
        self.assertTrue(np.all((train.center_frame_120hz >= 520) & (train.center_frame_120hz < 664)))
        self.assertTrue(np.all((validation.center_frame_120hz >= 520) & (validation.center_frame_120hz < 664)))
        contact = OUTPUT_LAYOUT["contact_logit"]
        np.testing.assert_array_equal(dataset.y_mean[contact], 0.0)
        np.testing.assert_array_equal(dataset.y_std[contact], 1.0)

    def test_is_deterministic_and_normalization_is_train_only(self) -> None:
        train = _source("train", "released_train")
        validation = _source("validation", "released_val")
        first = build_vertical_dataset((train, validation))
        reversed_result = build_vertical_dataset((validation, train))
        changed_validation = build_vertical_dataset(
            (train, _source("validation", "released_val", joint_offset=1.0))
        )
        self.assertEqual(first.dataset_sha256, reversed_result.dataset_sha256)
        np.testing.assert_array_equal(first.x_mean, changed_validation.x_mean)
        np.testing.assert_array_equal(first.y_std, changed_validation.y_std)

    def test_round_trips_safe_arrays_and_rejects_duplicate_or_split_overlap(self) -> None:
        sources = (_source("train", "released_train"), _source("validation", "released_val"))
        dataset = build_vertical_dataset(sources)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            save_vertical_dataset(root, dataset)
            loaded = load_vertical_dataset(root)
            self.assertEqual(loaded.dataset_sha256, dataset.dataset_sha256)
            np.testing.assert_array_equal(loaded.splits["train"].x, dataset.splits["train"].x)
        with self.assertRaisesRegex(ValueError, "duplicate PFNN source"):
            build_vertical_dataset((sources[0], sources[0]))
        with self.assertRaisesRegex(ValueError, "train/validation source overlap"):
            build_vertical_dataset((sources[0], replace(sources[0], role="validation")))

    def test_real_orchestrator_fits_only_terrain_cycles_and_writes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pfnn_root = root / "pfnn"
            animations = pfnn_root / "data" / "animations"
            animations.mkdir(parents=True)
            retarget_root = root / "retarget"
            (retarget_root / "retargets").mkdir(parents=True)
            items = []
            definitions = (
                ("flat", "train", ["idle_transition", "straight"]),
                ("turn", "train", ["left_turn", "right_turn"]),
                ("steps_train", "train", ["ascent", "descent"]),
                ("steps_val", "validation", ["ascent", "descent"]),
            )
            selection_items = []
            for index, (stem, role, coverage) in enumerate(definitions):
                bvh = animations / f"{stem}.bvh"
                bvh.write_text("fixture", encoding="utf-8")
                phase = np.remainder(np.arange(1200) / 120.0, 1.0)
                np.savetxt(animations / f"{stem}.phase", phase, fmt="%.9f")
                (animations / f"{stem}.gait").write_text("fixture", encoding="utf-8")
                (animations / f"{stem}_footsteps.txt").write_text(
                    "500 550\n600 650\n700 750\n", encoding="utf-8"
                )
                relative = Path("retargets") / f"{stem}.npz"
                motion = retarget_root / relative
                motion.write_bytes(f"motion-{stem}".encode())
                receipt = motion.with_suffix(".receipt.json")
                receipt.write_text("{}", encoding="utf-8")
                items.append(
                    {
                        "stem": stem,
                        "role": role,
                        "start_frame_120hz": 400,
                        "stop_frame_120hz": 800,
                        "coverage": coverage,
                        "output": relative.as_posix(),
                        "output_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
                        "receipt": receipt.relative_to(retarget_root).as_posix(),
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    }
                )
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                selection_items.append(
                    {
                        "stem": stem,
                        "role": role,
                        "start_frame_120hz": 400,
                        "stop_frame_120hz": 800,
                        "coverage": coverage,
                        "bvh_sha256": digest(bvh),
                        "phase_sha256": digest(animations / f"{stem}.phase"),
                        "gait_sha256": digest(animations / f"{stem}.gait"),
                        "footsteps_sha256": digest(animations / f"{stem}_footsteps.txt"),
                    }
                )
            selection_payload = {
                "schema": "g1-pfnn-vertical-slice-selection/v1",
                "items": sorted(selection_items, key=lambda value: (value["role"], value["stem"])),
            }
            encoded = json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
            selection_payload["sha256"] = hashlib.sha256(encoded).hexdigest()
            (retarget_root / "source-selection.json").write_text(
                json.dumps(selection_payload), encoding="utf-8"
            )
            retarget_manifest = {
                "schema": "g1-pfnn-vertical-slice-retarget/v1",
                "status": "accepted",
                "selection_sha256": selection_payload["sha256"],
                "items": items,
            }
            (retarget_root / "retarget-manifest.json").write_text(
                json.dumps(retarget_manifest), encoding="utf-8"
            )
            patches = pfnn_root / "patches.npz"
            np.savez(patches, X=np.zeros((1, 2, 2)), C=np.zeros((1, 4)))
            fit_calls = []

            def load_one(path, fk, **arguments):
                del path, fk, arguments
                return _clip("released_train")

            def contacts_one(*, display_start_frame, display_frame_count, **arguments):
                del display_start_frame, arguments
                contacts = np.zeros((display_frame_count, 4), dtype=np.bool_)
                contacts[:, :2] = True
                return contacts

            def fit_one(**arguments):
                fit_calls.append(
                    (
                        arguments["cycle_start"],
                        arguments["cycle_stop"],
                        arguments["display_start"],
                        arguments["display_count"],
                    )
                )
                count = arguments["display_count"]
                return PFNNTerrainFit(
                    patch=np.zeros((2, 2)),
                    patch_coord=np.zeros(4),
                    contact_center_xz=np.zeros(2),
                    patch_height_mean=0.0,
                    stance_height_mean=0.0,
                    rbf_centers_xz=np.array([[-1.0, 0.0], [1.0, 0.0]]),
                    rbf_epsilon=np.ones(2),
                    rbf_weights=np.zeros((1, 2)),
                    source_contacts=np.ones((count, 4), dtype=np.bool_),
                    source_start_frame=arguments["display_start"],
                    source_frame_count=count,
                    cycle_start_frame=arguments["cycle_start"],
                    cycle_stop_frame=arguments["cycle_stop"],
                    selected_patch_index=0,
                    fitting_error=0.0,
                    source_sha256=hashlib.sha256(Path(arguments["source"]).read_bytes()).hexdigest(),
                    patches_sha256=hashlib.sha256(Path(arguments["patches_path"]).read_bytes()).hexdigest(),
                )

            output = root / "output"
            result = build_vertical_dataset_from_retarget(
                retarget_root=retarget_root,
                pfnn_root=pfnn_root,
                patches_path=patches,
                model_path=root / "unused.xml",
                output=output,
                fk=object(),
                load_one=load_one,
                contacts_one=contacts_one,
                fit_one=fit_one,
            )
            self.assertTrue((output / "dataset" / "manifest.json").is_file())
            self.assertEqual(result.dataset_sha256, load_vertical_dataset(output / "dataset").dataset_sha256)
            self.assertEqual(
                fit_calls,
                [
                    (500, 600, 520, 80),
                    (600, 700, 600, 80),
                    (500, 600, 520, 80),
                    (600, 700, 600, 80),
                ],
            )
            self.assertEqual(len(tuple((output / "terrain").glob("*.npz"))), 4)


if __name__ == "__main__":
    unittest.main()
