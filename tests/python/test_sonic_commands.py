from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields as dataclass_fields
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import mm_sonic.commands as commands_module
from mm_sonic.commands import (
    CommandSample,
    PERTURBATIONS,
    RouteDefinition,
    command_script_bytes,
    command_script_sha256,
    compile_route_commands,
    flat_command_script,
    freeze_command_script,
    load_registered_route,
    route_frame_schedule,
)
from mm_sonic.coordinator import CommandSample as CoordinatorCommandSample
from mm_sonic.joints import ContractError


ROOT = Path(__file__).resolve().parents[2]
ROUTE_CLI = ROOT / "sonic" / "build" / "route_schedule_cli"
TERRAIN_ROOT = Path(
    os.environ.get(
        "SONIC_TERRAIN_DIR",
        "/home/ubuntu/projects/motion-matching/resources/g1_terrain",
    )
)
ROUTES = (
    ("grail-curb-low", "curb-forward"),
    ("ramp-10-up-down", "up-landing-down"),
    ("stairs-shallow", "ascent-landing-descent"),
)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class CommandValueTests(unittest.TestCase):
    def test_commands_module_owns_the_only_public_command_type(self) -> None:
        self.assertIs(CoordinatorCommandSample, CommandSample)
        command = CommandSample(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(FrozenInstanceError):
            command.chunk_index = 1

        for values, expected in (
            ((-1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), "chunk_index"),
            ((0, (0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), "velocity"),
            ((0, (0.0, 0.0, math.nan), (1.0, 0.0, 0.0, 0.0)), "finite"),
            ((0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)), "unit"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    CommandSample(*values)

    def test_flat_script_has_the_exact_registered_boundaries(self) -> None:
        commands = flat_command_script()
        self.assertEqual(len(commands), 30)
        self.assertEqual([item.chunk_index for item in commands], list(range(30)))

        for index, command in enumerate(commands):
            if index < 5:
                yaw_degrees = 0.0
                speed = 0.0
            elif index < 15:
                yaw_degrees = 0.0
                speed = 0.5
            elif index < 25:
                yaw_degrees = 4.5 * (index - 14)
                speed = 0.5
            else:
                yaw_degrees = 45.0
                speed = 0.0
            yaw = math.radians(yaw_degrees)
            expected_velocity = (
                speed * math.cos(yaw),
                speed * math.sin(yaw),
                0.0,
            )
            expected_heading = (
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            )
            with self.subTest(index=index):
                self.assertEqual(command.requested_velocity_mujoco, expected_velocity)
                self.assertEqual(command.desired_heading_mujoco_wxyz, expected_heading)

        self.assertEqual(commands[15].desired_heading_mujoco_wxyz[3], math.sin(math.radians(2.25)))
        self.assertEqual(commands[24].desired_heading_mujoco_wxyz[3], math.sin(math.radians(22.5)))
        self.assertEqual(commands[25].requested_velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(
            commands[25].desired_heading_mujoco_wxyz,
            commands[24].desired_heading_mujoco_wxyz,
        )

    def test_flat_command_json_is_complete_canonical_and_hash_stable(self) -> None:
        commands = flat_command_script()
        encoded = command_script_bytes(
            scene_id="sonic-flat-baseline",
            route_id="flat-12s",
            commands=commands,
        )
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertNotIn(b" ", encoded)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema"], "mm-sonic-command-script/v1")
        self.assertEqual(decoded["chunk_intervals"], 10)
        self.assertEqual(decoded["source_rate_hz"], 25)
        self.assertEqual(decoded["target_rate_hz"], 50)
        self.assertEqual(decoded["chunk_count"], 30)
        self.assertEqual(decoded["duration_s"], 12.0)
        self.assertEqual(len(decoded["commands"]), 30)
        self.assertEqual(
            command_script_sha256(
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                commands=commands,
            ),
            "8137b0a6d951da2a5ead427698479d490150e963c5a458cad11a766968de6d85",
        )
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), command_script_sha256(
            scene_id="sonic-flat-baseline", route_id="flat-12s", commands=commands
        ))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.json"
            first = freeze_command_script(
                path,
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                commands=commands,
            )
            second = freeze_command_script(
                path,
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                commands=commands,
            )
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), encoded)
            metadata = path.stat()
            self.assertTrue(path.is_file())
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(metadata.st_mode & 0o222, 0)
            changed = list(commands)
            changed[0] = CommandSample(0, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            with self.assertRaisesRegex(ContractError, "frozen command"):
                freeze_command_script(
                    path,
                    scene_id="sonic-flat-baseline",
                    route_id="flat-12s",
                    commands=changed,
                )

    def test_frozen_command_path_rejects_symlink_and_hardlink_aliases(self) -> None:
        commands = flat_command_script()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_bytes(
                command_script_bytes(
                    scene_id="sonic-flat-baseline",
                    route_id="flat-12s",
                    commands=commands,
                )
            )
            for name, make_alias in (
                ("symlink", lambda path: path.symlink_to(source)),
                ("hardlink", lambda path: os.link(source, path)),
            ):
                alias = root / f"{name}.json"
                make_alias(alias)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ContractError, "regular single-link"):
                        freeze_command_script(
                            alias,
                            scene_id="sonic-flat-baseline",
                            route_id="flat-12s",
                            commands=commands,
                        )

            real_parent = root / "real/subdirectory"
            real_parent.mkdir(parents=True)
            (root / "ancestor-alias").symlink_to(root / "real", target_is_directory=True)
            with self.assertRaisesRegex(ContractError, "symlink-free directory"):
                freeze_command_script(
                    root / "ancestor-alias/subdirectory/commands.json",
                    scene_id="sonic-flat-baseline",
                    route_id="flat-12s",
                    commands=commands,
                )

    def test_registered_perturbations_are_exact_and_physical_only(self) -> None:
        self.assertEqual(
            PERTURBATIONS,
            (
                ("lateral_p003", 0.03, 0.0),
                ("lateral_m003", -0.03, 0.0),
                ("lateral_p006", 0.06, 0.0),
                ("lateral_m006", -0.06, 0.0),
                ("yaw_p002", 0.0, math.radians(2.0)),
                ("yaw_m002", 0.0, math.radians(-2.0)),
                ("yaw_p004", 0.0, math.radians(4.0)),
                ("yaw_m004", 0.0, math.radians(-4.0)),
                ("combined_p", 0.03, math.radians(2.0)),
                ("combined_m", -0.03, math.radians(-2.0)),
            ),
        )

    def test_stage_c_trial_expansion_freezes_every_scientific_identity(self) -> None:
        self.assertTrue(
            hasattr(commands_module, "expand_stage_c_scored_trials"),
            "missing immutable Stage C trial expander",
        )
        trials = commands_module.expand_stage_c_scored_trials(
            scene_id="grail-curb-low",
            route_id="curb-forward",
            command_artifact_id="commands/grail-curb-low.json",
            command_sha256="a" * 64,
            duration_s=4.8,
            mm_initial_state_sha256="b" * 64,
            mm_reference_sha256_by_condition={
                "aware": "c" * 64,
                "blind": "d" * 64,
            },
        )
        self.assertEqual(len(trials), 20)
        self.assertTrue(
            all(
                isinstance(trial, commands_module.StageCTrialIdentity)
                for trial in trials
            )
        )
        with self.assertRaises(FrozenInstanceError):
            trials[0].condition = "blind"

        expected_pairs = {
            (condition, perturbation_id)
            for condition in ("aware", "blind")
            for perturbation_id, _lateral, _yaw in PERTURBATIONS
        }
        self.assertEqual(
            {(trial.condition, trial.perturbation_id) for trial in trials},
            expected_pairs,
        )
        global_fields = (
            "scene_id",
            "route_id",
            "command_artifact_id",
            "command_sha256",
            "duration_s",
            "mm_initial_state_sha256",
        )
        for name in global_fields:
            self.assertEqual({getattr(trial, name) for trial in trials}, {
                getattr(trials[0], name)
            })
        expected_weights = {"aware": 4.0, "blind": 0.0}
        expected_references = {"aware": "c" * 64, "blind": "d" * 64}
        expected_perturbations = {
            perturbation_id: (lateral, yaw)
            for perturbation_id, lateral, yaw in PERTURBATIONS
        }
        for condition in expected_weights:
            condition_trials = tuple(
                trial for trial in trials if trial.condition == condition
            )
            self.assertEqual(len(condition_trials), 10)
            self.assertEqual(
                {trial.terrain_weight for trial in condition_trials},
                {expected_weights[condition]},
            )
            self.assertEqual(
                {trial.mm_reference_sha256 for trial in condition_trials},
                {expected_references[condition]},
            )
            for trial in condition_trials:
                self.assertEqual(
                    (
                        trial.physical_lateral_offset_m,
                        trial.physical_yaw_offset_rad,
                    ),
                    expected_perturbations[trial.perturbation_id],
                )

        self.assertEqual(
            tuple(field.name for field in dataclass_fields(trials[0])),
            (
                "scene_id",
                "route_id",
                "condition",
                "terrain_weight",
                "perturbation_id",
                "physical_lateral_offset_m",
                "physical_yaw_offset_rad",
                "command_artifact_id",
                "command_sha256",
                "duration_s",
                "mm_initial_state_sha256",
                "mm_reference_sha256",
            ),
        )

    def test_stage_c_trial_validator_rejects_every_identity_confound(self) -> None:
        self.assertTrue(
            hasattr(commands_module, "validate_stage_c_scored_trials"),
            "missing Stage C trial invariance validator",
        )
        trials = commands_module.expand_stage_c_scored_trials(
            scene_id="ramp-10-up-down",
            route_id="up-landing-down",
            command_artifact_id="commands/ramp.json",
            command_sha256="1" * 64,
            duration_s=12.0,
            mm_initial_state_sha256="2" * 64,
            mm_reference_sha256_by_condition={
                "aware": "3" * 64,
                "blind": "4" * 64,
            },
        )
        self.assertEqual(
            commands_module.validate_stage_c_scored_trials(trials),
            trials,
        )
        mutations = (
            ("missing trial", trials[:-1]),
            ("duplicate trial", trials[:-1] + (trials[0],)),
            (
                "command artifact",
                (
                    replace(
                        trials[0],
                        command_artifact_id="commands/replacement.json",
                    ),
                )
                + trials[1:],
            ),
            (
                "command_sha256",
                (replace(trials[0], command_sha256="5" * 64),) + trials[1:],
            ),
            (
                "duration_s",
                (replace(trials[0], duration_s=13.0),) + trials[1:],
            ),
            (
                "initial state",
                (
                    replace(trials[0], mm_initial_state_sha256="6" * 64),
                )
                + trials[1:],
            ),
            (
                "reference",
                (replace(trials[0], mm_reference_sha256="7" * 64),)
                + trials[1:],
            ),
        )
        for expected, changed in mutations:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    commands_module.validate_stage_c_scored_trials(changed)

        with self.assertRaisesRegex(ContractError, "terrain weight"):
            replace(trials[0], terrain_weight=0.0)
        with self.assertRaisesRegex(ContractError, "physical perturbation"):
            replace(trials[0], physical_lateral_offset_m=0.031)

    def test_direct_route_definitions_enforce_the_compiler_invariants(self) -> None:
        valid = {
            "scene_id": "scene",
            "route_id": "route",
            "waypoints_xz_holden": ((0.0, 0.0), (1.0, 0.0)),
            "landing_hold_seconds": 0.0,
        }
        self.assertEqual(
            len(compile_route_commands(RouteDefinition(**valid))),
            5,
        )
        cases = (
            ("scene_id", {**valid, "scene_id": ""}),
            ("route_id", {**valid, "route_id": ""}),
            (
                "immutable waypoint",
                {**valid, "waypoints_xz_holden": [[0.0, 0.0], [1.0, 0.0]]},
            ),
            (
                "at least two",
                {**valid, "waypoints_xz_holden": ((0.0, 0.0),)},
            ),
            (
                "finite binary32",
                {**valid, "waypoints_xz_holden": ((0.0, 0.0), (math.nan, 0.0))},
            ),
            (
                "degenerate",
                {**valid, "waypoints_xz_holden": ((0.0, 0.0), (0.0, 0.0))},
            ),
            (
                "landing hold",
                {**valid, "landing_hold_seconds": -1.0},
            ),
            (
                "landing hold",
                {**valid, "landing_hold_seconds": 1.0},
            ),
        )
        for expected, fields in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    RouteDefinition(**fields)


@unittest.skipUnless(TERRAIN_ROOT.is_dir(), "registered terrain directory unavailable")
class RouteParityTests(unittest.TestCase):
    def cpp_schedule(self, scene_path: Path, route_id: str) -> dict[str, object]:
        completed = subprocess.run(
            (str(ROUTE_CLI), str(scene_path), route_id),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stderr, "")
        return json.loads(completed.stdout)

    def test_python_port_matches_authoritative_cpp_for_every_route_frame(self) -> None:
        self.assertTrue(ROUTE_CLI.is_file(), f"missing route CLI: {ROUTE_CLI}")
        for scene_id, route_id in ROUTES:
            scene_path = TERRAIN_ROOT / "scenes" / scene_id / "scene.json"
            route = load_registered_route(scene_path, route_id)
            python_frames = route_frame_schedule(route)
            cpp = self.cpp_schedule(scene_path, route_id)
            with self.subTest(scene_id=scene_id, route_id=route_id):
                self.assertEqual(cpp["schema"], "mm-sonic-route-schedule/v1")
                self.assertEqual(cpp["scene_id"], scene_id)
                self.assertEqual(cpp["route_id"], route_id)
                self.assertEqual(cpp["source_rate_hz"], 25)
                self.assertEqual(cpp["motion_frames"], len(python_frames))
                self.assertEqual(len(cpp["frames"]), len(python_frames))
                for expected, observed in zip(python_frames, cpp["frames"], strict=True):
                    self.assertEqual(observed["frame_index"], expected.frame_index)
                    self.assertEqual(observed["waypoint_index"], expected.waypoint_index)
                    self.assertFalse(observed["complete"])
                    self.assertEqual(
                        observed["velocity_holden_bits"],
                        [f32_bits(value) for value in expected.velocity_holden],
                    )

    def test_chunk_means_preserve_route_displacement_and_only_final_block_is_padded(self) -> None:
        for scene_id, route_id in ROUTES:
            scene_path = TERRAIN_ROOT / "scenes" / scene_id / "scene.json"
            route = load_registered_route(scene_path, route_id)
            frames = route_frame_schedule(route)
            commands = compile_route_commands(route)
            with self.subTest(scene_id=scene_id, route_id=route_id):
                self.assertEqual(len(commands), math.ceil(len(frames) / 10))
                self.assertEqual([item.chunk_index for item in commands], list(range(len(commands))))
                frame_dx = sum(item.velocity_holden[0] for item in frames) / 25.0
                frame_dz = sum(item.velocity_holden[2] for item in frames) / 25.0
                chunk_dx = sum(item.requested_velocity_mujoco[0] for item in commands) * 0.4
                chunk_dz = -sum(item.requested_velocity_mujoco[1] for item in commands) * 0.4
                self.assertAlmostEqual(chunk_dx, frame_dx, places=6)
                self.assertAlmostEqual(chunk_dz, frame_dz, places=6)
                if len(frames) % 10:
                    final_source = frames[-(len(frames) % 10):]
                    expected_x = sum(item.velocity_holden[0] for item in final_source) / 10.0
                    expected_z = sum(item.velocity_holden[2] for item in final_source) / 10.0
                    self.assertAlmostEqual(commands[-1].requested_velocity_mujoco[0], expected_x)
                    self.assertAlmostEqual(commands[-1].requested_velocity_mujoco[1], -expected_z)
                for previous, command in zip(commands, commands[1:]):
                    speed = math.hypot(*command.requested_velocity_mujoco[:2])
                    if speed == 0.0:
                        self.assertEqual(
                            command.desired_heading_mujoco_wxyz,
                            previous.desired_heading_mujoco_wxyz,
                        )

    def test_route_sources_reject_symlink_and_hardlink_aliases(self) -> None:
        registered = TERRAIN_ROOT / "scenes/grail-curb-low/scene.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_bytes(registered.read_bytes())
            for name, make_alias in (
                ("symlink", lambda path: path.symlink_to(source)),
                ("hardlink", lambda path: os.link(source, path)),
            ):
                alias = root / f"{name}.json"
                make_alias(alias)
                with self.subTest(name=name, implementation="python"):
                    with self.assertRaisesRegex(ContractError, "regular single-link"):
                        load_registered_route(alias, "curb-forward")
                with self.subTest(name=name, implementation="cpp"):
                    completed = subprocess.run(
                        (str(ROUTE_CLI), str(alias), "curb-forward"),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("regular single-link", completed.stderr)

            nested = root / "real/nested"
            nested.mkdir(parents=True)
            nested_source = nested / "scene.json"
            nested_source.write_bytes(registered.read_bytes())
            (root / "ancestor-alias").symlink_to(root / "real", target_is_directory=True)
            aliased_source = root / "ancestor-alias/nested/scene.json"
            with self.assertRaisesRegex(ContractError, "symlink-free directory"):
                load_registered_route(aliased_source, "curb-forward")
            completed = subprocess.run(
                (str(ROUTE_CLI), str(aliased_source), "curb-forward"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("regular single-link", completed.stderr)


class RegistryTests(unittest.TestCase):
    def test_committed_experiment_registries_have_the_frozen_values(self) -> None:
        flat = json.loads((ROOT / "sonic/configs/experiments/flat.json").read_text())
        terrain = json.loads(
            (ROOT / "sonic/configs/experiments/terrain_baseline.json").read_text()
        )
        self.assertEqual(flat["schema"], "mm-sonic-flat-experiment/v1")
        self.assertEqual(flat["scene_id"], "sonic-flat-baseline")
        self.assertEqual(flat["route_id"], "flat-12s")
        self.assertEqual(flat["chunk_count"], 30)
        self.assertEqual(flat["duration_s"], 12.0)
        self.assertEqual(flat["terrain_weight"], 0.0)
        self.assertEqual(
            flat["success"],
            {
                "minimum_pelvis_local_height_m": 0.45,
                "minimum_pelvis_up_dot": 0.5,
                "forbidden_contact_groups": ["pelvis", "knees", "torso", "hands"],
                "known_good_metric_multiplier": 1.5,
                "known_good_zero_epsilon": 1.0e-8,
                "require_exact_command_coverage": True,
                "require_exact_frame_coverage": True,
            },
        )
        self.assertEqual(
            terrain,
            {
                "schema": "mm-sonic-terrain-experiment/v1",
                "chunk_intervals": 10,
                "source_rate_hz": 25,
                "target_rate_hz": 50,
                "conditions": {"aware": 4.0, "blind": 0.0},
                "perturbations": [
                    {
                        "id": perturbation_id,
                        "physical_lateral_offset_m": lateral,
                        "physical_yaw_offset_rad": yaw,
                    }
                    for perturbation_id, lateral, yaw in PERTURBATIONS
                ],
                "trial_identity_contract": {
                    "shared_across_conditions_and_perturbations": [
                        "scene_id",
                        "route_id",
                        "command_artifact_id",
                        "command_sha256",
                        "duration_s",
                        "mm_initial_state_sha256",
                    ],
                    "shared_within_condition": ["mm_reference_sha256"],
                    "condition_fields": ["condition", "terrain_weight"],
                    "physical_perturbation_fields": [
                        "perturbation_id",
                        "physical_lateral_offset_m",
                        "physical_yaw_offset_rad",
                    ],
                },
                "scenes": [
                    {"scene_id": "grail-curb-low", "route_id": "curb-forward"},
                    {"scene_id": "ramp-10-up-down", "route_id": "up-landing-down"},
                    {"scene_id": "stairs-shallow", "route_id": "ascent-landing-descent"},
                ],
                "success": {
                    "target_radius_m": 0.25,
                    "duration_multiplier": 1.25,
                    "minimum_pelvis_local_height_m": 0.45,
                    "minimum_pelvis_up_dot": 0.5,
                    "forbidden_contact_groups": ["pelvis", "knees", "torso", "hands"],
                    "require_exact_frame_coverage": True,
                },
                "hypothesis": {
                    "aware_minimum_successes": 8,
                    "aware_minus_blind_minimum": 3,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
