import math
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mm_sonic.torch_contact_oracle_actions import (
    ContactPhaseAction,
    ContactPhaseActionIndex,
    ContactPhaseInventory,
    action_from_profiles,
)
from mm_sonic.torch_contact_oracle_rollout import (
    run_oracle_matrix,
    run_oracle_route,
    save_oracle_matrix,
)
from mm_sonic.torch_contact_oracle_search import (
    OracleCost,
    OracleExpansionDiagnostics,
    OraclePlan,
    OracleSearchFailure,
    OracleState,
    advance_state,
    place_action,
)
from mm_sonic.torch_terrain_omni_routes import (
    OmniRoute,
    RouteCommand,
    RouteOutcomeContract,
    StairFrame,
)


def _action() -> ContactPhaseAction:
    frames = 7
    root = torch.zeros((frames, 3))
    root[:, 0] = torch.linspace(0.0, 0.06, frames)
    feet = torch.zeros((frames, 2, 3))
    feet[:, :, 0] = root[:, 0, None]
    feet[:, 0, 1] = 0.1
    feet[:, 1, 1] = -0.1
    feet[:, :, 2] = 0.035
    orientation = torch.zeros((frames, 4))
    orientation[:, 0] = math.cos(0.2)
    orientation[:, 1] = math.sin(0.2)
    return ContactPhaseAction(
        clip_index=0,
        start_frame=10,
        end_frame=17,
        swing_foot=1,
        entry_support=torch.tensor((True, True)),
        exit_support=torch.tensor((True, True)),
        support_mask=torch.ones((frames, 2), dtype=torch.bool),
        joint_position=torch.zeros((frames, 29)),
        joint_velocity=torch.zeros((frames, 29)),
        root_position_local=root,
        root_yaw_local=torch.zeros(frames),
        root_orientation_local_wxyz=orientation,
        foot_position_local=feet,
        foot_surface_delta_m=torch.zeros((frames, 2)),
        minimum_swing_clearance_m=0.0,
    )


def _state() -> OracleState:
    return OracleState(
        root_position_world=torch.zeros(3),
        root_yaw_world=torch.zeros(()),
        root_orientation_world_wxyz=torch.tensor(
            (math.cos(0.2), math.sin(0.2), 0.0, 0.0)
        ),
        foot_position_world=torch.tensor(
            ((0.0, 0.1, 0.035), (0.0, -0.1, 0.035))
        ),
        support_mask=torch.tensor((True, True)),
        joint_position=torch.zeros(29),
        joint_velocity=torch.zeros(29),
        route_frame=0,
    )


class _RecordingPlanner:
    def __init__(self, action, *, beam_marker=1):
        self.action = action
        self.beam_marker = beam_marker
        self.requested_route_frames = []

    def __call__(self, state, _schedule):
        self.requested_route_frames.append(state.route_frame)
        first = place_action(self.action, state)
        horizon = 1 if state.route_frame >= 12 else 2
        placements = [first]
        if horizon == 2:
            placements.append(
                place_action(self.action, advance_state(state, first))
            )
        costs = tuple(OracleCost() for _ in placements)
        return OraclePlan(
            action_indices=tuple(0 for _ in placements),
            placements=tuple(placements),
            step_costs=costs,
            total_cost=OracleCost(),
            horizon_landings=horizon,
            expansion=OracleExpansionDiagnostics(
                1, {}, (self.beam_marker,) * horizon
            ),
        )


class ContactOracleRolloutTests(unittest.TestCase):
    def test_executes_one_edge_then_replans_at_terminal_contact(self):
        route = OmniRoute(
            name="fourteen-frame",
            commands=(
                RouteCommand((0.3, 0.0), 0.0, 14, "move", reset_before=True),
            ),
            required_outcome="traverse",
            outcome=RouteOutcomeContract(),
        )
        stair = StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3)
        planner = _RecordingPlanner(_action())

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        run = run_oracle_route(
            route=route,
            stair_frame=stair,
            initial_state=_state(),
            planner=planner,
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )

        self.assertEqual(planner.requested_route_frames, [0, 6, 12])
        self.assertEqual(run.completed_frames, 14)
        self.assertEqual(
            run.arrays["planned_horizon"].tolist(),
            [2] * 7 + [2] * 6 + [1],
        )
        self.assertTrue(run.completed_without_exception)
        self.assertTrue(math.isfinite(run.metrics.stance_slide_m.total))
        np.testing.assert_allclose(
            run.arrays["qpos"][:, 3:7],
            run.arrays["root_orientation_world_wxyz"],
        )
        self.assertGreater(abs(run.arrays["qpos"][0, 4]), 0.1)
        self.assertEqual(len(run.plan_events[0].step_costs), 2)
        self.assertEqual(len(run.plan_events[0].planned_landing_world_xyz), 2)
        self.assertIsNotNone(run.plan_events[0].emitted_landing_world_xyz)
        self.assertEqual(
            len(run.plan_events[0].entry_joint_position_error_rad), 2
        )

    def test_emitted_qpos_reproduces_saved_feet_with_authoritative_fk(self):
        from mm_sonic.torch_g1_fk import MujocoG1FootKinematics

        xml = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
        if not Path(xml).is_file():
            self.skipTest("G1 MuJoCo model is unavailable")
        helper = MujocoG1FootKinematics(xml)
        frames = 2
        roll = 0.4
        orientation = np.repeat(
            np.array(
                [[math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0]]
            ),
            frames,
            axis=0,
        )
        joints = np.zeros((frames, 29), dtype=np.float64)
        roots = np.zeros((frames, 3), dtype=np.float64)
        unshifted = helper.foot_positions(joints, roots, orientation)
        roots[:, 2] = 0.035 - float(unshifted[0, :, 2].min())
        feet = helper.foot_positions(joints, roots, orientation)
        surface = feet[..., 2] - 0.035
        action = action_from_profiles(
            clip_index=0,
            start_frame=0,
            landing_frame=1,
            support_mask=torch.tensor(
                ((True, False), (True, True)), dtype=torch.bool
            ),
            joint_position=torch.tensor(joints, dtype=torch.float32),
            joint_velocity=torch.zeros((frames, 29)),
            root_position_world=torch.tensor(roots, dtype=torch.float32),
            root_orientation_world_wxyz=torch.tensor(
                orientation, dtype=torch.float32
            ),
            foot_position_world=torch.tensor(feet, dtype=torch.float32),
            foot_surface_height_m=torch.tensor(surface, dtype=torch.float32),
        )
        initial = OracleState(
            root_position_world=torch.tensor(roots[0], dtype=torch.float32),
            root_yaw_world=torch.zeros(()),
            root_orientation_world_wxyz=torch.tensor(
                orientation[0], dtype=torch.float32
            ),
            foot_position_world=torch.tensor(feet[0], dtype=torch.float32),
            support_mask=torch.tensor((True, False)),
            joint_position=torch.zeros(29),
            joint_velocity=torch.zeros(29),
            route_frame=0,
        )
        reference_xy = torch.tensor(feet[0, :, :2], dtype=torch.float32)
        reference_height = torch.tensor(surface[0], dtype=torch.float32)

        def source_surface(points):
            distance = torch.sum(
                (points[..., None, :] - reference_xy) ** 2, dim=-1
            )
            return reference_height[torch.argmin(distance, dim=-1)]

        planner = _RecordingPlanner(action)
        run = run_oracle_route(
            route=OmniRoute(
                name="authoritative-fk",
                commands=(
                    RouteCommand(
                        (0.1, 0.0), 0.0, 2, "move", reset_before=True
                    ),
                ),
                required_outcome="mixed",
                outcome=RouteOutcomeContract(),
            ),
            stair_frame=StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3),
            initial_state=initial,
            planner=planner,
            terrain_sampler=source_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )

        self.assertTrue(run.completed_without_exception, run.failure)
        actual = helper.foot_positions(
            run.arrays["joint_position"],
            run.arrays["root_position_world"],
            run.arrays["root_orientation_world_wxyz"],
        )
        np.testing.assert_allclose(
            actual, run.arrays["foot_position_world"], rtol=0.0, atol=2e-7
        )

    def test_zero_speed_frames_hold_without_invoking_planner(self):
        route = OmniRoute(
            name="hold",
            commands=(
                RouteCommand((0.0, 0.0), 0.0, 3, "stop", reset_before=True),
            ),
            required_outcome="mixed",
            outcome=RouteOutcomeContract(),
        )

        def forbidden_planner(_state, _schedule):
            raise AssertionError("planner must not run while stopped")

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        run = run_oracle_route(
            route=route,
            stair_frame=StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3),
            initial_state=_state(),
            planner=forbidden_planner,
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )

        self.assertEqual(run.completed_frames, 3)
        self.assertTrue(run.completed_without_exception)
        self.assertEqual(run.arrays["selected_action_index"].tolist(), [-1, -1, -1])
        self.assertTrue((run.arrays["root_position_world"] == 0.0).all())

    def test_stop_boundary_never_freezes_an_unstable_mid_action_pose(self):
        route = OmniRoute(
            name="move-then-stop",
            commands=(
                RouteCommand((0.3, 0.0), 0.0, 2, "move", reset_before=True),
                RouteCommand((0.0, 0.0), 0.0, 3, "stop"),
            ),
            required_outcome="mixed",
            outcome=RouteOutcomeContract(),
        )
        action = _action()
        support = action.support_mask.clone()
        support[1:-1, 1] = False
        action = replace(action, support_mask=support)
        planner = _RecordingPlanner(action)

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        run = run_oracle_route(
            route=route,
            stair_frame=StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3),
            initial_state=_state(),
            planner=planner,
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )

        self.assertTrue(run.completed_without_exception)
        self.assertEqual(
            run.arrays["selected_action_index"].tolist(), [-1, -1, -1, -1, -1]
        )
        np.testing.assert_allclose(
            run.arrays["root_position_world"][2:],
            np.repeat(
                run.arrays["root_position_world"][0][None, :], 3, axis=0
            ),
        )

    def test_plan_diagnostics_participate_in_route_hash(self):
        route = OmniRoute(
            name="diagnostic-hash",
            commands=(
                RouteCommand((0.3, 0.0), 0.0, 2, "move", reset_before=True),
            ),
            required_outcome="mixed",
            outcome=RouteOutcomeContract(),
        )

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        kwargs = dict(
            route=route,
            stair_frame=StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3),
            initial_state=_state(),
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )
        first = run_oracle_route(
            planner=_RecordingPlanner(_action(), beam_marker=1), **kwargs
        )
        second = run_oracle_route(
            planner=_RecordingPlanner(_action(), beam_marker=2), **kwargs
        )

        for name in first.arrays:
            if name != "plan_time_ns":
                np.testing.assert_array_equal(
                    first.arrays[name], second.arrays[name]
                )
        self.assertNotEqual(
            first.deterministic_sha256, second.deterministic_sha256
        )

    def test_search_failure_is_structured_and_never_replaced(self):
        route = OmniRoute(
            name="failure",
            commands=(
                RouteCommand((0.3, 0.0), 0.0, 3, "move", reset_before=True),
            ),
            required_outcome="mixed",
            outcome=RouteOutcomeContract(),
        )

        def failed_planner(_state, _schedule):
            raise OracleSearchFailure({"joint-position": 17})

        def flat_surface(points):
            return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

        run = run_oracle_route(
            route=route,
            stair_frame=StairFrame((0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3),
            initial_state=_state(),
            planner=failed_planner,
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
        )

        self.assertEqual(run.completed_frames, 0)
        self.assertFalse(run.completed_without_exception)
        self.assertEqual(run.failure.stage, "oracle-search")
        self.assertEqual(
            run.failure.message, "no feasible one-contact oracle action"
        )
        self.assertEqual(
            dict(run.failure.rejected_by_reason), {"joint-position": 17}
        )

    def test_failure_hash_is_independent_of_rejection_mapping_order(self):
        route = OmniRoute(
            name="failure-hash",
            commands=(
                RouteCommand((0.3, 0.0), 0.0, 2, "move", reset_before=True),
            ),
            required_outcome="mixed",
            outcome=RouteOutcomeContract(),
        )

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        def run(rejected):
            def failed_planner(_state, _schedule):
                raise OracleSearchFailure(rejected)

            return run_oracle_route(
                route=route,
                stair_frame=StairFrame(
                    (0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3
                ),
                initial_state=_state(),
                planner=failed_planner,
                terrain_sampler=flat_surface,
                clip_paths=("clip0",),
                dataset_identity="dataset",
                config_identity="config",
            )

        first = run({"entry-foot-error": 2, "joint-position": 1})
        second = run({"joint-position": 1, "entry-foot-error": 2})

        self.assertEqual(
            first.deterministic_sha256, second.deterministic_sha256
        )

    def test_matrix_isolates_route_failures_and_hashes_deterministically(self):
        routes = (
            OmniRoute(
                name="hold-success",
                commands=(
                    RouteCommand(
                        (0.0, 0.0), 0.0, 2, "stop", reset_before=True
                    ),
                ),
                required_outcome="mixed",
                outcome=RouteOutcomeContract(),
            ),
            OmniRoute(
                name="move-failure",
                commands=(
                    RouteCommand(
                        (0.3, 0.0), 0.0, 2, "move", reset_before=True
                    ),
                ),
                required_outcome="mixed",
                outcome=RouteOutcomeContract(),
            ),
        )
        index = ContactPhaseActionIndex(
            actions=(_action(),),
            inventory=ContactPhaseInventory(1, {}),
            exact_successor_indices=(None,),
        )

        def failed_planner(_state, _schedule):
            raise RuntimeError("no safe action")

        def flat_surface(points):
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        kwargs = dict(
            routes=routes,
            stair_frame=StairFrame(
                (0.0, 0.0), 0.0, 0.6, 0.3, 0.18, 3
            ),
            initial_state=_state(),
            planner=failed_planner,
            terrain_sampler=flat_surface,
            clip_paths=("clip0",),
            dataset_identity="dataset",
            config_identity="config",
            action_index=index,
            constraints=None,
        )
        from mm_sonic.torch_contact_oracle_search import OracleConstraints

        kwargs["constraints"] = OracleConstraints()
        first = run_oracle_matrix(**kwargs)
        second = run_oracle_matrix(**kwargs)

        self.assertEqual(first.runs[0].completed_frames, 2)
        self.assertEqual(first.runs[1].failure.stage, "oracle-search")
        self.assertFalse(first.matrix_pass)
        self.assertEqual(first.deterministic_sha256, second.deterministic_sha256)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "matrix"
            save_oracle_matrix(first, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                summary["deterministic_sha256"], first.deterministic_sha256
            )
            route_dir = output / "routes" / "hold-success"
            self.assertTrue((route_dir / "rollout.npz").is_file())
            self.assertTrue((route_dir / "diagnostics.json").is_file())
            with np.load(route_dir / "rollout.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["qpos"].shape, (2, 36))
            with self.assertRaises(FileExistsError):
                save_oracle_matrix(first, output)


if __name__ == "__main__":
    unittest.main()
