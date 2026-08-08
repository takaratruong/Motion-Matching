from .layout import CONTACT_ORDER, INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S

__all__ = [
    "CONTACT_ORDER",
    "INPUT_LAYOUT",
    "OUTPUT_LAYOUT",
    "TRAJECTORY_TIMES_S",
    "PhaseFunctionedNetwork",
    "catmull_rom_phase_banks",
    "RecurrentTrajectoryState",
    "PlannedTrajectory",
    "initialize_recurrent_state",
    "derive_training_desired_velocity",
    "plan_recurrent_trajectory",
    "pack_recurrent_input",
    "advance_recurrent_state",
]


def __getattr__(name: str):
    """Load Torch-dependent model symbols only when a caller requests them."""

    if name in {"PhaseFunctionedNetwork", "catmull_rom_phase_banks"}:
        from .model import PhaseFunctionedNetwork, catmull_rom_phase_banks

        return {
            "PhaseFunctionedNetwork": PhaseFunctionedNetwork,
            "catmull_rom_phase_banks": catmull_rom_phase_banks,
        }[name]
    if name in {
        "RecurrentTrajectoryState",
        "PlannedTrajectory",
        "initialize_recurrent_state",
        "derive_training_desired_velocity",
        "plan_recurrent_trajectory",
        "pack_recurrent_input",
        "advance_recurrent_state",
    }:
        from .recurrence import (
            PlannedTrajectory,
            RecurrentTrajectoryState,
            advance_recurrent_state,
            derive_training_desired_velocity,
            initialize_recurrent_state,
            pack_recurrent_input,
            plan_recurrent_trajectory,
        )

        return {
            "RecurrentTrajectoryState": RecurrentTrajectoryState,
            "PlannedTrajectory": PlannedTrajectory,
            "initialize_recurrent_state": initialize_recurrent_state,
            "derive_training_desired_velocity": derive_training_desired_velocity,
            "plan_recurrent_trajectory": plan_recurrent_trajectory,
            "pack_recurrent_input": pack_recurrent_input,
            "advance_recurrent_state": advance_recurrent_state,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
