from .layout import CONTACT_ORDER, INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S

__all__ = [
    "CONTACT_ORDER",
    "INPUT_LAYOUT",
    "OUTPUT_LAYOUT",
    "TRAJECTORY_TIMES_S",
    "PhaseFunctionedNetwork",
    "catmull_rom_phase_banks",
]


def __getattr__(name: str):
    """Load Torch-dependent model symbols only when a caller requests them."""

    if name in {"PhaseFunctionedNetwork", "catmull_rom_phase_banks"}:
        from .model import PhaseFunctionedNetwork, catmull_rom_phase_banks

        return {
            "PhaseFunctionedNetwork": PhaseFunctionedNetwork,
            "catmull_rom_phase_banks": catmull_rom_phase_banks,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
