import importlib.util
from pathlib import Path
import unittest

from mm_sonic.joints import ContractError
from mm_sonic.torch_grail_contact_window_search import ContactWindow


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "resources" / "run_g1_grail_contact_inventory.py"
SPEC = importlib.util.spec_from_file_location(
    "run_g1_grail_contact_inventory", PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate(window_id, cost):
    return ContactWindow(
        window_id=window_id,
        role="uneven_walk",
        source_clip=f"clip-{window_id}",
        start_frame=1,
        stop_frame=20,
        support_start=(True, False),
        support_stop=(True, False),
        height_pattern_error_m=cost,
        timing_error_frames=0.0,
        sole_transform_error_m=0.0,
        heading_error_rad=0.0,
        pelvis_error_m=0.0,
        start_pose=(0.0, 0.0),
        stop_pose=(0.0, 0.0),
        start_velocity=(0.0, 0.0),
        stop_velocity=(0.0, 0.0),
    )


class GrailContactInventoryTests(unittest.TestCase):
    def test_ranking_is_deterministic_and_bounded(self):
        ranked = MODULE._rank_windows(
            (candidate("b", 0.01), candidate("a", 0.01), candidate("c", 0.02)),
            maximum_results=2,
        )

        self.assertEqual([item.window_id for item in ranked], ["a", "b"])

    def test_empty_inventory_is_explicit_failure(self):
        with self.assertRaisesRegex(ContractError, "no GRAIL contact"):
            MODULE._rank_windows((), maximum_results=10)


if __name__ == "__main__":
    unittest.main()
