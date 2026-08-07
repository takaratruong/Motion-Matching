import unittest

from mm_sonic.select_cowarped_exotic_collection import select


def _row(bank: str, clip: int, source_mode: str, compound_mode: str) -> dict:
    return {
        "collection_bank": bank,
        "pilot": f"/{bank}/{clip}/{source_mode}/{compound_mode}",
        "clip_family": "stairs",
        "clip_index": clip,
        "source_mode": source_mode,
        "compound_mode": compound_mode,
        "clip_traversal": "up" if clip % 2 == 0 else "down",
        "maximum_stance_run_drift_m": 0.001,
        "maximum_foot_penetration_m": 0.002,
    }


class ExoticCollectionSelectionTest(unittest.TestCase):
    def test_respects_physical_source_cap_and_covers_banks(self):
        rows = [
            _row(bank, clip, source_mode, compound_mode)
            for bank in ("stair", "rough")
            for clip in range(6)
            for source_mode in ("diagonal", "lane")
            for compound_mode in ("reverse", "stop_restart")
        ]
        chosen = select(rows, count=12, maximum_per_physical_source=2)

        self.assertEqual(len(chosen), 12)
        self.assertEqual({row["collection_bank"] for row in chosen}, {"stair", "rough"})
        physical_counts: dict[tuple[str, int], int] = {}
        for row in chosen:
            key = (str(row["clip_family"]), int(row["clip_index"]))
            physical_counts[key] = physical_counts.get(key, 0) + 1
        self.assertLessEqual(max(physical_counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
