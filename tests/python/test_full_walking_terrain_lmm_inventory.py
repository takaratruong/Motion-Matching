from __future__ import annotations

import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from mm_sonic.full_walking_terrain_lmm_contracts import inventory_manifest_bytes
from mm_sonic.full_walking_terrain_lmm_inventory import (
    build_inventory,
    load_inventory,
)

from resources.g1_terrain_builder.artifacts import canonical_json_bytes

ROOTS = {
    "bank": Path(
        "/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate"
    ),
    "pfnn_root": Path("/home/ubuntu/datasets/pfnn/pfnn"),
    "grail_root": Path("/home/ubuntu/datasets/GRAIL/data"),
    "takara": Path("/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"),
}


class FullWalkingTerrainLmmInventoryTests(unittest.TestCase):
    @unittest.skipUnless(
        all(path.exists() for path in ROOTS.values()),
        "full PFNN/GRAIL/Takara metadata roots are not present",
    )
    def test_real_metadata_inventory_is_complete_and_walking_only(self):
        inventory = build_inventory(**ROOTS)

        kinds = Counter(record.authority["kind"] for record in inventory.sources)
        families = Counter(record.family for record in inventory.sources)
        self.assertEqual(
            kinds,
            {"pfnn": 80, "grail": 15_837, "takara": 1},
        )
        self.assertEqual(len(inventory.sources), 15_918)
        self.assertEqual(
            {family: families[family] for family in ("curb", "slope", "stair")},
            {"curb": 1_769, "slope": 1_880, "stair": 12_188},
        )
        self.assertRegex(inventory.build_id, r"^[0-9a-f]{64}$")
        self.assertRegex(inventory.manifest_sha256, r"^[0-9a-f]{64}$")

        pfnn = [
            record for record in inventory.sources if record.authority["kind"] == "pfnn"
        ]
        self.assertEqual(sum(record.mirror_of is not None for record in pfnn), 40)
        source_ids = {record.source_id for record in inventory.sources}
        self.assertTrue(
            all(record.mirror_of in source_ids for record in pfnn if record.mirror_of)
        )

        grail_categories = {
            record.authority["category"]
            for record in inventory.sources
            if record.authority["kind"] == "grail"
        }
        self.assertEqual(grail_categories, {"curb", "slope", "stair_p1", "stair_p2"})
        excluded_pfnn_identity = re.compile(
            r"(?:^|[:_])(?:jog|run|crouch|crawl|jump)(?:$|[:_])",
            re.IGNORECASE,
        )
        self.assertFalse(
            [
                record.source_id
                for record in pfnn
                if excluded_pfnn_identity.search(record.source_id)
            ]
        )

        payload = inventory_manifest_bytes(inventory)
        self.assertNotIn(b"/home/ubuntu", payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_bytes(payload)
            self.assertEqual(
                load_inventory(
                    path, expected_manifest_sha256=inventory.manifest_sha256
                ),
                inventory,
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_inventory(path, expected_manifest_sha256="f" * 64)

            changed = json.loads(payload)
            changed["build_id"] = "f" * 64
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "build_id"):
                load_inventory(path)

            changed = json.loads(payload)
            changed["sources"].pop()
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "cardinality"):
                load_inventory(path)


if __name__ == "__main__":
    unittest.main()
