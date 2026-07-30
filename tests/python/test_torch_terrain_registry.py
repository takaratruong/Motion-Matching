import hashlib
from pathlib import Path
import tempfile
import unittest

from resources.g1_torch_terrain_builder.registry import (
    CANDIDATE_SPECS,
    SourceSpec,
    resolve_source_spec,
)


class ExpandedTerrainRegistryTests(unittest.TestCase):
    def test_registry_pins_exact_candidate_identity_and_families(self):
        by_name = {
            spec.logical_name: (spec.family, spec.motion_sha256)
            for spec in CANDIDATE_SPECS
        }
        self.assertEqual(len(CANDIDATE_SPECS), 24)
        self.assertEqual(len(by_name), 24)
        self.assertEqual(
            by_name["staircase-side-stepto"],
            (
                "stair-local",
                "aba4f13be96bb888acc99bdaec11ef2434195a28d5bb69c8803b6b03644be0be",
            ),
        )
        self.assertEqual(
            by_name["up-continuous-33"],
            (
                "stair-local",
                "03da1a99a161327fd6650e8b304934e182ab34b3ff9ec8cc0978c69700a6696c",
            ),
        )
        self.assertEqual(
            by_name["chair-step-climbing-final"],
            (
                "curb-chair",
                "69c545d85349033d1dbdc41041ea2baec730e3700f16dd1075c53fd08c99dc76",
            ),
        )
        self.assertEqual(
            sum(spec.family == "grail" for spec in CANDIDATE_SPECS),
            4,
        )
        self.assertNotIn("crane", by_name)

    def test_resolver_rejects_changed_hash_symlink_and_escape(self):
        payload = b"registered motion"
        digest = hashlib.sha256(payload).hexdigest()
        spec = SourceSpec(
            logical_name="fixture",
            family="stair-local",
            source_adapter="native-npz",
            motion_relative_path="fixture/motion.npz",
            motion_sha256=digest,
            terrain_adapter="fixed-staircase",
            geometry_relative_paths=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fixture" / "motion.npz"
            path.parent.mkdir()
            path.write_bytes(payload)
            resolved = resolve_source_spec(
                spec, source_root=root, grail_root=root
            )
            self.assertEqual(resolved.motion_path, path.resolve())

            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_source_spec(spec, source_root=root, grail_root=root)

            path.unlink()
            target = root / "target.npz"
            target.write_bytes(payload)
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "real file"):
                resolve_source_spec(spec, source_root=root, grail_root=root)

            escape = SourceSpec(
                logical_name="escape",
                family="stair-local",
                source_adapter="native-npz",
                motion_relative_path="../target.npz",
                motion_sha256=digest,
                terrain_adapter="fixed-staircase",
                geometry_relative_paths=(),
            )
            with self.assertRaisesRegex(ValueError, "inside"):
                resolve_source_spec(escape, source_root=root, grail_root=root)

    def test_source_spec_rejects_duplicate_or_invalid_contract_fields(self):
        with self.assertRaisesRegex(ValueError, "logical_name"):
            SourceSpec(
                logical_name="",
                family="stair-local",
                source_adapter="native-npz",
                motion_relative_path="motion.npz",
                motion_sha256="0" * 64,
                terrain_adapter="fixed-staircase",
                geometry_relative_paths=(),
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            SourceSpec(
                logical_name="bad-hash",
                family="stair-local",
                source_adapter="native-npz",
                motion_relative_path="motion.npz",
                motion_sha256="xyz",
                terrain_adapter="fixed-staircase",
                geometry_relative_paths=(),
            )


if __name__ == "__main__":
    unittest.main()
