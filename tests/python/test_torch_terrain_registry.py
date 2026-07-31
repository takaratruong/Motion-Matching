import hashlib
import math
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
                "e92ad315d8ed212a9fe9cd656199e5bf1389993530ddc2850a4dbe113e0ef18c",
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
        side = next(
            spec
            for spec in CANDIDATE_SPECS
            if spec.logical_name == "staircase-side-stepto"
        )
        self.assertEqual(side.terrain_adapter, "scaled-staircase-084")
        self.assertEqual(
            side.geometry_sha256,
            (
                "78b560b01880e7202f5d36cf55cb66ab8ab98189fdb3f152c4406bf17557ac2f",
                "412cab2eb06f0501d20009823c7127c5293257d87b1e86ffdb3da2c7697ec5de",
                "8d3e9e204fa2646dbac9726ffe0da163f0b5811259981a76ee49c420079ff22c",
                "dc8679af4fa022512613f42e340567fe9ec27697e71613f9d1b9695a767b1ca4",
            ),
        )
        self.assertEqual(
            sum(spec.family == "grail" for spec in CANDIDATE_SPECS),
            4,
        )
        downhill = next(
            spec
            for spec in CANDIDATE_SPECS
            if spec.logical_name == "down-continuous-33"
        )
        self.assertEqual(
            downhill.motion_to_terrain_xy_yaw,
            (-0.29, 3.99, -math.pi / 2.0),
        )
        backward_stair = next(
            spec
            for spec in CANDIDATE_SPECS
            if spec.logical_name == "staircase-final-v3"
        )
        self.assertEqual(
            backward_stair.motion_to_terrain_xy_yaw,
            (-0.24, -0.10, math.pi / 2.0),
        )
        chair_climb = next(
            spec
            for spec in CANDIDATE_SPECS
            if spec.logical_name == "chair-step-climbing-final"
        )
        self.assertEqual(
            chair_climb.motion_to_terrain_xy_yaw,
            (
                -0.025842694938182836,
                0.11438202857971191,
                0.0,
            ),
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
        for transform in (
            [0.0, 0.0, 0.0],
            (0.0, 0.0),
            (0.0, float("nan"), 0.0),
            (0.0, 0.0, "yaw"),
        ):
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(
                    ValueError, "motion-to-terrain transform"
                ):
                    SourceSpec(
                        logical_name="bad-transform",
                        family="stair-local",
                        source_adapter="native-npz",
                        motion_relative_path="motion.npz",
                        motion_sha256="0" * 64,
                        terrain_adapter="fixed-staircase",
                        geometry_relative_paths=(),
                        motion_to_terrain_xy_yaw=transform,
                    )


if __name__ == "__main__":
    unittest.main()
