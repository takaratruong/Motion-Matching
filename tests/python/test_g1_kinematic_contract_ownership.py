import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "g1_kinematic_contract.h"
PRODUCTION_SOURCES = tuple(sorted(ROOT.glob("*.h"))) + tuple(
    sorted(ROOT.glob("*.cpp"))
)


def production_matches(pattern):
    expression = re.compile(pattern, re.MULTILINE)
    matches = []
    for path in PRODUCTION_SOURCES:
        text = path.read_text(encoding="utf-8")
        matches.extend((path.name, match.group(0)) for match in expression.finditer(text))
    return matches


class G1KinematicContractOwnershipTests(unittest.TestCase):
    def test_contract_header_is_strict_safe_and_directly_consumed(self):
        self.assertTrue(CONTRACT.is_file(), "missing strict-safe G1 contract header")
        text = CONTRACT.read_text(encoding="utf-8")
        for forbidden in ("database.h", "g1_skeleton.h", "g1_ik.h"):
            self.assertNotIn(forbidden, text)
        for consumer in ("g1_skeleton.h", "g1_ik.h", "g1_clearance.cpp"):
            consumer_text = (ROOT / consumer).read_text(encoding="utf-8")
            self.assertIn('#include "g1_kinematic_contract.h"', consumer_text)

    def test_contract_definitions_have_one_production_owner(self):
        owner_patterns = {
            "bone enum": r"\benum\s+G1Bone\b",
            "skeleton signature": r"\bG1_SkeletonSignature\s*\[\]\s*=",
            "leg layout": r"\bstruct\s+G1LegConfig\s*\{",
            "leg factory": r"\bG1LegConfig\s+g1_leg_config\s*\(",
            "left factory": r"\bG1LegConfig\s+g1_left_leg_config\s*\(",
            "right factory": r"\bG1LegConfig\s+g1_right_leg_config\s*\(",
            "25 Hz predicate": r"\bbool\s+g1_dt_is_exact_25_hz\s*\(",
        }
        for name, pattern in owner_patterns.items():
            matches = production_matches(pattern)
            self.assertEqual(len(matches), 1, name)
            self.assertEqual(matches[0][0], CONTRACT.name, name)

        self.assertEqual(
            production_matches(r"0x3d23d70a"),
            [(CONTRACT.name, "0x3d23d70a")],
        )
        self.assertEqual(production_matches(r"\bg1_ik_dt_is_exact_25_hz\b"), [])

    def test_fixed_geometry_literals_have_one_production_owner(self):
        geometry_literals = (
            r"vec3\(-0\.05f, -0\.03f, -0\.025f\)",
            r"vec3\(\+0\.12f, -0\.03f, \+0\.030f\)",
            r"vec3\(-0\.078f, -0\.17f, 0\.0f\)",
            r"vec3\(0\.0f, -0\.28f, 0\.0f\)",
        )
        for pattern in geometry_literals:
            matches = production_matches(pattern)
            self.assertEqual(len(matches), 1, pattern)
            self.assertEqual(matches[0][0], CONTRACT.name)


if __name__ == "__main__":
    unittest.main()
