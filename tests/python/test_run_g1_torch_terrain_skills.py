import unittest

from resources.run_g1_torch_terrain_skills import build_parser, select_routes


class TerrainSkillsCliTest(unittest.TestCase):
    def test_parser_accepts_qualification_slice(self):
        args = build_parser().parse_args(
            [
                "--dataset", "data", "--config", "config.json",
                "--g1-xml", "g1.xml", "--output", "out",
                "--qualification-slice",
            ]
        )
        self.assertTrue(args.qualification_slice)

    def test_route_selection_preserves_frozen_order(self):
        routes = select_routes(
            ["diagonal-down-left", "cross-tread-left-to-right"], False
        )
        self.assertEqual(
            [route.name for route in routes],
            ["cross-tread-left-to-right", "diagonal-down-left"],
        )

    def test_qualification_slice_has_six_routes(self):
        self.assertEqual(len(select_routes([], True)), 6)


if __name__ == "__main__":
    unittest.main()
