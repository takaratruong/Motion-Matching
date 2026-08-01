import unittest

from resources.run_g1_torch_contact_oracle import build_parser, select_routes


class ContactOracleCliTests(unittest.TestCase):
    def test_parser_exposes_only_oracle_inputs(self):
        args = build_parser().parse_args(
            [
                "--dataset",
                "data",
                "--config",
                "oracle.json",
                "--g1-xml",
                "g1.xml",
                "--output",
                "out",
                "--device",
                "cuda:7",
                "--route",
                "diagonal-down-left",
            ]
        )

        self.assertEqual(args.route, ["diagonal-down-left"])
        self.assertFalse(hasattr(args, "contact_segments"))
        self.assertFalse(hasattr(args, "foothold_arm"))

    def test_selected_routes_follow_frozen_inventory_order(self):
        routes = select_routes(
            ["diagonal-down-left", "cross-tread-left-to-right"]
        )
        self.assertEqual(
            [route.name for route in routes],
            ["cross-tread-left-to-right", "diagonal-down-left"],
        )
        with self.assertRaisesRegex(ValueError, "unknown route"):
            select_routes(["does-not-exist"])
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            select_routes(["diagonal-down-left", "diagonal-down-left"])


if __name__ == "__main__":
    unittest.main()
