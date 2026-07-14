import os
import subprocess
import unittest


class RaylibBootstrapTests(unittest.TestCase):
    def test_print_lock_is_exact_and_offline(self):
        completed = subprocess.run(
            ["bash", "scripts/bootstrap_raylib.sh", "--print-lock"],
            check=True, text=True, capture_output=True,
            env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
        )
        self.assertEqual(
            completed.stdout,
            "raylib dbc56a87da87d973a9c5baa4e7438a9d20121d28\n"
            "raygui 25c8c65a6e5f0f4d4b564a0343861898c6f2778b\n",
        )

    def test_make_dry_run_uses_local_dependencies(self):
        completed = subprocess.run(
            ["make", "-n", "controller"], check=True,
            text=True, capture_output=True,
        )
        self.assertIn(".deps/raylib/src", completed.stdout)
        self.assertIn(".deps/raygui/src", completed.stdout)
