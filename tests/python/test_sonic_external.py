import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mm_sonic.external as external
from mm_sonic.external import (
    ExternalInputError,
    ExternalInputs,
    verify_external,
    verify_gear_checkout,
)


PINNED_PERMUTATION = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

LOCK_PATHS = {
    "joint_names_source": "gear_sonic/envs/manager_env/robots/g1.py",
    "policy_parameters_source": (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
        "policy_parameters.hpp"
    ),
    "zmq_example_source": (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/"
        "test_zmq_manager.py"
    ),
    "zmq_decoder_source": (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
        "input_interface/zmq_packed_message_subscriber.hpp"
    ),
    "stream_merger_source": (
        "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
        "input_interface/streamed_motion_merger.hpp"
    ),
}

KNOWN_GOOD_REFERENCE = (
    "gear_sonic_deploy/reference/example/"
    "walking_quip_360_R_002__A428"
)

REFERENCE_FILES = (
    "body_ang_vel.csv",
    "body_lin_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "info.txt",
    "joint_pos.csv",
    "joint_vel.csv",
    "metadata.txt",
)


class ExternalPreflightTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.checkout = self.root / "gear"
        self.checkout.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "tests@example.invalid")
        self._git("config", "user.name", "External Preflight Tests")

        for key, relative in LOCK_PATHS.items():
            target = self.checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if key == "policy_parameters_source":
                values = ", ".join(str(value) for value in PINNED_PERMUTATION)
                target.write_text(
                    "const std::array<int, 29> mujoco_to_isaaclab = {"
                    f"{values}"
                    "};\n",
                    encoding="utf-8",
                )
            else:
                target.write_text(f"official fixture: {key}\n", encoding="utf-8")

        reference = self.checkout / KNOWN_GOOD_REFERENCE
        reference.mkdir(parents=True)
        for name in REFERENCE_FILES:
            (reference / name).write_text(f"fixture: {name}\n", encoding="utf-8")

        self._commit("fixture checkout")
        self.lock_path = self.root / "gear_sonic.lock.json"
        self._write_lock(self._head())

        self.external_root = self.root / "inputs"
        self.external_root.mkdir()
        self.policy = self.external_root / "policy.onnx"
        self.observation_config = self.external_root / "observation.yaml"
        self.encoder = self.external_root / "encoder.onnx"
        self.terrain_dir = self.external_root / "terrain"
        self.source_mjcf = self.external_root / "g1.xml"
        self.output_root = self.root / "runs"
        self.policy.write_bytes(b"policy checkpoint\n")
        self.observation_config.write_text("history: 4\n", encoding="utf-8")
        self.encoder.write_bytes(b"encoder checkpoint\n")
        self.terrain_dir.mkdir()
        (self.terrain_dir / "scene.bin").write_bytes(b"terrain fixture\n")
        self.source_mjcf.write_text("<mujoco/>\n", encoding="utf-8")

    def tearDown(self):
        self._temporary.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ("git", "-C", str(self.checkout), *args),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _commit(self, message):
        self._git("add", ".")
        self._git("commit", "-q", "-m", message)

    def _head(self):
        return self._git("rev-parse", "HEAD")

    def _write_lock(self, commit):
        lock = {
            "schema": "gear-sonic-lock/v1",
            "repository": "https://github.com/NVlabs/GR00T-WholeBodyControl.git",
            "commit": commit,
            "known_good_reference": KNOWN_GOOD_REFERENCE,
            **LOCK_PATHS,
        }
        self.lock_path.write_text(
            json.dumps(lock, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _args(self, **overrides):
        values = {
            "gear_checkout": self.checkout,
            "policy": self.policy,
            "observation_config": self.observation_config,
            "encoder": self.encoder,
            "terrain_dir": self.terrain_dir,
            "source_mjcf": self.source_mjcf,
            "output_root": self.output_root,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_wrong_commit_is_rejected(self):
        self._write_lock("0" * 40)

        with self.assertRaisesRegex(ExternalInputError, "commit"):
            verify_gear_checkout(self.checkout, self.lock_path)

    def test_missing_official_source_is_rejected(self):
        missing = self.checkout / LOCK_PATHS["zmq_decoder_source"]
        missing.unlink()
        self._git("add", "-u")
        self._git("commit", "-q", "-m", "remove required source")
        self._write_lock(self._head())

        with self.assertRaisesRegex(ExternalInputError, "zmq_decoder_source"):
            verify_gear_checkout(self.checkout, self.lock_path)

    def test_modified_target_permutation_is_rejected(self):
        parameters = self.checkout / LOCK_PATHS["policy_parameters_source"]
        text = parameters.read_text(encoding="utf-8")
        parameters.write_text(
            text.replace(", 28};", ", 27};"),
            encoding="utf-8",
        )
        self._commit("modify permutation")
        self._write_lock(self._head())

        with self.assertRaisesRegex(ExternalInputError, "permutation"):
            verify_gear_checkout(self.checkout, self.lock_path)

    def test_missing_known_good_reference_file_is_rejected(self):
        (self.checkout / KNOWN_GOOD_REFERENCE / "joint_vel.csv").unlink()
        self._git("add", "-u")
        self._git("commit", "-q", "-m", "remove known-good file")
        self._write_lock(self._head())

        with self.assertRaisesRegex(ExternalInputError, "joint_vel.csv"):
            verify_gear_checkout(self.checkout, self.lock_path)

    def test_dirty_checkout_is_rejected_for_scored_runs_and_recorded_otherwise(self):
        (self.checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")

        with self.assertRaisesRegex(ExternalInputError, "dirty"):
            verify_gear_checkout(self.checkout, self.lock_path, scored=True)

        verified = verify_gear_checkout(
            self.checkout,
            self.lock_path,
            scored=False,
        )
        self.assertTrue(verified.gear_dirty)

    def test_permutation_is_validated_from_the_bytes_that_were_hashed(self):
        parameters = self.checkout / LOCK_PATHS["policy_parameters_source"]
        valid_text = parameters.read_text(encoding="utf-8")
        invalid_text = valid_text.replace(", 28};", ", 27};")
        parameters.write_text(invalid_text, encoding="utf-8")
        self._commit("commit invalid permutation")
        self._write_lock(self._head())

        real_sha256_file = external._sha256_file
        swapped = False

        def swap_between_hash_and_validation(path):
            nonlocal swapped
            digest = real_sha256_file(path)
            if path == parameters:
                parameters.write_text(valid_text, encoding="utf-8")
                swapped = True
            elif swapped and path.name == "body_ang_vel.csv":
                parameters.write_text(invalid_text, encoding="utf-8")
            return digest

        with mock.patch.object(
            external,
            "_sha256_file",
            side_effect=swap_between_hash_and_validation,
        ):
            with self.assertRaisesRegex(ExternalInputError, "permutation"):
                verify_gear_checkout(self.checkout, self.lock_path)

    def test_dirty_mutation_after_hashing_is_rejected(self):
        last_reference = self.checkout / KNOWN_GOOD_REFERENCE / "metadata.txt"
        real_sha256_file = external._sha256_file

        def mutate_after_hash(path):
            digest = real_sha256_file(path)
            if path == last_reference:
                (self.checkout / "late-untracked.txt").write_text(
                    "concurrent mutation\n",
                    encoding="utf-8",
                )
            return digest

        with mock.patch.object(
            external,
            "_sha256_file",
            side_effect=mutate_after_hash,
        ):
            with self.assertRaisesRegex(ExternalInputError, "changed"):
                verify_gear_checkout(self.checkout, self.lock_path)

    def test_head_change_after_hashing_is_rejected(self):
        last_reference = self.checkout / KNOWN_GOOD_REFERENCE / "metadata.txt"
        joint_names = self.checkout / LOCK_PATHS["joint_names_source"]
        real_sha256_file = external._sha256_file

        def commit_after_hash(path):
            digest = real_sha256_file(path)
            if path == last_reference:
                joint_names.write_text(
                    "concurrent committed source\n",
                    encoding="utf-8",
                )
                self._commit("concurrent checkout commit")
            return digest

        with mock.patch.object(
            external,
            "_sha256_file",
            side_effect=commit_after_hash,
        ):
            with self.assertRaisesRegex(ExternalInputError, "changed"):
                verify_gear_checkout(self.checkout, self.lock_path)

    def test_output_cannot_duplicate_or_descend_from_an_input_directory(self):
        for output_root in (self.terrain_dir, self.terrain_dir / "nested-runs"):
            with self.subTest(output_root=output_root):
                with self.assertRaisesRegex(ExternalInputError, "output"):
                    ExternalInputs.from_cli(
                        self._args(output_root=output_root),
                    )

    def test_terrain_file_added_after_inventory_is_rejected(self):
        inventoried_file = self.terrain_dir / "scene.bin"
        real_sha256_file = external._sha256_file

        def add_after_hash(path):
            digest = real_sha256_file(path)
            if path == inventoried_file:
                (self.terrain_dir / "late.bin").write_bytes(b"late terrain\n")
            return digest

        with mock.patch.object(
            external,
            "_sha256_file",
            side_effect=add_after_hash,
        ):
            with self.assertRaisesRegex(
                ExternalInputError,
                "directory input changed",
            ):
                ExternalInputs.from_cli(self._args())

    def test_terrain_file_removed_after_hashing_is_rejected(self):
        inventoried_file = self.terrain_dir / "scene.bin"
        real_sha256_file = external._sha256_file

        def remove_after_hash(path):
            digest = real_sha256_file(path)
            if path == inventoried_file:
                inventoried_file.unlink()
            return digest

        with mock.patch.object(
            external,
            "_sha256_file",
            side_effect=remove_after_hash,
        ):
            with self.assertRaisesRegex(
                ExternalInputError,
                "directory input changed",
            ):
                ExternalInputs.from_cli(self._args())

    def test_missing_checkpoint_is_rejected(self):
        self.policy.unlink()

        with self.assertRaisesRegex(ExternalInputError, "policy"):
            ExternalInputs.from_cli(self._args())

    def test_observation_config_must_be_a_file(self):
        self.observation_config.unlink()
        self.observation_config.mkdir()

        with self.assertRaisesRegex(ExternalInputError, "observation_config"):
            ExternalInputs.from_cli(self._args())

    def test_valid_inputs_are_absolute_and_hashes_are_deterministic(self):
        inputs = ExternalInputs.from_cli(self._args())
        first = verify_external(inputs, self.lock_path)
        second = verify_external(inputs, self.lock_path)

        for value in vars(inputs).values():
            if value is not None:
                self.assertTrue(value.is_absolute())
                self.assertEqual(value, value.resolve())
        self.assertEqual(first.inputs, inputs)
        self.assertEqual(first.gear_commit, self._head())
        self.assertEqual(first.hashes, second.hashes)
        self.assertEqual(
            first.hashes["policy"],
            hashlib.sha256(b"policy checkpoint\n").hexdigest(),
        )
        self.assertEqual(
            set(first.hashes),
            {
                "encoder",
                "gear:joint_names_source",
                "gear:known_good_reference/body_ang_vel.csv",
                "gear:known_good_reference/body_lin_vel.csv",
                "gear:known_good_reference/body_pos.csv",
                "gear:known_good_reference/body_quat.csv",
                "gear:known_good_reference/info.txt",
                "gear:known_good_reference/joint_pos.csv",
                "gear:known_good_reference/joint_vel.csv",
                "gear:known_good_reference/metadata.txt",
                "gear:policy_parameters_source",
                "gear:stream_merger_source",
                "gear:zmq_decoder_source",
                "gear:zmq_example_source",
                "observation_config",
                "policy",
                "source_mjcf",
                "terrain_dir",
            },
        )
        for digest in first.hashes.values():
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(
            first.known_good_reference,
            (self.checkout / KNOWN_GOOD_REFERENCE).resolve(),
        )

    def test_encoder_none_is_valid_and_omits_encoder_hash(self):
        inputs = ExternalInputs.from_cli(self._args(encoder=None))

        verified = verify_external(inputs, self.lock_path)

        self.assertIsNone(verified.inputs.encoder)
        self.assertNotIn("encoder", verified.hashes)


if __name__ == "__main__":
    unittest.main()
