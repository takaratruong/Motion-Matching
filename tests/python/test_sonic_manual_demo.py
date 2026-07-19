from __future__ import annotations

import json
import math
import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.hands import NEUTRAL_HAND_TARGETS, hand_targets_record
from mm_sonic.joints import ContractError
from mm_sonic.manual_demo import CommandRecorder, _parser
from mm_sonic.manual_evidence import parse_manual_command_artifact


def _stand(index: int) -> CommandSample:
    return CommandSample(index, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _forward(index: int) -> CommandSample:
    return CommandSample(index, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


class CommandRecorderTests(unittest.TestCase):
    def test_onscreen_is_explicit_and_opt_in(self) -> None:
        self.assertFalse(_parser().parse_args([]).onscreen)
        self.assertTrue(_parser().parse_args(["--onscreen"]).onscreen)

    def test_records_committed_commands_in_chunk_order(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=2,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))
        recorder.record(_stand(1))
        recorder.record(_forward(2))

        parsed = parse_manual_command_artifact(recorder.artifact_bytes())

        self.assertEqual(parsed.mode, "script")
        self.assertEqual(parsed.preload_chunks, 2)
        self.assertEqual(
            [command.chunk_index for command in parsed.commands], [0, 1, 2]
        )

    def test_rejects_out_of_order_commit(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=2,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))

        with self.assertRaisesRegex(ContractError, "chunk"):
            recorder.record(_forward(2))

    def test_records_neutral_hand_control_in_v3_command_artifact(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=1,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))
        recorder.record(_forward(1))

        document = json.loads(recorder.artifact_bytes())

        self.assertEqual(document["schema"], "mm-sonic-manual-command/v3")
        self.assertEqual(
            document["hand_control"], hand_targets_record(NEUTRAL_HAND_TARGETS)
        )


if __name__ == "__main__":
    unittest.main()
