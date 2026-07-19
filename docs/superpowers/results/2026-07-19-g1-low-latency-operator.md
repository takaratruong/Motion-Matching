# G1 low-latency operator result

## Outcome

Select a two-chunk manual lookahead for the user-facing driver. This reduces
command lookahead from 1.6 seconds to 0.8 seconds while preserving the
qualified four-chunk default for formal evidence runs.

The one-chunk candidate remained upright but is rejected: its final two-second
displacement was 0.402407 m, above the unchanged 0.25 m settled-stop bound.

## Selected two-chunk physics trial

- Run root: `/home/ubuntu/mm-sonic-low-latency-depth2-runs/manual-sonic/manual-20260719T045800291966Z-53592`
- Summary SHA-256: `4ae01f9add158daf94d6b47159bba4eaa1767b2895bdeab83a2b49389f78119d`
- Easy video: `/home/ubuntu/g1-sonic-0.8s-pass.mp4`
- Video SHA-256: `4f59de7afb06666dcd8a9bae1d79506b002707e520e987b33661c9a7be4c83cf`
- Media: H.264, 1280x720, 50 fps, 600 frames, 12.0 seconds
- Preload: 2 chunks / 0.8 seconds
- State rows: 600; control steps: 2401; simulated time: 12.005 seconds
- No production fall marker
- Minimum root height: 0.743727618 m
- Minimum pelvis-up dot: 0.996927325
- Path distance: 5.179320087 m
- Net yaw change: 0.737539103 rad
- Final two-second displacement: 0.190914359 m
- Hand tracking error and hard-limit checks: pass

Visual inspection of the full six-frame montage and final hand close-up found
an upright, grounded walking/turning trajectory, substantially symmetric
closed hands, ordinary arm motion, and no flailing or suspension.

## Rejected one-chunk comparison

- Run root: `/home/ubuntu/mm-sonic-low-latency-depth1-runs/manual-sonic/manual-20260719T050011086264Z-74169`
- Summary SHA-256: `5457995fc6b628eebb7342ea64a57fc6502133db73bea7c0ca80cf7bf9969104`
- Preserved video: `/home/ubuntu/g1-sonic-0.4s-rejected.mp4`
- Video SHA-256: `bf30d1cb98badc4db92c84474d5dc298ff20d0349600776dc5815e2c713b67b2`
- Preload: 1 chunk / 0.4 seconds
- State rows: 600; no production fall marker
- Minimum root height: 0.746519720 m
- Minimum pelvis-up dot: 0.996566414
- Path distance: 5.188351510 m
- Net yaw change: 0.744228431 rad
- Final two-second displacement: 0.402407364 m — reject
- Hand tracking error and hard-limit checks: pass

## Code and verification

- Branch: `research/g1-low-latency-driver`
- Implementation commit: `0009c5b8a72e2997fa719589858ad83399b62ece`
- The parser default remains four chunks; only an explicit non-scored manual
  option permits depths one through four.
- Frozen manual/evidence gate: 52 passed.
- Post-visual hand/scene/manual/replay gate: 116 passed.
- Post-visual rolling operator/coordinator gate: 116 passed.
- Protected neutral-hand/actuator evaluator: pass, 43 actuators, profile
  `582ab41307e5e89149c9b572d1c5e6c5bf50716580994da9c41a8f80f2759e52`.

Reliable Claude job
`job-0f7d49747eca5f9754d31dcb179125e9` produced and reviewed the exact
two-file change. The protected tests passed, then the foreman correctly
stopped at the repository's active Git-LFS hook boundary. Decision
`decision-001` atomically retained the candidate as a terminal authenticated
handoff. Codex reapplied it with only test-class organization cleanup and
reran the frozen gate.

The supervising release is the detached SHA
`de2ec6d3c1d46d772369a3b447b287b1d1e72e2d`, package/plugin version
`0.1.1+codex.20260716220019`. Its exact virtual environment passed 680 tests
in 1618.73 seconds and its real public foreman/recovery canary passed accepted
generation `g002` with two protected gate artifacts.

## User entry point

`/home/ubuntu/drive-g1-sonic.sh` now launches the low-latency worktree with
`--preload-chunks 2` and reports 0.8 seconds of lookahead. The formal manual
evidence auditor remains fixed at four chunks.
