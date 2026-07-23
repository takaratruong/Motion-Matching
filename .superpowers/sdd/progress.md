# Mixed Table/Ground Grasp Search Progress

- Baseline: `18e33928b44b5e343a90d040f786d7e14747d16c` (13 interaction-source tests passing).
- Task 1: complete (commits `18e3392..e4a3c15`, review clean).
- Task 2: complete (commits `e4a3c15..27e9dac`, one review fix, re-review clean).
- Task 3: complete (commits `27e9dac..2582848`, review clean).
- Task 4: complete (commits `2582848..4626814`, one review fix, re-review clean).
- Task 5: complete (commits `4626814..6ff59c8`, one review fix, re-review clean).
- Task 6: complete (real pack built/validated, fixes through `941357b`, final reviews clean, live viewer verified at 685,860 KiB RSS).

## Contact-Anchored Exhaustive Grasp Motion Matching

- Baseline/design: `50f5542`; implementation plan: `0478665`.
- Task 1: `611fef2` (exact contact-relative placement at 12 upright yaw samples).
- Task 2: `6e679c7` (all clips shaped from grasp-relative placement; 1 mm acceptance gate).
- Task 3: `de14dad` (bounded deterministic parallel evaluation of all 4,608 instances).
- Task 4: `2176faa` (furniture-first exhaustive flat viewer with explicit Enter search).
- Task 5: `test: measure exhaustive shared-grasp reach coverage` (probe/report commit containing this ledger entry).
- Automated gate: `python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v`; `make build/tests/test_reach_search build/tests/test_reach_coverage build/tests/test_reach_database build/tests/test_interaction_hand_trajectories build/tests/test_interaction_ik g1_reach_coverage_probe g1_reach_coverage_viewer`; then each of the five test binaries. All passed on 2026-07-23.
- Unchanged-IK gate: `git diff --exit-code 50f5542 -- interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h` passed with no output.
- Real evidence command: `/usr/bin/time -f 'elapsed=%e rss_kb=%M' ./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 --json build/g1-reaches/contact-anchored-coverage-report-v3.json`.
- Real report: `build/g1-reaches/contact-anchored-coverage-report-v3.json`; every fixture completed all 4,608/4,608 instances below 30 seconds and every accepted endpoint was within 1 mm.
- Fixture evidence: open space 355 accepted (178 left, 177 right, 3 azimuth sectors, 2.59 s); table 0 accepted (3.09 s); shelf 42 (4 sectors, 3.66 s); below table 128 (4 sectors, 3.66 s); lower table 2 (2 sectors, 3.90 s). The unchanged IK position gate rejected roughly 3,940 instances per fixture; table survivors were then eliminated by scene collision.
