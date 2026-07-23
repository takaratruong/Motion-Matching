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
- Task 5: `27eb950` (shared-grasp real-pack coverage probe and report contract).
- Task 6 review fixes: `46e5006`, `ad28d84` (deadline-complete semantics, oriented-object rendering, rejected-option inspection, explicit zero-coverage evidence, and deterministic deadline testing).
- Live grounding correction: whole-body placement now preserves recorded root Y and applies grasp-height displacement only through active-arm IK; grounded 5 cm height-retarget regression added.
- Automated gate: `python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v`; `make build/tests/test_reach_search build/tests/test_reach_coverage build/tests/test_reach_database build/tests/test_interaction_hand_trajectories build/tests/test_interaction_ik g1_reach_coverage_probe g1_reach_coverage_viewer`; then each of the five test binaries. All passed on 2026-07-23.
- Unchanged-IK gate: `git diff --exit-code 50f5542 -- interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h` passed with no output.
- Real evidence command: `/usr/bin/time -f 'elapsed=%e rss_kb=%M' ./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 --json build/g1-reaches/contact-anchored-coverage-report-v3.json`.
- Real report: `build/g1-reaches/contact-anchored-coverage-report-v3.json`; every fixture completed all 4,608/4,608 instances below 30 seconds and every accepted endpoint was within 1 mm.
- Grounded fixture evidence after strict per-sample height-ramp IK: open space 388 accepted (194 left, 194 right, 7 azimuth sectors, 5.04 s); table 0 (5.54 s); shelf 0 (5.53 s); below table 0 (5.73 s); lower table 8 (2 sectors, 6.87 s). Zero-coverage fixtures are explicit failures, not vacuous zero-error successes. The reduced furniture coverage is the truthful result when grasp height may not float the full body.

## Grounded G1 Mesh Coverage Viewer

- Design: `7133a06`; plan: `f557b3c`.
- Scene/camera: `986816c` lowers the main tabletop to `0.65 m` and adds bounded left-drag orbit, middle-drag pan, and wheel zoom.
- Mesh: `f75f5af` feeds the certified articulated G1 renderer and optional blue skeleton from the same selected 31-bone `WorldPose`; `M` and `B` are visual-only toggles with skeleton fallback.
- Review hardening: `8c6213c` checks window readiness before mesh loading, wraps/bounds camera state, and exercises certified-GLB load/update/draw/unload in a hidden runtime test. Focused re-review found no remaining Critical or Important issues.
- Automated gate: viewer Python contracts, certified mesh renderer test, exhaustive search/coverage/database/trajectory/IK tests, probe/viewer release builds, and unchanged-IK diff all passed on 2026-07-23.
- Lowered-table real report: `build/g1-reaches/contact-anchored-coverage-report-v3.json`; every fixture completed 4,608/4,608 within 30 seconds.
- Lowered-table counts: open space 388 accepted (7 sectors, 5.05 s); main table 0 (6.17 s); shelf 2 (1 sector, 5.77 s); below table 0 (5.49 s); lower table 2 (1 sector, 6.59 s). Gates were not weakened.

## Swept Grasp Collision and Bilateral Posture IK

- Swept object filter: `a4f00d8` analytically sweeps all three active wrist-chain joints against the fully oriented object; `9917c08` removes the five-sample exemption so only terminal endpoint contact is permitted.
- Posture solver: imported reviewed left solver at `24e0aaa`, generalized direct metadata/FK/limits to both hands at `a3a1911`, and integrated temporally seeded final-`0.6 s` shaping at `439ac9b`. Exhaustive reach shaping has no arm-only IK fallback.
- Focused gate: 10 viewer/source contracts plus posture IK, hand trajectory, reach coverage, reach search, reach database, legacy IK, certified GLB lifecycle, probe, and viewer builds passed on 2026-07-23.
- Real report: `build/g1-reaches/contact-anchored-coverage-report-v4.json`; every fixture processed all 4,608 identities, integrity passed, and each fixture completed below 10 seconds.
- Strict-quality counts after posture IK and swept collision: open space 6 accepted (3 left, 3 right, 4 azimuth sectors, 7.58 s); table 0 (7.92 s); shelf 0 (8.14 s); below table 0 (8.24 s); lower table 0 (9.11 s). No IK, orientation, or collision threshold was weakened; total five-fixture wall time was 41.07 s at 55,332 KiB peak RSS.
- Focused implementation review found no remaining Critical or Important issues in oriented sweep semantics, bilateral metadata/FK/limits, temporal seeding, unowned-pose preservation, finite handling, or thread safety.

## Posture-IK Coverage Regression Fix

- Design and plan: `afd859e`, `8968b31`.
- Implementation: `feb79b0` adds bilateral quality-first/task-feasible posture IK; `c934569` restores a fully aligned final five-frame approach; `bf5d07b` makes the measured pre-IK open-space baseline a hard probe gate; `eb93c5a` measures root sectors across every accepted placed candidate instead of one clip per yaw.
- Review fixes: `a45c132` rejects invalid configuration/seed inputs before either solver stage and adds executable translation/continuity regressions; `6098568` extracts and behaviorally tests the production aggregate sector metric. Focused re-review found no remaining Critical or Important issues.
- Focused verification on 2026-07-23: 12 viewer/probe Python contracts plus posture IK, hand trajectory, reach coverage, reach search, reach database, and legacy IK tests passed; probe and viewer release binaries rebuilt successfully.
- Root and collision regressions passed: IK modifies only the three waist and seven active-arm joints, root placement/height stays exact, and strict swept object/environment collision checks retain final-contact-only semantics.
- Real report: `build/g1-reaches/contact-anchored-coverage-report-v5.json`; all five fixtures processed 4,608/4,608 instances in under 17 seconds each and `search_integrity_passed` is true.
- Restored open-space coverage: 477 accepted (231 left, 246 right), 10 root-azimuth sectors, 0.0007471 m maximum accepted position error, 0.1177661 rad maximum approach error, 0.0048828 rad maximum orientation error, and 13.1638 seconds.
