# G1 IK Clamp Boundary-Convergence Amendment

> **Execution order:** This is a blocking prerequisite to the bounded-candidate plan. Use test-driven development and obtain independent review before changing that plan's execution base.

**Goal:** Make exact quaternion identity measure as exact zero, make requested/actual clamp diagnostics use the public original-baseline origin, and extend the authenticated near-cap frontier without weakening the exact cap or downstream result invariants.

**Root cause:** In the certified `d69d8fb` controller, right swing-lift candidate 26 reaches `ik_clamp_local_delta` with valid unit quaternions and cap `0x3eb33333`. Attempts 0–15 are valid but precise-over-cap. Under the then-current normalized-baseline debugger measurement, the 16th attempted index leaves precise angle `0.34999999456471464`, only `5.25e-10` rad above the cap; index 16 (the 17th evaluation) produces `0.34999999358418676`, below both caps. These values establish the exhaustion boundary; the corrected public-origin measurements may differ. The old fixed 16-evaluation frontier therefore returns false solely from exhaustion.

A second defect is ownership: the old function measures candidate acceptance from `normalized_baseline`, while independent callers measure from the original `baseline`. Repeated binary32 normalization is not idempotent. A fixed valid example was internally accepted/reported as `0x3ac0117d` but publicly remeasured as `0x3ac025cf`, above cap `0x3ac01ed9`.

## Frozen design

- Both inputs must pass the unchanged finite/unit validation before any identity rule.
- `ik_checked_quat_angle(q, q)` owns precise and rounded positive zero for exact component-bit identity. Invalid identical inputs still fail transactionally. Every nonidentity input retains the existing normalize/dot/`acos` path.
- Inside `ik_clamp_local_delta`, exact validated `baseline`/`desired` component-bit identity is an immediate no-op: publish the original baseline bits, requested/actual positive zero, and `limited = false`. This occurs before normalized-dot requested-angle construction.
- Normalize baseline and desired for shortest-arc/slerp construction, and materialize one canonical desired quaternion. Measure public requested correction exactly once with `ik_checked_quat_angle(original_baseline, canonical_desired)`. Its precise value owns limited/unlimited classification; its rounded value owns `requested_radians`.
- For an unlimited result, publish the canonical desired quaternion and set `actual_radians` to the same public rounded requested word. Requested/actual bits therefore remain equal, preserving downstream orientation validity.
- For a limited result, construct candidates with the unchanged normalized-space slerp and monotone downward progress, but measure every candidate using `ik_checked_quat_angle(original_baseline, candidate)`. Only a public precise/rounded dual-bounded candidate with positive `actual_radians` may be accepted and published. Its rounded public measurement owns `actual_radians`.
- The production frontier is a named 64 evaluations, indices `0..63`. This is an evidence-scoped frontier, not a universal convergence proof: the authentic case needs 17 and a deterministic million-input review found a valid case needing 50. Capacity 64 covers both with margin and matches the certified root-reach frontier scale.
- Finite exhaustion remains `false` with byte-unchanged output. It is not converted into a zero-correction limited success, because existing position/orientation validators require limited corrections to be positive. Any validation, normalization, trigonometric, binary32 commit, or public measurement failure also remains false and transactional.
- Do not add an epsilon, accept rounded equality alone, change any correction/terrain threshold, change `G1FootOrientationResult`/`G1LegSolveResult` validity, or introduce a new production status/schema field.

## Task 0: Regress and repair the public clamp frontier

**Files:**

- Modify: `ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`
- Verify only: `g1_ik.h`, `g1_ik_runtime.h`, `g1_ik_root_reach.cpp`, `controller.cpp`

1. Add exact-identity angle tests using the authentic baseline `(0x3f7fbb26,0x219469e4,0xbd3bb4cb,0x24bb68fa)` and the long-frontier baseline `(0x3e8d9fbf,0x3eef7400,0x3f277667,0x3f06b2a8)`. Require precise/rounded positive zero. Require identical zero, non-unit, NaN, and infinity inputs to remain false with both sentinels unchanged. Freeze representative nonidentity output words.
2. Add an exact same-input clamp regression with a non-idempotent valid quaternion and a cap below its false normalized self-angle. Require original baseline bits, requested/actual positive zero, and unlimited classification. Keep the existing scaled-identity and antipodal tests coherent with this ownership rule.
3. Add the authentic live clamp fixture:
   - baseline `(0x3f7fbb26,0x219469e4,0xbd3bb4cb,0x24bb68fa)`
   - desired `(0xbf7a925b,0x3d89e499,0x3c88eb81,0xbe455bf7)`
   - maximum `0x3eb33333`.
   Require production success, `limited=true`, positive actual, unit output, exact rounded/publication equality, and independent public precise/rounded values within cap.
4. Before production changes, run the strict test and record RED from the new identity/public-frontier assertions.
5. Under `G1_IK_ENABLE_TEST_SEAMS` only, expose a fixed 64-entry audit and explicit attempt-limit wrapper. Each valid limited attempt records interpolation angle, public-origin precise/rounded angle, acceptance, and count. Finite exhaustion records `finite_exhausted=true` and may publish the audit while leaving `IKClampResult` byte-unchanged; invalid math leaves both output and audit unchanged. No audit/limit API or identifying token survives no-seam preprocessing, symbols, strings, or a five-argument negative compile.
6. Freeze authentic causality: explicit limit 16 returns false with unchanged output, 16 valid strictly decreasing public-over-cap attempts, and `finite_exhausted=true`; limit 17 succeeds at index 16 without exhaustion. Require production capacity 64 to stop at the same index and equal the 17-result bit-for-bit.
7. Freeze the long-frontier fixture: cap `0x3c5c0d24`, baseline `(0x3e8d9fbf,0x3eef7400,0x3f277667,0x3f06b2a8)`, desired `(0xbeaf6a08,0x3e910963,0x3f43bf83,0x3eeee422)`. Require limit 49 to exhaust transactionally, limit 50 to succeed with a positive dual-bounded public correction, and production 64 to equal the 50-result. If public-origin ownership shifts the exact first success, freeze the newly observed first-success count only after independently proving the preceding prefix is valid/public-over-cap.
8. Add the public-origin mismatch fixture: cap `0x3ac01ed9`, baseline `(0x3e55c293,0x3f24d993,0x3f3bcbd7,0x3d767112)`, desired `(0x3d207104,0xbe34e99a,0xbf423675,0x3f2038fc)`. The old result `(0x3e559a8b,0x3f24cc43,0x3f3bdd91,0x3d73fe9e)` reported `0x3ac0117d` but publicly measures precise `0.0014659705526341572`, rounded `0x3ac025cf`, over cap. Require the corrected frontier to reject that attempt, continue, and publish only a positive result whose public rounded word equals `actual_radians` and whose public precise/rounded values are within cap.
9. Extend the existing baseline/desired/axis sweep: for every success, independently remeasure from the original baseline; require rounded/publication equality and both public caps. Unlimited results require requested/actual bit equality. Limited results require requested `>= maximum` and positive actual. Add transactional invalid/exhaustion checks.
10. Reconstruct the captured full two-bone call and one-ULP cap neighbors from these exact words:
   - root `be647f5e,3f081988,bedb4c3a`; middle `be7c58de,3eae106c,bee42d20`; end `be8e5e82,3d526040,bf056f33`; target `be7d214e,3de3ff94,bf0c0f74`
   - hinge `3f7e742d,bde0b8d5,3ae6531b`; root local `3f7fbb26,219469e4,bd3bb4cb,24bb68fa`; middle local `3f75488c,a45cdd35,254b5306,be9296b3`
   - root global `bf33ce15,3e1cc18d,3f30e3a5,bd9c113d`; middle global `bf31dd0a,bd518909,3f34b453,3e032664`; root-parent global `bf3bb8ec,3e202adb,3f2876aa,bd8d8941`
   - reach buffer from `g1_right_leg_config()` and maximum `0x3eb33333`.
   Require full solve success, limited classification, positive applied correction, and independent public cap compliance.
11. Run strict and established fast-caller `test_g1_ik` binaries linked to the same strict clearance/root-reach objects; require full pass and byte-identical `--parity`. Run no-seam production/negative ownership, explicit-limit negative compile, corrected explicit `if ...; then exit 1` preprocessing/nm/string privacy guards, and ASan/UBSan.
12. Rebuild the disposable mixed strict-kernel/fast-controller binary using the exact commands below. Run IK-off/on low-curb 32 frames twice; require normal exits, exactly 32 rows, byte-identical repeats, no global error, and unchanged old rows `0..15` against SHA-256 `b19ea7554ea1a24277d890c3ce99caa392753b28985e546353084b9b5b2f4421` (17 lines including header).

```bash
out=/tmp/g1-ik-clamp-boundary-cert/live
rm -rf "$out" && mkdir -p "$out/build" "$out/logs"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -frounding-math \
  -DNDEBUG -I. -c g1_clearance.cpp -o "$out/build/clearance.o"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -frounding-math \
  -DNDEBUG -I. -c g1_ik_root_reach.cpp -o "$out/build/root-reach.o"
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src -c controller.cpp \
  -o "$out/build/controller.o"
g++ "$out/build/controller.o" "$out/build/clearance.o" \
  "$out/build/root-reach.o" -o "$out/controller" \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
for repetition in 0 1; do
  for ik in 0 1; do
    DISPLAY=:1 MM_IK="$ik" "$out/controller" \
      --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
      --terrain-scene grail-curb-low --test-mode route \
      --test-route curb-forward --test-frames 32 \
      --test-heading forward --terrain-weight 4 \
      --log "$out/logs/low-curb-${ik}-${repetition}.csv"
  done
done
```

13. Run the same 32-frame command shape over shallow/standard stairs, low curb, and 5/10-degree ramps in both modes. This is regression/safety evidence, not Gate L release certification.
14. Obtain independent code/spec review, commit only `ik.h`, `tests/cpp/test_g1_ik.cpp`, and this plan, then update the bounded-candidate plan/brief execution base and regenerate its baseline artifacts.

**Non-goals:** No downstream result-validator/status change, candidate recovery, scene/route/schema/threshold change, live visualizer interaction, renderer change, or G1 mesh work.
