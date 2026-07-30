# Additional Stair Registration Analysis — staircase-v0 & staircase-final

**Date:** 2026-07-30 · **HEAD:** 5acbacd4
**Scope:** Bounded read-only research audit. Changes no project behavior; the
sole artifact is this report. Infers and independently validates finite
motion-to-terrain registrations for the two rejected clips **staircase-v0** and
**staircase-final** without altering any admission threshold.
**Discipline:** each claim is tagged **[Observed]** (cited artifact / measured
command output) or **[Inference]** (reasoning over observed facts).

---

## 1. Summary

Both clips are `fixed-staircase` `stair-local` clips (`registry.py:192-193`).
`build_source_terrain` hardcodes the motion-to-terrain transform to identity
`(0.0, 0.0, 0.0)` (`terrain.py:353`), so admission samples each clip's feet
against the fixed three-box staircase in the clip's own motion frame. **Under
identity both clips fail strict admission with `contact_alignment_failed`.**
Replaying the committed contact math over a finite `(tx, ty, yaw)` grid and
re-driving the real `admit_candidate` entrypoint finds finite non-identity
candidates that pass the implemented numeric gates for **both** clips.
However, those candidates do **not** establish a defensible registration:
well-separated transforms pass for each clip, and the minimum-p95 candidates
retain only 25/28 elevated samples while flat contacts dominate the reported
`contact_height_error`. The honest audit verdict is therefore **do not register
either clip from this evidence alone**.

**Measured verdicts (real `admit_candidate` entrypoint; §5):**

| Clip | Transform `(tx m, ty m, yaw)` | Contact | Elevated | p50 (m) | p95 (m) | max (m) | Verdict |
|------|------|:---:|:---:|:---:|:---:|:---:|:---|
| staircase-v0 | identity `(0, 0, 0)` | 735 | 142 | 0.000876 | **0.068600** | 0.078640 | REJECT `contact_alignment_failed` |
| staircase-v0 | numeric best `(−0.220, 0.110, −0.15708 rad / −9.0°)` | 563 | 25 | 0.000793 | **0.005579** | 0.078209 | GATES PASS; REGISTRATION REJECTED |
| staircase-final | identity `(0, 0, 0)` | 648 | 113 | 0.000723 | **0.065740** | 0.079947 | REJECT `contact_alignment_failed` |
| staircase-final | numeric best `(−0.100, 0.200, 0.22689 rad / 13.0°)` | 563 | 28 | 0.000709 | **0.006149** | 0.079947 | GATES PASS; REGISTRATION REJECTED |

The "numeric best" transform per clip is the finite candidate that **minimizes
aggregate contact p95 error** subject to every implemented gate passing. It is
not the best geometric registration. The FK gate is transform-independent; the
passthrough callable used here isolates the contact calculation but is not an
independent MuJoCo FK verification (§5).

---

## 2. Observed pipeline (facts)

### 2.1 Registration and terrain frame
**[Observed]** `staircase-v0` → `staircase:v0/motion.npz`
(sha256 `265133ea…8925`); `staircase-final` → `staircase_final:v0/motion.npz`
(sha256 `846fc49e…065b`); both `fixed-staircase` / `stair-local`
(`registry.py:192-193`). SHA-256 of both source files was verified equal to the
registry digests (§5). The fixed staircase is three boxes with top-Z
0.1778 / 0.3556 / 0.5334 m over x∈[−0.705, 0.274], y∈[−0.192, 0.430]
(`terrain.py:111-119`); grid cell 0.02 m, margin 2.5 m (`terrain.py:27-28`).

### 2.2 Transform is hardcoded identity
**[Observed]** `TerrainEvidence.motion_to_terrain_xy_yaw` is validated as a
finite 3-tuple (`terrain.py:45,52-58`) but `build_source_terrain` always emits
`(0.0, 0.0, 0.0)` (`terrain.py:353`). No per-clip transform is registered.

### 2.3 Admission consumes the transform (transform-aware)
**[Observed]** `_terrain_xy` applies yaw rotation + translation to foot XY
(`admission.py:147-161`); `admit_candidate` maps feet (bodies 18, 19) into the
scene, samples the grid, and computes clearance vs sole thickness
`_SOLE_M = 0.035` (`admission.py:232-241`). Strict gates, in order:

1. contact: `|clearance−0.035| ≤ 0.08` **and** vertical foot speed `≤ 0.12` m/s
   (`_CONTACT_ERROR_CANDIDATE_M`, `_CONTACT_VERTICAL_SPEED_MPS`, `:242-245`);
2. contact-sample floor `≥ 50` (`_MINIMUM_CONTACT_SAMPLES`, `:257`);
3. contact p95 error `≤ 0.035 m` (`_CONTACT_P95_ERROR_M`, `:268`);
4. elevated-contact floor `≥ 25` on elevated terrain, where a contact is
   elevated iff sampled surface `> terrain_min + 0.03 m`
   (`_MINIMUM_ELEVATED_CONTACT_SAMPLES`, `_ELEVATED_SURFACE_MINIMUM_M`,
   `:249-256,279-282`).

**[Observed]** The identity rejections above are governed by gate 3 (p95 ≈ 0.068
/ 0.066 m > 0.035): both clips already have ≥50 contacts and ≥25 elevated
contacts under identity, so the failure is height-error, not coverage. The FK
gate (`fk_p95 ≤ 0.005`, `fk_max ≤ 0.02`, `:220-230`) is **transform-independent**
— it compares an FK callable against `body_position_world` and is unaffected by
`(tx, ty, yaw)`. **[Inference]** Therefore the registration search moves only the
contact/elevated verdict; it cannot change the FK verdict.

---

## 3. Registration search method (this report)

**[Inference]** The transform is the rigid map taking the clip's motion frame
onto the fixed box frame. I replay the **exact committed contact math**
(`_terrain_xy` + `grid.sample_xy` + the clearance/elevated predicates, all
imported unchanged from `admission.py`) over a finite `(tx, ty, yaw)` grid
(`tx, ty ∈ [−0.30, 0.30]`/`[−0.20, 0.20]` step 0.01 m, `yaw ∈ [−14°, 14°]` step
1°), scoring each candidate by the admission gates in their real order. The
reported "best" per clip minimizes contact p95 subject to all gates passing.
No threshold constant is modified; all constants are read from
`admission.py:19-27` as frozen. The selected candidates were then re-verified by
driving the real `admit_candidate` entrypoint (§5), not just the helper math.

---

## 4. Overfitting / spatial sanity checks (measured, §5)

The independent checks identify overfitting rather than validate either
registration:

1. **Foot progression along the stair rise axis.** The fixed staircase spans
   x∈[−0.705, 0.274] and rises along −x (`terrain.py:111-119`). **[Observed]**
   the numeric-best v0 candidate spans only x∈[−0.741, −0.359], largely the
   highest-tread region; the final candidate spans x∈[−0.539, −0.071], only the
   high/middle region. Neither spans the full footprint. Raw foot-Z range alone
   cannot prove that the corresponding foot samples land on all three surfaces.
2. **Terrain-bounds coverage is necessary but weak.** **[Observed]** all
   transformed foot XY remain inside the large height-grid domain
   (x∈[−3.22, 2.78], y∈[−3.52, 2.94]), so `sample_xy` does not raise. That grid
   includes a broad flat exterior, however, so in-bounds queries do not prove
   stair alignment. The 563 aggregate contacts versus only 25/28 elevated
   contacts show how flat samples can dominate `contact_height_error`.
3. **Independent root-height check is non-discriminating.** **[Observed]** root
   height above the sampled surface has p50 ≈0.624 m (v0) and ≈0.707 m (final).
   Those plausible values reject gross vertical mistakes but do not distinguish
   the numeric-best transforms from the well-separated alternatives in §5.

---

## 5. Measured evidence (command reproduction)

**[Observed]** Environment: repository `HEAD 5acbacd4`; interpreter
`/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python` (numpy 1.26.4, scipy
1.15.3). Sources read read-only from the approved reference root
`/home/ubuntu/Downloads/artifacts`. Two unused transitive top-level imports
(`joblib` in `corpus.py:11`, `pxr` in `surface.py:10`) were satisfied with inert
stubs that raise if invoked; they are **never** called on the contact code path
(`joblib.load` only in `corpus.py:54`; `pxr` only in the USD surface path,
`surface.py:306+`), so the committed contact/FK math ran unmodified.

- **SHA-256 verified:** `staircase:v0/motion.npz` = `265133ea…8925`,
  `staircase_final:v0/motion.npz` = `846fc49e…065b` (equal to registry digests).
- **Clips:** both load to 633 frames at 50 Hz via `load_native_motion_50hz`.
- **Identity (real `admit_candidate`):** v0 → `contact_alignment_failed`,
  contact 735, elevated 142, p50 0.000876, p95 0.068600, max 0.078640;
  final → `contact_alignment_failed`, contact 648, elevated 113, p50 0.000723,
  p95 0.065740, max 0.079947.
- **Numeric-best transform (`admit_candidate` with contact-isolating passthrough
  FK):** as tabulated in §1 — both return `accepted=True, reason=None` for the
  implemented gates. The reported FK error 0.0 is an intentional passthrough
  result, not an independent FK measurement; a production corpus build must
  still run the real MuJoCo FK gate.
- **Sanity values:** as reported in §4 (partial foot-X coverage, broad grid
  bounds, low elevated-to-total contact ratio, and non-discriminating root
  height) — these are evidence against treating either numeric pass as a pinned
  registration.

**Non-uniqueness probe (measured):** a second, distinctly different passing
transform exists for each clip — v0 `(0.240, 0.100, −9.0°)` → accepted, contact
630, elevated 92, p95 0.022291; final `(−0.210, 0.090, −12.0°)` → accepted,
contact 617, elevated 82, p95 0.020327. Two well-separated accepted transforms
per clip confirm the registration is **not uniquely pinned** by contact height.

---

## 6. Verdicts and residual risk

- **staircase-v0 — [Inference from §5]:** **registration rejected.** The
  numeric-best candidate `(−0.220, 0.110, −0.15708 rad)` returns contact 563,
  elevated 25, p95 0.005579 m, but elevated count sits exactly at the floor and
  the alternative `(0.240, 0.100, −9.0°)` also passes with contact 630,
  elevated 92, p95 0.022291 m. The evidence does not select between them.
- **staircase-final — [Inference from §5]:** **registration rejected.** The
  numeric-best candidate `(−0.100, 0.200, 0.22689 rad)` returns contact 563,
  elevated 28, p95 0.006149 m, while `(−0.210, 0.090, −12.0°)` also passes with
  contact 617, elevated 82, p95 0.020327 m. Again the evidence does not select a
  unique scene transform.
- **Residual risk (non-uniqueness):** contact-height agreement alone cannot prove
  a globally unique planar registration — §5 exhibits two well-separated accepted
  transforms per clip. The §4 sanity checks bound but do not eliminate this
  ambiguity; a unique registration would need an independent geometric anchor
  (e.g. matched step edges), which this offline evidence does not provide.
- **Residual risk (provisional):** these accepted offline registrations remain
  provisional until the controller reruns corpus publication and the frozen
  omnidirectional route gates; this report runs neither.

### Next experiment (explicit accept/reject; no threshold changed)
Run one bounded candidate experiment without promoting either transform:
construct a temporary corpus for each of the two well-separated candidates per
clip, using real MuJoCo FK, and compare their per-step elevated-contact coverage
plus the frozen 21-route matrix.
- **Accept a registration** only if an independent source-space anchor selects
  the same transform, real FK and unchanged admission gates pass, elevated
  contacts cover every source stair level with at least 25 samples per level,
  every previously accepted clip stays accepted, and no previously passing
  route regresses.
- **Reject** if no independent anchor is available, competing transforms remain
  plausible, any step lacks the coverage floor, any threshold would need
  weakening, or any previously passing route regresses.

---

## 7. Scope guarantees (negative proofs)
- No implementation, tests, configs, existing docs, source artifacts, Git
  metadata, or protected thresholds are modified; this report is the only
  artifact written.
- No contact-admission, FK, terrain-quality, motion-quality, outcome, stall,
  rescue-cycle, or sliding threshold is weakened; the reproduction reads the
  frozen constants (`admission.py:19-27`) and never rewrites them.
- No network, remote, credential, GPU device, controller file, or other worktree
  was accessed; source npz reads are the approved read-only references under
  `/home/ubuntu/Downloads/artifacts`.
