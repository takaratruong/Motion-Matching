# Privileged Global Terrain Motion Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed greedy terrain viewer with a fully privileged,
globally planned, purely kinematic G1 motion oracle, then expose measured paths for
removing privilege and generating SONIC/diffusion-policy data.

**Architecture:** Four gated implementation plans build one system in dependency
order: an audited canonical corpus, a fragment-based flat synthesizer, a globally
planned terrain synthesizer, and a frozen benchmark/viewer/export layer. Each phase
must produce working, independently testable software and pass its exit gate before
the next phase begins.

**Tech Stack:** Python 3.10+, NumPy, SciPy 1.15.3, MuJoCo, Zarr 2.18, OpenUSD at the
import boundary, Shapely 2.1, Matplotlib/ImageIO/FFmpeg, unittest.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-08-01-privileged-global-terrain-motion-oracle-design.md`.
- Preserve the current dirty shared worktree. Never reset, stash, revert, or include
  unrelated prior edits in a commit.
- Existing untracked terrain modules are live prerequisites; do not create an
  isolated worktree until their ownership is resolved or their required state is
  intentionally captured.
- Commit only the files listed by the active task.
- Use test-driven development and run the exact focused test before and after each
  implementation step.
- Every test-only constructor/helper shown in a code block is implemented
  independently in `tests/python/terrain_oracle_test_utils.py` by the first task
  that uses it. It may construct production datatypes, but it may not call the
  production behavior under test. Whenever such a helper is added, include that
  utility file in the task's otherwise explicit `git add` command.
- Never modify Takara's or Justin's repositories.
- Never classify SONIC tracker rollouts as clean source kinematics.
- Do not proceed through a failed phase gate merely to obtain a viewer.

---

## Ordered plan set

1. [Phase 1: Audited canonical corpus](2026-08-01-terrain-oracle-01-audited-corpus.md)
   imports and audits clean Takara/BONES, Justin, c490 GRAIL, and G1 LAFAN data,
   validates complete symmetry, groups splits, and freezes coverage.
2. [Phase 2: Contact fragments and flat synthesis](2026-08-01-terrain-oracle-02-fragments-flat-synthesis.md)
   segments contact-to-contact motion, validates warp neighborhoods, searches
   complete routes, and proves smooth flat start/stop/turn/omnidirectional control.
3. [Phase 3: Privileged global terrain synthesis](2026-08-01-terrain-oracle-03-global-terrain-synthesis.md)
   extracts exact support surfaces, searches footholds and fragments globally,
   warps around contact events, solves multi-frame IK, and backtracks from an
   independent final audit.
4. [Phase 4: Benchmark, viewer, and export](2026-08-01-terrain-oracle-04-benchmark-viewer-export.md)
   freezes balanced supported/negative suites, renders visual evidence, connects the
   Switch-controller viewer, exports policy-safe commands, and runs privilege
   ablations.

## Stable cross-phase interfaces

| Interface | Defined | Consumers |
|---|---|---|
| `CanonicalTerrainMesh`, `TerrainBinding`, `CanonicalClip` | Phase 1 Task 1 | all later phases |
| `CorpusManifest`, `ClipRecord`, `MeshRecord` | Phase 1 Task 2 | fragment builder and all audits |
| `CoverageManifest`, `SplitManifest` | Phase 1 Task 8 | warp, terrain generation, benchmark |
| `FragmentRef`, `FragmentFeatures`, `WarpNeighborhood` | Phase 2 Tasks 1–3 | search and reconstruction |
| `FragmentBank` | Phase 2 Task 4 | flat and terrain search |
| `MotionRequest`, `LocalTrajectory` | Phase 2 Task 5 | every privileged/ablated provider |
| `MotionPlan`, `UnsupportedReason` | Phase 2 Task 6 | flat reconstruction |
| `TerrainWarpOverlay`, `TerrainMotionPlan` | Phase 3 Tasks 4–7 | global reconstruction |
| `ReconstructedMotion`, `SynthesisResult` | Phases 2–3 | audit, benchmark, viewer, export |
| `FinalAudit`, `BenchmarkResult` | Phases 3–4 | acceptance and research write-up |

## Privilege-removal ladder

All eight stages use the same robot-centred `MotionRequest` boundary and the same
frozen benchmark:

1. full route + complete mesh + true global state;
2. finite receding route + complete mesh + true global state;
3. finite route + exact robot-centred terrain + true global state;
4. finite route + partial robot-centred terrain + true global state;
5. finite route + partial terrain + estimated relative state;
6. local critically damped two-stick intent + partial terrain + no global
   odometry;
7. oracle-generated motion tracked by SONIC and learned by the diffusion policy;
8. optional removal of the runtime kinematic planner only if the learned policy
   absorbs the oracle behavior without losing benchmark quality.

Each stage gets a typed capability object containing only its permitted inputs.
Global state may never enter a stage that claims not to use it.

`SynthesisResult` has one fail-closed contract:

```python
@dataclass(frozen=True)
class UnsupportedReason:
    code: str
    message: str
    missing_transition: str | None
    violated_margin: str | None

@dataclass(frozen=True)
class SynthesisResult:
    supported: bool
    motion_path: Path | None
    plan_sha256: str | None
    audit_path: Path
    reason: UnsupportedReason | None
    safe_stop_motion_path: Path | None
    attempt_count: int
    forbidden_edge_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.supported:
            if (
                self.motion_path is None
                or self.plan_sha256 is None
                or self.reason is not None
            ):
                raise ContractError("supported result is incomplete")
        elif self.reason is None or self.safe_stop_motion_path is None:
            raise ContractError(
                "unsupported result requires a reason and audited safe stop"
            )
```

## Execution checkpoints

- [ ] **Checkpoint 1:** Complete Phase 1 and review the full corpus audit/coverage
  report and robot-mesh contact sheets.
- [ ] **Checkpoint 2:** Complete Phase 2 and review the symmetric flat route matrix
  and videos against the retained flat-only baseline.
- [ ] **Checkpoint 3:** Complete Phase 3 and review one audited route/video from
  every covered terrain stratum before creating the interactive frontend.
- [ ] **Checkpoint 4:** Freeze Phase 4 benchmarks, run supported and negative suites,
  inspect visual evidence, and record exact hashes/results.
- [ ] **Checkpoint 5:** Only after the privileged oracle passes, choose the next
  privilege-ablation stage or export clean targets for SONIC/noised collection.

## Completion command

At the end of every checkpoint, run:

```bash
ORACLE_PYTHON=${ORACLE_PYTHON:-/move/u/justingu/miniconda3/envs/isaac6_test/bin/python}
MUJOCO_GL=egl PYTHONPATH=sonic/python:. "$ORACLE_PYTHON" -B \
  -m unittest discover -s tests/python -p 'test_oracle_*.py' -v
git diff --check
git status --short
```

Review `git status --short` manually and confirm every staged path belongs to the
active task before committing. After Phase 3 creates `sonic/.oracle-venv`, invoke
the command with `ORACLE_PYTHON=sonic/.oracle-venv/bin/python`.
