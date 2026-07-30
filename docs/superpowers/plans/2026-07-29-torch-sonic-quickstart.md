# Torch Motion Matching + SONIC Quickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a self-contained root-level guide for launching the in-process Torch motion matcher with the SONIC tracker.

**Architecture:** Add one documentation entry point at the repository root. It links the exact qualified cluster command to a portable path-template command and keeps terrain and the C++ matcher explicitly outside the workflow.

**Tech Stack:** Markdown, Bash, Git, Python unittest

## Global Constraints

- Document branch `research/g1-low-latency-driver`.
- Use the isolated `sonic/.torch-mm-venv`.
- Use the authenticated GEAR checkout `simulation-lowstate-wait-v6`.
- Require the flat scene and terrain weight zero.
- Do not require or launch `mm_chunk_server`.

---

### Task 1: Publish and validate the quickstart

**Files:**
- Create: `TORCH_SONIC_QUICKSTART.md`
- Modify: `sonic/README.md`

**Interfaces:**
- Consumes: the implemented CLI in `mm_sonic.manual_demo` and the qualified external asset paths.
- Produces: a root-level entry point for cluster and portable launches.

- [ ] **Step 1: Write the root quickstart**

Include architecture, prerequisites, checkout, environment verification,
focused tests, the exact qualified launch, a portable path template, controls,
expected readiness lines, cleanup, and failure diagnosis.

- [ ] **Step 2: Link it from the subsystem README**

Add this sentence under the existing Torch matcher heading:

```markdown
For a clone-to-launch guide, see [`TORCH_SONIC_QUICKSTART.md`](../TORCH_SONIC_QUICKSTART.md).
```

- [ ] **Step 3: Validate documented paths and commands**

Run:

```bash
test -d /home/ubuntu/Downloads/takara_walk_50hz.npz_v0
test -x sonic/.torch-mm-venv/bin/python
test -x /home/ubuntu/projects/gear-sonic-worktrees/simulation-lowstate-wait-v6/gear_sonic_deploy/target/release/g1_deploy_onnx_ref
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -B -m unittest -q \
  tests.python.test_sonic_torch_motion_data \
  tests.python.test_sonic_torch_motion_features \
  tests.python.test_sonic_torch_motion_search \
  tests.python.test_sonic_torch_motion_matcher \
  tests.python.test_sonic_torch_motion_sonic \
  tests.python.test_sonic_manual_demo
git diff --check
```

Expected: every path check exits zero, all focused tests pass, and
`git diff --check` is silent.

- [ ] **Step 4: Commit and push**

```bash
git add TORCH_SONIC_QUICKSTART.md sonic/README.md
git commit -m "docs: add Torch SONIC quickstart"
git push checkpoint research/g1-low-latency-driver
```

