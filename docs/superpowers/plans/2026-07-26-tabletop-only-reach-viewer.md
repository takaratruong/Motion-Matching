# Tabletop-Only Reach Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the reach coverage viewer using only the corrected tabletop SOMA reaches.

**Architecture:** Generate an isolated pack from the corrected tabletop review corpus and annotations. Validate its provenance and count before launching the unchanged inbound-only viewer.

**Tech Stack:** Python reach-pack builder, JSON manifest validation, C++/raylib reach viewer.

## Global Constraints

- Preserve `build/g1-reaches/reach-pack-v5`.
- Include 222 captured tabletop reaches and 222 mirrored reaches.
- Require every manifest sequence ID to begin with `tabletop_soma/`.
- Retain one-way 3.6-second inbound playback with a contact hold.

---

### Task 1: Build and Launch the Tabletop-Only Pack

**Files:**
- Create: `build/g1-reaches/tabletop-reach-pack-v2/`
- Verify: `build/g1-reaches/tabletop-reach-pack-v2/manifest.json`

**Interfaces:**
- Consumes: corrected review corpus `build/g1-reaches/tabletop-review-v2` and annotations `build/g1-reaches/tabletop-annotations-v2.json`
- Produces: a reach pack accepted by `g1_reach_coverage_viewer PACK`

- [ ] **Step 1: Build the isolated pack**

```bash
python3 -m resources.build_g1_reach_database \
  --review build/g1-reaches/tabletop-review-v2 \
  --annotations build/g1-reaches/tabletop-annotations-v2.json \
  --output build/g1-reaches/tabletop-reach-pack-v2
```

Expected: exit 0 and a manifest reporting 222 captured, 222 mirrored, and 444 total reaches.

- [ ] **Step 2: Validate count and provenance**

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("build/g1-reaches/tabletop-reach-pack-v2/manifest.json").read_text()
)
assert manifest["captured_reaches"] == 222
assert manifest["mirrored_reaches"] == 222
assert manifest["total_reaches"] == 444
assert all(
    record["sequence_id"].startswith("tabletop_soma/")
    for record in manifest["reach_records"]
)
print("validated 444 tabletop-only reaches")
PY
```

Expected: `validated 444 tabletop-only reaches`.

- [ ] **Step 3: Verify viewer regression tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
```

Expected: 16 tests pass.

- [ ] **Step 4: Replace the running viewer**

```bash
viewer_pid=$(pgrep -n -f '^./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v5$' || true)
if [ -n "$viewer_pid" ]; then kill "$viewer_pid"; fi
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  ./g1_reach_coverage_viewer build/g1-reaches/tabletop-reach-pack-v2
```

Expected: the viewer opens and its manifest load succeeds.

- [ ] **Step 5: Trigger search and inspect the HUD**

Focus the viewer, press Enter, and confirm all displayed source labels begin
with `tabletop_soma/`. Cycle with `]` and confirm each animation moves inbound
for 3.6 seconds and holds at contact.
