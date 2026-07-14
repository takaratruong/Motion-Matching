# Motion Matching & Code vs Data Driven Displacement

This repo contains the source code for all the demos from [this article](https://theorangeduck.com/page/code-vs-data-driven-displacement).

It also contains basic example implementations of Motion Matching and Learned Motion Matching in the style of [this paper](https://theorangeduck.com/page/learned-motion-matching).

# Installation

This demo uses [raylib](https://www.raylib.com/) and [raygui](https://github.com/raysan5/raygui) so you will need to first install those. Once installed, the demo itself is a pretty straight forward to make - just compile `controller.cpp`.

I've included a basic `Makefile` which you can use if you are using raylib on Windows. You may need to edit the paths in the `Makefile` but assuming default installation locations you can just run `Make`.

If you are on Linux or another platform you will probably have to hack this `Makefile` a bit.

# G1 tabletop interaction data replay

Gate 1 converts the complete GRAIL pickup-table corpus to the frozen 25 Hz G1 interaction schema, validates the published pack in Python, and replays the same binary artifacts through the headless C++ probe. Set up an isolated Python environment and fetch the source modalities with:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-interaction.txt
python -m resources.fetch_grail_pickup_table --output /path/to/GRAIL
```

Run the reproducible full-corpus gate with explicit local paths:

```bash
make gate1-interaction \
  GRAIL_PICKUP_ROOT=/path/to/GRAIL/data/pickup_table \
  G1_XML=/path/to/g1_29dof.xml \
  G1_INTERACTION_DIR=resources/g1_interaction
```

The target first runs the explicit safe interaction test suite, then builds with reviewed rejections enabled, validates the pack, applies the full-corpus Gate 1 assertions, and runs the C++ replay digest. Unknown rejection codes and unexpected exceptions abort publication.

The output directory contains exactly five reproducible files:

- `interaction_database.bin`: canonical 31-bone poses, interaction labels, object state, and clip ranges.
- `interaction_features.bin`: the database-only 71-D feature rows and normalization values.
- `manifest.json`: schema, skeleton, feature, provenance, clip, and count metadata.
- `evaluation_split.json`: deterministic object-disjoint database and held-out identities.
- `validation_report.json`: included/rejected clip totals, frames, numeric bounds, and exact rejection records.

`resources/g1_interaction/` is ignored generated output. Rebuild it with the command above rather than committing it. A different `G1_INTERACTION_DIR` is also treated as operator-owned generated output.

The first authorized full-corpus run must replace the pending cells below with the exact published values; no result has been inferred from the unit fixtures or diagnostic subset.

| Measurement | Source | Full-corpus result |
| --- | --- | --- |
| Source clips | `validation_report.json: source_clips` | Pending full-corpus run |
| Included clips | `validation_report.json: included_clips` | Pending full-corpus run |
| Rejected clips | `validation_report.json: rejected_clips` | Pending full-corpus run |
| Included frames | `validation_report.json: included_frames` | Pending full-corpus run |
| Database objects | length of `evaluation_split.json: database_objects` | Pending full-corpus run |
| Held-out objects | length of `evaluation_split.json: heldout_objects` | Pending full-corpus run |

Inspect the exact rejection code, stage, sequence, and message without modifying the pack:

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("resources/g1_interaction/validation_report.json").read_text()
)
print(json.dumps(report["rejections_by_code"], indent=2, sort_keys=True))
for rejection in report["rejections"]:
    print(rejection)
PY
```

Gate 1 replays data only; it does not yet make the character pick up an object.

# Playable G1 Tabletop Pickup

The desktop controller now combines the existing flat locomotion controller with one authored G1 tabletop pickup. Ordinary locomotion, rendering, and the ownership handoff run at a fixed 60 Hz. The interaction runtime advances through the deterministic integer scheduler at exactly 25 updates per second; it is not driven by wall-clock frame time.

Bootstrap the pinned local Raylib and Raygui dependencies, then build the diagnostic interaction pack from the local GRAIL pickup data:

```bash
make bootstrap-raylib
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DEMO_INTERACTION_LIMIT=5 \
make demo-interaction-pack
```

`GRAIL_ROOT`, `G1_XML`, and `DEMO_INTERACTION_LIMIT` override the source data, robot XML, and diagnostic source limit. `INTERACTION_DEMO_PACK` overrides the generated pack directory, whose default is the ignored `resources/g1_interaction/`. The builder publishes the pack transactionally and validates it at 25 Hz with one held-out object.

Run the complete reproducible gate on Linux with an X11 display:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-interaction
```

The gate uses the isolated `build/task12/interaction_query_probe_safe` parity tool and the explicit safe interaction test suite. `MM_INTERACTION_QUERY_PROBE` selects the executable used by the query-parity test; the safe Make targets set it automatically to that absolute build-directory path. The gate checks the display with a bounded `xdpyinfo` call, bounds the controller run to 45 seconds, and fails on a timeout or nonzero autodemo exit. The locomotion feature write is redirected through `MM_FEATURES_OUTPUT`, and the gate verifies that tracked `resources/features.bin` is unchanged.

After pickup, Carry, and Reset succeed, each evidence file is atomically replaced, with rollback on ordinary publication failure. The pair is not crash-atomic as a unit:

- `playable-evidence/pickup.jsonl` contains the deterministic per-render record.
- `playable-evidence/pickup.png` is captured from the final Carry frame before Reset.

`PLAYABLE_EVIDENCE_DIR`, `PLAYABLE_LOG_PATH`, `PLAYABLE_SCREENSHOT_PATH`, and `PLAYABLE_FEATURES_OUTPUT` override those gate paths. The evidence records the observed Carry mode as `recorded` when a certified carrying range matches or `layered` when the locomotion-plus-upper-body fallback is used; either mode preserves the attachment.

For standalone evidence validation, `PLAYABLE_LOG` and `PLAYABLE_SCREENSHOT` select the final JSONL and PNG to validate.

For manual play, build the controller and point it at the generated pack:

```bash
make controller
MM_INTERACTION_PACK=resources/g1_interaction ./controller
```

Use WASD or the left gamepad stick to move, the arrow keys or right stick to control the camera/facing, and the existing walk/strafe controls for locomotion. Near the highlighted target, press `F` (gamepad right-face-left) to request pickup. Press `X` (right-face-up) to cancel while cancellation is allowed, and `R` (right-face-right) to reset the held object. Invalid or out-of-range requests leave the object unmoved and report a diagnostic reason.

`MM_INTERACTION_PACK` overrides the runtime pack. `MM_FEATURES_OUTPUT` overrides the ordinary locomotion feature output while retaining the existing `resources/features.bin` default. `MM_INTERACTION_AUTODEMO=1` is reserved for deterministic evidence generation and requires both `MM_INTERACTION_LOG` and `MM_INTERACTION_SCREENSHOT`; their values must be nonempty and distinct, and both must have existing parent directories. Malformed paths, unavailable packs, unexpected runtime states, evidence I/O failures, window closure, and internal or external timeouts produce a nonzero exit.

The playable baseline selects one contiguous pickup for one authored tabletop target; staged interaction matching, multi-object selection, placement, shelves, and articulated doors/drawers are follow-on work.

# Web Demo

If you want to compile the web demo you will need to first [install emscripten](https://github.com/raysan5/raylib/wiki/Working-for-Web-%28HTML5%29). Then you should be able to (on Windows) run `emsdk_env` followed by `make PLATFORM=PLATFORM_WEB`. You then need to run `wasm-server.py`, and from there will be able to access `localhost:8080/controller.html` in your web browser which should contain the demo.

# Learned Motion Matching

Most of the code and logic you can find in `controller.cpp`, with the Motion Matching search itself in `database.h`. The structure of the code is very similar to the previously mentioned [paper](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf) but not identical in all respects. For example, it does not contain some of the briefly mentioned optimizations to the animation database storage and there are no tags used to disambiguate walking and running.

If you want to re-train the networks you need to look in the `resources` folder. First you will need to run `train_decompressor.py`. This will use `database.bin` and `features.bin` to produce `decompressor.bin`, which represents the trained decompressor network, and `latent.bin`, which represents the additional features learned for each frame in the database. It will dump also out some images and `.bvh` files you can use to examine the progress (as well as write Tensorboard logs to the `resources/runs` directory). Once the decompressor is trained and you have a well trained network and corresponding `latent.bin`, you can then train the stepper and the projector (at the same time) using `train_stepper.py` and `train_projector.py`. Both of these will also output networks (`stepper.bin` and `projector.bin`) as well as some images you can use to get a rough sense of the progress and accuracy.

The data required if you want to regenerate the animation database is from [this dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) which is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (unlike the code, which is licensed under MIT).

If you re-generate the database you will also need to re-generate the matching database `features.bin`, which is done every time you re-run the demo. Similarly if you change the weights or any other properties that affect the matching the database will need to be re-generated and the networks re-trained.
