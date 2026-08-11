# Published PFNN to G1 Prototype Launcher

## Goal

Preserve the working published Daniel Holden PFNN → pinned GMR → G1 MuJoCo
prototype as a repeatable repository-owned tool. Scene 6 (urban) is the default.
Scenes 1–6 must always start the published PFNN and G1 viewer with the matching
heightmap.

## Scope

This is packaging only. It does not change PFNN inference, GMR retargeting, G1
poses, root motion, contacts, terrain logic, or controls.

Repository-owned files will include:

- the existing PFNN export patch and binary stream schema;
- the existing PFNN-to-G1 bridge, terrain-mesh converter, and G1 viewer;
- a six-entry scene manifest;
- one launcher with `prepare`, `start`, `switch`, `status`, and `stop` commands;
- a concise runbook and focused tests.

No daemon, web UI, training changes, or additional motion postprocessing will be
added.

## Scene Contract

| Scene | PFNN world | Key | Published heightmap |
|---:|---:|---:|---|
| 1 | 0 | 1 | `hmap_000_smooth.txt` |
| 2 | 1 | 2 | `hmap_000_smooth.txt` |
| 3 | 2 | 3 | `hmap_004_smooth.txt` |
| 4 | 3 | 4 | `hmap_007_smooth.txt` |
| 5 | 4 | 5 | `hmap_013_smooth.txt` |
| 6 | 5 | 6 | `hmap_urban_001_smooth.txt` |

The launcher generates or reuses the matching display mesh at the pinned GMR
morphology scale. The viewer rejects frames whose exported world ID differs
from the selected scene, preventing a source/viewer mismatch.

## Lifecycle

`start` validates dependencies and hashes, prepares a FIFO and logs in a stable
runtime directory, launches the bridge/viewer, launches the published exporter,
selects the requested PFNN world, and waits for the matching exported world ID.
It records exact PIDs and commands. `switch` performs a synchronized stop/start;
`stop` signals only authenticated recorded processes; `status` reports the scene,
PIDs, world ID, and queue health. A second `start` refuses to overwrite a live
instance.

The current manually launched prototype remains running until the packaged
launcher has passed its non-live tests and dry-run checks.

## Verification

Focused tests cover the six scene mappings, command construction, mismatch
rejection, exact-PID lifecycle records, terrain generation, and latest-frame
display behavior. A final smoke launch uses scene 6 and confirms the exporter
and viewer report world 5 with no display backlog before replacing the manual
process.
