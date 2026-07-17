# G1 Sonic Sealed Replay Video Design

## Purpose

Create fast, inspectable video evidence for the already-qualified G1 Sonic Stage B run. The video must show the recorded robot motion and the key qualification statistics without changing or rerunning the sealed simulation.

This artifact is supporting visual evidence. The sealed machine-readable Stage B evidence remains authoritative. Every output must state that it is a deterministic post-run state replay, not a live screen recording and not proof of keyboard or hardware control.

## Frozen source

The sole source run is:

`sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/stage-b/stage-b-20260717T190656038623Z-b746af3f`

The renderer reads only:

- `dynamic/stream/scored-sim-logs/state.jsonl`, containing 600 recorded states at 50 Hz over 12 seconds;
- `scene/gear_scene.xml` and its run-local MuJoCo dependencies;
- `stage-b-evidence.json`, `manifest.json`, and `inventory.json` for displayed metrics and provenance.

Before and after rendering, the existing inventory verifier must confirm that the source run is unchanged.

## Output and placement

Generated files live in the run-set sibling directory `video-evidence/`, outside the sealed run root:

- `g1-sonic-stage-b-r13-replay.mp4`: 12-second, 50-fps H.264 video;
- `g1-sonic-stage-b-r13-montage.png`: six representative frames;
- `g1-sonic-stage-b-r13-video-metadata.json`: provenance, hashes, media properties, and replay disclaimer;
- `render_stage_b_replay.py`: the exact self-contained renderer used to produce the artifacts.

The sealed run inventory does not include these companion files and must not be regenerated to include them.

## Rendering architecture

The renderer performs a single streaming pass over `state.jsonl`:

1. Validate the state count, step/time sequence, finite numeric values, and MuJoCo `qpos` width before encoding.
2. Load the run-local scene in MuJoCo with an offscreen EGL renderer.
3. For each recorded state, assign `qpos` and `qvel`, call forward kinematics, and render two synchronized views:
   - a fixed world view that exposes the full trajectory;
   - a pelvis-tracking view that makes body motion and contacts visible.
4. Compose the two views into one 1280×720 frame with an unobtrusive overlay containing the run identifier, frame, simulation time, selected pass metrics, and the label `POST-RUN SEALED STATE REPLAY`.
5. Stream RGB frames to `ffmpeg` for deterministic 50-fps H.264 encoding, then create the six-frame montage.
6. Probe the MP4 and write a metadata sidecar containing source-file SHA-256 values, output SHA-256 values, renderer identity, dimensions, frame rate, frame count, duration, and the replay disclaimer.

The renderer writes temporary files in the output directory and atomically replaces final outputs only after all validation succeeds.

## Evidence boundaries

The video must not obscure unfavorable evidence. The overlay includes the recorded 3.813 m horizontal pelvis drift alongside the Stage B pass state. It may also show the 12.0-second active duration, absence of forbidden contacts, minimum pelvis height, minimum pelvis-up dot product, joint RMSE, and orientation RMS.

The video does not establish real-time control, physical-robot deployment, or operator drivability. Those require a separate interactive-control implementation and qualification cycle. This task ends with visual evidence for the sealed Stage B simulation only.

## Failure handling

Rendering fails closed if any source hash changes during the job, inventory verification fails, the state stream is incomplete or non-finite, MuJoCo cannot load the run-local scene, encoding fails, or the encoded media does not match the required duration, rate, frame count, resolution, and pixel format. Partial final artifacts are removed or left only under temporary names.

If EGL is unavailable, diagnose the rendering backend and use another existing headless MuJoCo backend only if it produces the same validated state replay. A statistics-only video is the last fallback and must be labeled as such.

## Acceptance gates

The task is complete only when all of the following hold:

- the source inventory verifies before and after rendering;
- exactly 600 recorded states become exactly 600 video frames at 50 fps;
- `ffprobe` reports 1280×720 H.264/yuv420p and 12.0 seconds;
- the first, middle, and final montage frames are visually inspected for a loaded robot, visible terrain, synchronized labels, and plausible pose progression;
- metadata contains and independently verifies the source and artifact SHA-256 values;
- the replay disclaimer is visible in the video and recorded in metadata;
- no process from the render job remains running.
