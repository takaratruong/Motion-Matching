# Native-G1 PFNN terrain demo

Run from the repository worktree:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.terrain_pfnn_viewer
```

Use `W`, `A`, `S`, and `D` to drive and `Esc` to close the viewer. The default
artifact is the 16-frame recurrently fine-tuned classic PFNN trained on the
retargeted released-PFNN locomotion slice plus all GRAIL terrain families. The
default course has a flat run-up and then the scaled, continuous height function
fitted to `WalkingUpSteps08_000`; the same function drives both the rendered mesh
and the runtime terrain probes.

The generated artifacts used by the default command are:

- `sonic/runs/native-g1-pfnn/expanded/model-mixed-filtered-rollout16-final/best.pt`
- `sonic/runs/native-g1-pfnn/expanded/mixed-corpus-filtered/manifest.json`
- `sonic/runs/native-g1-pfnn/expanded/vertical-corpus/terrain/WalkingUpSteps08_000__01550_01675.npz`

For a headless traversal check:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.terrain_pfnn_viewer \
  --no-viewer --smoke-steps 240 --trace-every 60
```
