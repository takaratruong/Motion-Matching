# Published PFNN → G1 launcher

Run these commands from the repository root. Scene 6 is the default.

```bash
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher prepare
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher start --scene 6
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher switch --scene 5
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher status
PYTHONPATH=sonic/python python -m mm_sonic.published_pfnn_g1_launcher stop
```

| Scene | PFNN world | Key | Heightmap |
|---:|---:|---:|---|
| 1 | 0 | 1 | `hmap_000_smooth.txt` |
| 2 | 1 | 2 | `hmap_000_smooth.txt` |
| 3 | 2 | 3 | `hmap_004_smooth.txt` |
| 4 | 3 | 4 | `hmap_007_smooth.txt` |
| 5 | 4 | 5 | `hmap_013_smooth.txt` |
| 6 | 5 | 6 | `hmap_urban_001_smooth.txt` |

Controls remain in the published PFNN window: WASD movement, Shift to run,
Ctrl to strafe, arrow keys for the camera, and Q/E to zoom.

The launcher stores prepared artifacts under
`~/.cache/native-g1-pfnn/published-g1` and the active receipt, FIFO, and logs
under `~/.cache/native-g1-pfnn/published-g1-live`. `status` authenticates the
recorded PIDs before `stop` or `switch` acts on them.
