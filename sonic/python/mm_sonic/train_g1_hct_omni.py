"""Register the HCT G1 task, then hand off to IsaacLab's stock RSL-RL trainer."""

from __future__ import annotations

import builtins
import os
import runpy
import sys

ISAACLAB_TRAIN = "/move/u/justingu/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
sys.path.insert(0, os.path.dirname(ISAACLAB_TRAIN))

_real_import = builtins.__import__
_registered = False


def _import_hook(name, *args, **kwargs):
    global _registered
    module = _real_import(name, *args, **kwargs)
    if not _registered and name.startswith("isaaclab_tasks"):
        _registered = True
        _real_import("mm_sonic.g1_hct_omni_env")
        print("[g1_hct_omni] registered Terrain-G1-HCT-Omni-v0", flush=True)
    return module


builtins.__import__ = _import_hook
sys.argv[0] = ISAACLAB_TRAIN
runpy.run_path(ISAACLAB_TRAIN, run_name="__main__")
