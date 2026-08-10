# Arrow-key Driving Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Up/Down/Left/Right drive the interactive terrain viewer while preserving WASD and the existing stop/reset/exit behavior.

**Architecture:** Normalize `pynput` special-key objects at the listener boundary into the existing `w`, `s`, `a`, and `d` tokens. Keep `KeyboardCommandSource` and `CommandState` unchanged so keyboard aliases, gamepad priority, and motion runtime semantics remain stable.

**Tech Stack:** Python 3.11, `pynput.keyboard`, MuJoCo passive viewer, pytest, Ruff.

## Global Constraints

- Up or W means forward; Down or S means backward.
- Left or A steers left; Right or D steers right.
- Space, R, X, Q, and Escape retain their current behavior.
- Press and release must share one normalization function so arrow state cannot stick.
- No model, corpus, search, terrain, gamepad, or runtime-motion behavior changes.
- Restart the running viewer after verification because Python does not hot-reload the listener module.

---

### Task 1: Normalize special arrow keys in the interactive listener

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py:117-180`
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py:918-965`
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py:1831-1844`
- Test: `tests/python/test_hybrid_terrain_lmm_viewer.py:1145-1180`

**Interfaces:**
- Consumes: `pynput.keyboard.Key` members and printable `key.char` values.
- Produces: `_keyboard_command_token(key: object, keyboard_module: object) -> str | None`, returning existing internal tokens `w`, `s`, `a`, or `d` for arrows and otherwise returning `key.char`.
- Preserves: `KeyboardCommandSource.press`, `KeyboardCommandSource.release`, `handle_key_press`, and `CommandState.from_keyboard` semantics.

- [ ] **Step 1: Write the failing special-key normalization test**

```python
def test_arrow_keys_map_to_level_safe_drive_commands(self):
    keyboard = SimpleNamespace(
        Key=SimpleNamespace(
            up=object(), down=object(), left=object(), right=object(), esc=object()
        )
    )
    keys = KeyboardCommandSource()

    for special, expected in (
        (keyboard.Key.up, CommandState(speed=1.0)),
        (keyboard.Key.down, CommandState(speed=-1.0)),
        (keyboard.Key.left, CommandState(steering=1.0)),
        (keyboard.Key.right, CommandState(steering=-1.0)),
    ):
        token = viewer_module._keyboard_command_token(special, keyboard)
        keys.press(token)
        self.assertEqual(keys.snapshot()[0], expected)
        keys.release(token)
        self.assertEqual(keys.snapshot()[0], CommandState())
```

- [ ] **Step 2: Run the regression and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_viewer.py \
-k 'arrow_keys_map_to_level_safe_drive_commands'
```

Expected: fail because `_keyboard_command_token` does not exist.

- [ ] **Step 3: Implement the minimal normalization seam**

```python
def _keyboard_command_token(key: object, keyboard_module: object) -> str | None:
    arrows = {
        keyboard_module.Key.up: "w",
        keyboard_module.Key.down: "s",
        keyboard_module.Key.left: "a",
        keyboard_module.Key.right: "d",
    }
    return arrows.get(key, getattr(key, "char", None))
```

Use it symmetrically in `run_interactive`:

```python
def on_press(key: object) -> bool | None:
    escape = key == keyboard_module.Key.esc
    return handle_key_press(
        keys, _keyboard_command_token(key, keyboard_module), escape=escape
    )

def on_release(key: object) -> None:
    keys.release(_keyboard_command_token(key, keyboard_module))
```

Update the overlay instruction line to:

```python
"Up/Down speed | Left/Right steer | WASD aliases | Space stop | R reset | X/Esc exit"
```

- [ ] **Step 4: Add opposing-arrow and WASD compatibility assertions**

Extend the test to press Up+Down and Left+Right and assert `CommandState()` on each opposing pair. Retain the existing `test_keyboard_wasd_space_and_reset_are_level_safe` unchanged.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
tests/python/test_hybrid_terrain_lmm_viewer.py \
tests/python/test_full_walking_terrain_lmm_viewer.py
```

Expected: all collected viewer tests pass with zero failures.

- [ ] **Step 6: Run static verification**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff check \
sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
tests/python/test_hybrid_terrain_lmm_viewer.py
/home/ubuntu/miniconda3/envs/diffsim/bin/ruff format --check \
sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
tests/python/test_hybrid_terrain_lmm_viewer.py
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile \
sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 7: Commit the implementation**

```bash
git add \
  sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py \
  tests/python/test_hybrid_terrain_lmm_viewer.py
git commit -m "fix: drive terrain viewer with arrow keys"
```

- [ ] **Step 8: Restart and inspect the real viewer**

Stop only the exact existing viewer PID after confirming its command line, relaunch the same full-corpus/model/ramp command under `DISPLAY=:1`, confirm the MuJoCo window opens, and verify Up/Down/Left/Right through the real listener without removing the diagnostic label.
