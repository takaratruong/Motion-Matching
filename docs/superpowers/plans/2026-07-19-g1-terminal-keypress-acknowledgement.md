# G1 Terminal Keypress Acknowledgement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Print immediate proof of every terminal key received by the interactive G1 controller while keeping controls usable after unsupported input.

**Architecture:** Keep the strict `TerminalKeyBuffer` unchanged. Extend `TerminalInputReader` with an optional string event sink: accepted controls are fed and reported, unsupported batches are atomically rejected, reported as ignored, and reading continues. The manual demo supplies a flushed stdout sink; scripted and scored paths remain unchanged.

**Tech Stack:** Python 3.10+, `termios`, `pty`, `unittest`/pytest, existing `mm_sonic` terminal operator stack.

## Global Constraints

- Interactive accepted input prints messages such as `KEY W -> forward` and `KEY <SPACE> -> stand` before the next motion boundary.
- Unsupported input prints one safely escaped `KEY ... -> ignored` message and cannot terminate the reader.
- Unsupported batches remain atomic; an ANSI sequence cannot activate a valid letter inside it.
- CR and LF remain no-op commands but print an ignored acknowledgement.
- EOF, terminal I/O errors, and invalid UTF-8 retain the existing surfaced-failure behavior.
- Motion matching, command sampling, lookahead, SONIC physics, scripted runs, evidence formats, and scored paths do not change.

---

### Task 1: Resilient acknowledged terminal reader

**Files:**
- Modify: `tests/python/test_sonic_operator_terminal.py`
- Modify: `sonic/python/mm_sonic/operator_terminal.py`

**Interfaces:**
- Consumes: existing `TerminalKeyBuffer.feed(keys: str) -> None` atomic validation.
- Produces: `TerminalInputReader(..., event_sink: Callable[[str], None] | None = None, join_timeout: float = 2.0)`.
- Produces messages formatted as `KEY <label> -> <field-or-ignored>`.

- [ ] **Step 1: Write failing PTY regressions**

Add a predicate waiter and two tests to `TerminalInputReaderTests`:

```python
def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_reports_each_accepted_key(self) -> None:
    controller_fd, follower_fd = pty.openpty()
    self.addCleanup(os.close, controller_fd)
    self.addCleanup(os.close, follower_fd)
    buffer = TerminalKeyBuffer()
    events: list[str] = []

    with TerminalInputReader(follower_fd, buffer, event_sink=events.append):
        os.write(controller_fd, b"w ")
        self.assertEqual(
            _sample_until(buffer),
            OperatorState(forward=True, stand=True),
        )
        self.assertTrue(_wait_until(lambda: len(events) == 2))

    self.assertEqual(events, ["KEY W -> forward", "KEY <SPACE> -> stand"])


def test_reports_unsupported_sequence_and_keeps_reading(self) -> None:
    controller_fd, follower_fd = pty.openpty()
    self.addCleanup(os.close, controller_fd)
    self.addCleanup(os.close, follower_fd)
    buffer = TerminalKeyBuffer()
    events: list[str] = []

    with TerminalInputReader(follower_fd, buffer, event_sink=events.append):
        os.write(controller_fd, b"\x1b[A")
        self.assertTrue(_wait_until(lambda: len(events) == 1))
        self.assertEqual(buffer.sample(), OperatorState())
        os.write(controller_fd, b"w")
        self.assertEqual(_sample_until(buffer), OperatorState(forward=True))
        self.assertTrue(_wait_until(lambda: len(events) == 2))

    self.assertEqual(
        events,
        ["KEY <ESC>[A -> ignored", "KEY W -> forward"],
    )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error \
  /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/.venv/bin/python \
  -B -m unittest -v tests.python.test_sonic_operator_terminal
```

Expected: the new tests fail because `TerminalInputReader.__init__` does not accept `event_sink`.

- [ ] **Step 3: Implement the minimal event sink and continue-on-invalid behavior**

Add the import and rendering helpers:

```python
from collections.abc import Callable


def _key_label(character: str) -> str:
    if character == " ":
        return "<SPACE>"
    if character in _IGNORED:
        return "<ENTER>"
    if character == "\x1b":
        return "<ESC>"
    if character.isprintable():
        return character.upper()
    return ascii(character)[1:-1]


def _accepted_key_event(character: str) -> str:
    action = _KEY_TO_FIELD.get(character.lower(), "ignored")
    return f"KEY {_key_label(character)} -> {action}"


def _ignored_key_event(keys: str) -> str:
    return f"KEY {''.join(_key_label(character) for character in keys)} -> ignored"
```

Extend construction and store the sink:

```python
def __init__(
    self,
    fd: int,
    buffer: TerminalKeyBuffer,
    *,
    event_sink: Callable[[str], None] | None = None,
    join_timeout: float = 2.0,
) -> None:
    # retain the existing validation and fields
    self._event_sink = event_sink
```

Replace the feed at the reader boundary with:

```python
keys = data.decode("utf-8")
try:
    self._buffer.feed(keys)
except ContractError:
    if self._event_sink is not None:
        self._event_sink(_ignored_key_event(keys))
    continue
if self._event_sink is not None:
    for character in keys:
        self._event_sink(_accepted_key_event(character))
```

- [ ] **Step 4: Run the reader suite and verify GREEN**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error \
  /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/.venv/bin/python \
  -B -m unittest -v tests.python.test_sonic_operator_terminal
```

Expected: all terminal operator tests pass, including the existing EOF failure test.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/python/test_sonic_operator_terminal.py \
  sonic/python/mm_sonic/operator_terminal.py
git commit -m "fix: keep G1 terminal input reader alive"
```

### Task 2: Print acknowledgements in the interactive manual demo

**Files:**
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`

**Interfaces:**
- Consumes: Task 1 `event_sink: Callable[[str], None] | None`.
- Produces: `_print_terminal_event(event: str) -> None`, passed only by interactive mode.

- [ ] **Step 1: Write the failing print-sink test**

Import `redirect_stdout`, `StringIO`, and `_print_terminal_event`, then add:

```python
class TerminalEventPrintingTests(unittest.TestCase):
    def test_prints_one_complete_event_line(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            _print_terminal_event("KEY W -> forward")
        self.assertEqual(output.getvalue(), "KEY W -> forward\n")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error \
  /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/.venv/bin/python \
  -B -m unittest -v tests.python.test_sonic_manual_demo
```

Expected: collection fails because `_print_terminal_event` does not exist.

- [ ] **Step 3: Implement and wire the flushed event printer**

Add:

```python
def _print_terminal_event(event: str) -> None:
    print(event, flush=True)
```

Construct the interactive reader as:

```python
reader = (
    TerminalInputReader(
        sys.stdin.fileno(),
        key_buffer,
        event_sink=_print_terminal_event,
    )
    if namespace.mode == "interactive"
    else None
)
```

- [ ] **Step 4: Run focused and protected regression gates**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error \
  /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/.venv/bin/python \
  -B -m unittest -v \
  tests.python.test_sonic_operator_terminal \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_operator \
  tests.python.test_sonic_operator_runtime
```

Expected: all selected tests pass with no warnings or errors.

Then run the exact frozen manual/evidence gate used for this branch:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=sonic/python \
  PYTHONWARNINGS=error \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline/sonic/.venv/bin/python \
  -B -m unittest -v \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_manual_evidence
```

Expected: 52 tests pass with no warning; the formal four-chunk default remains
unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add tests/python/test_sonic_manual_demo.py \
  sonic/python/mm_sonic/manual_demo.py
git commit -m "feat: print G1 terminal key acknowledgements"
```

### Task 3: Replace and visibly verify the live driver

**Files:**
- Runtime artifact: `/home/ubuntu/g1-sonic-keypress-forward.png`
- Runtime artifact: `/home/ubuntu/g1-sonic-keypress-stand.png`

**Interfaces:**
- Consumes: `/home/ubuntu/drive-g1-sonic.sh 100000` and the committed Tasks 1-2.
- Produces: a persistent interactive terminal and MuJoCo viewer for the user.

- [ ] **Step 1: Cleanly stop only the current G1 manual-demo process**

Send SIGINT to the exact current manual-demo PID, wait for its owned MM, GEAR,
and simulator children to exit, and leave unrelated desktop processes intact.

```bash
pid=$(pgrep -f '^sonic/.venv/bin/python -u -m mm_sonic.manual_demo --mode interactive --onscreen --preload-chunks 2 --chunks 100000$')
test "$(printf '%s\n' "$pid" | wc -w)" -eq 1
kill -INT "$pid"
```

- [ ] **Step 2: Launch the replacement and wait for live boundaries**

Launch `/home/ubuntu/drive-g1-sonic.sh 100000` in a terminal titled
`TYPE KEYS HERE - G1 CONTROLS`. Wait until the terminal displays `LIVE` and at
least one `boundary` line before testing input.

```bash
DISPLAY=:1 gnome-terminal --title='TYPE KEYS HERE - G1 CONTROLS' -- \
  bash -lc '/home/ubuntu/drive-g1-sonic.sh 100000; exec bash'
```

- [ ] **Step 3: Verify visible receipt and command application**

With the control terminal focused, inject `W`; require both `KEY W -> forward`
and a later boundary with `vx=+0.50`. Capture
`/home/ubuntu/g1-sonic-keypress-forward.png`.

Inject Space; require both `KEY <SPACE> -> stand` and a later boundary with
`vx=+0.00`. Capture `/home/ubuntu/g1-sonic-keypress-stand.png`.

- [ ] **Step 4: Verify physics and handoff**

Require the manual-demo process, MM server, GEAR process, and MuJoCo viewer to
remain live. Inspect the current state log for root height and pelvis-up, check
that no fall marker exists, keep the control terminal above the viewer, and
tell the user to type only in that terminal.
