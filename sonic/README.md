# Motion Matching to GEAR-SONIC

This directory is an isolated integration package for driving the pinned
GEAR-SONIC deployment from the motion-matching runtime. It does not vendor or
modify GEAR-SONIC, model checkpoints, terrain artifacts, generated references,
or run evidence.

Every run must supply the GEAR checkout, policy checkpoint, observation
configuration, optional encoder checkpoint, terrain directory, and source G1
MJCF explicitly. `mm_sonic.external` resolves and validates those paths,
confines outputs away from input trees, verifies the checkout against
`configs/gear_sonic.lock.json`, and records deterministic SHA-256 identities.

Python packaging is rooted here rather than at the repository root. A local
environment may be created at `sonic/.venv`; builds and run artifacts belong in
the ignored `sonic/build` and `sonic/runs` directories.

```bash
python -m venv sonic/.venv
sonic/.venv/bin/pip install -e 'sonic[integration]'
PYTHONPATH=sonic/python sonic/.venv/bin/python -m unittest \
  tests.python.test_sonic_external -v
```

The `integration` extra installs the three pinned upstream runtime dependencies
(`scipy==1.15.3`, `PyYAML==6.0.3`, `cyclonedds==0.10.2`) alongside `mujoco` and
`pyzmq`. It intentionally does **not** declare a PyPI `unitree_sdk2py`: the
pinned Unitree Python SDK is resolved from
`external_dependencies/unitree_sdk2_python` inside the authenticated
`--gear-checkout`, so callers never supply a Unitree `PYTHONPATH` and the
checkout is never mutated.

## Stage A integration gate

The Stage A CLI is non-interactive. All external inputs and the isolated output
root are mandatory command-line paths; `SONIC_*` environment variables are not
used as input defaults. `--encoder` is optional only for policies whose
authenticated observation contract has no encoder.

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli preflight \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs

PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-a \
  --mode known-good-stream \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs
```

The three Stage A modes execute ordered prefixes of the registered seven-gate
contract. Only `known-good-stream` can create a complete Stage A pass: it binds
the file dynamic metric, stream delivery audit, and stream dynamic result to
the same policy, encoder, observation configuration, external commit, model,
generated flat scene, initial qpos, and canonical reference hashes. Later
stages must call `mm_sonic.metrics.validate_stage_a_prerequisite` on the
retained `stage-a-evidence.json`; a file-only result is intentionally rejected.
The validator authenticates the sealed inventory first, requires the evidence
registry hash to match the current canonical `stage_a.json`, and reconstructs
the exact raw `stage-a`/`known-good-stream` option set plus its sealed canonical
invocation cwd. Only argv-relative tokens are resolved against that cwd before
checking external paths, output-root placement, safe environment, ordered
passing gates, literal
per-gate hash claims, primary output digests, merged identity, metrics,
top-level outputs, and terminal `complete`/pass outcome. The stream-delivery
primary output is canonicalized again and its embedded delivery-audit digest is
recomputed after removing only the digest and publication-summary fields. A
stale registry or a self-consistent but forged evidence claim is not a
prerequisite.

Exit status is `0` for the requested completed gate prefix, `2` for a
configuration or integration failure, `3` for a scientific failure, and `4`
when a required external checkpoint, Git-LFS payload, dependency, execution
provider, or GPU is absent. Exit `4` writes immutable, result-free `not_run`
evidence whenever the explicit inputs make safe output creation possible. It
is never a feasibility result. Raw runs remain under ignored `sonic/runs`.

### Known-good timing and scoring contract

The pinned GEAR policy runs at wall-clock 50 Hz while the official MuJoCo
backend advances independently at 500 Hz. Stage A therefore never derives a
policy-row count or target/state alignment from simulator-step counts.

Both known-good modes cold-start GEAR only through authenticated
`WAIT_FOR_CONTROL`. File mode resets frame zero and arms playback there. Stream
mode enables ZMQ there and archives four independently labelled preload
transcripts: one `readiness` publication for frame 0, 22 `logical`
publications covering frames `1..440`, one `padding` publication for transport
indices `441..486`, and one `receipt_fence` publication at index 487. Padding
and the receipt fence repeat canonical pose 440 and are never scored. Every
publication requires exact, separately ranged pinned Start, processing, merge,
and `*** End of ZMQ decoding processing ***` lines. Arbitrary pinned decoder
diagnostics may occur between the merge and End lines and are SHA-256 bound.
Run-local CSV projection renders every binary32 value as an exact promoted
binary64 decimal, so GEAR's `std::stod` file path and the stream payload receive
identical numeric values rather than merely values that round back to binary32.
Publication N+1's authenticated Start-through-End completion event, rather
than its boundary snapshot alone, is the causal fence for publication N.

The `ZMQ STREAMING MODE: ENABLED` line alone is not preparation. Immediately
after its raw UTF-8 byte boundary, Stage A sends one `qe` PTY write and requires
the exact ordered acknowledgements `Delta heading left: 0.1 rad` and
`Delta heading right: 0 rad`. This fences the handler's reset tail and restores
net heading delta to zero before frame 0 can be published. GEAR stdout is
written and flushed to `dynamic/stream/gear.stdout` before those identical raw
bytes enter the stdout-only observer (stderr is archived separately), so every
recorded offset is an archive byte offset, including non-ASCII or split startup
output. Before CONTROL, the auditor
reopens that archive and writes
`dynamic/stream/preload-consumer-transcript.json`, binding the post-enable
fence and all 25 ordered sets of independent event-byte ranges to a frozen
stdout-prefix length and SHA-256. Every intervening region and the required
decoder tail after the final merge line are also range-hashed, while remaining
free to contain the pinned source's diagnostics. Later CONTROL output may
append to stdout without changing that authenticated prefix.

Before CONTROL, Stage A proves the target, `q.csv`, and `base_quat.csv` logs
contain zero data rows. It stops GEAR in WAIT, resets a distinct scored MuJoCo
epoch, and advances one pre-CONTROL step to publish fresh LowState. The reset
reuses the already authenticated same-scene simulator backend and DDS publisher
while rotating the physical log epoch; rebuilding the process-global Unitree
channel at this boundary is forbidden. It then
resumes once, activates CONTROL without stopping on the transition marker, and
advances physics in small increments until the target log is the exact
canonical 441-row sequence `0..440`. Overshoot, omission, duplication,
truncation, inode replacement, timeout, or a post-stop row fails integration.

Tracking uses the 441 same-CONTROL-tick GEAR `q.csv` and `base_quat.csv` rows,
after validating exact pinned headers, indices, shared timestamp prefixes,
strictly increasing monotonic time, joint permutation, finite values, and unit
base quaternions. MuJoCo state/contact logs remain physical cadence evidence,
not policy-aligned tracking rows. Evidence reports the one-step pre-CONTROL
prime, CONTROL-active drive steps/duration, and total scored-epoch
steps/duration separately; retained state/contact row counts cover the full
scored log epoch, including the prime.

## Stage B motion-matching qualification

Stage B replaces the known-good pose input with the registered flat
motion-matching route. It accepts only an immutable passing
`known-good-stream` Stage A evidence file whose inventory, invocation,
identities, gate outputs, and registry are revalidated against the current
code. It then executes the same four prerequisite gates, constructs the
601-frame motion-matching reference, transports frames `0..600` through the
pinned GEAR ZMQ decoder, and scores the official G1 SONIC dynamics.

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m mm_sonic.cli stage-b \
  --stage-a-evidence /read-only/stage-a-evidence.json \
  --gear-checkout /read-only/GR00T-WholeBodyControl \
  --policy /read-only/model_decoder.onnx \
  --observation-config /read-only/observation_config.yaml \
  --encoder /read-only/model_encoder.onnx \
  --source-mjcf /read-only/g1_29dof.xml \
  --terrain-dir /read-only/g1_terrain \
  --output-root sonic/runs/stage-b
```

The registered contract in `configs/experiments/stage_b.json` is exact: 30
accepted 20-frame commands, 601 authoritative target rows, and 12.000 seconds
of CONTROL-active physics. At the official 0.005-second simulator step this is
2,400 CONTROL steps. A one-step scored prime makes the retained epoch 2,401
steps, with exactly 2,401 contact rows and 600 state rows at 50 Hz. Complete but
truncated JSONL is rejected by validating every step number and sampling time,
not merely by counting syntactically valid records.

GEAR remains a wall-clock 50 Hz controller. For Stage B only, the simulator
adapter omits its redundant per-step wall-clock sleep and advances physics in
row-paced batches. The registered 16-row control lead ensures all 2,400
CONTROL steps finish before frame 600 is emitted. As soon as a newline-complete
601st target row exists, the process group is stopped before the full target
audit; early terminal arrival, overshoot, content mismatch, inode replacement,
or any post-stop write fails integration. The terminal fence records simulator
counters immediately before the stop, requires an identical post-stop and
final snapshot, and proves that requested, CONTROL-active, and required steps
were already equal before GEAR stopped. No simulator step may be used to fill
the duration after that boundary. Stage A retains its default paced simulator
behavior.

The generated run-local scene is bound semantically, not by a path-sensitive
digest alone. Stage B verifies the complete generated and included XML against
the authenticated Stage A scene after normalizing only the run-local include
path, and binds the registered allowed feet, forbidden body groups, and terrain
geom IDs. A pass additionally requires the exact command/frame/duration/log
coverage checks, no forbidden contacts, pelvis height and uprightness limits,
and joint/pelvis tracking ratios relative to the authenticated known-good
baseline. Secondary drift, clearance, scuff, impulse, and timing measurements
remain diagnostics and cannot override those registered primary gates.
