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
