# Compact trajectory pose fix report

## RED

The new synthetic hand-trajectory regression begins with a valid full
database, clears every vector deliberately skipped by the compact loader,
and runs trajectory selection followed by shaping. Before the production
change, the focused command compiled but the binary exited with the observed
empty-vector failure:

```text
terminate called after throwing an instance of 'std::out_of_range'
what():  vector::_M_range_check: __n (which is 281) >= this->size() (which is 0)
```

The exception came from the generic `pose_at_frame` reader indexing omitted
velocity data during compact trajectory selection.

## GREEN

`interaction_hand_trajectories.cpp` now uses a local trajectory pose reader.
It recognizes compact data only when all ten skipped vectors are empty, reads
only retained positions and rotations, and leaves the remaining `Pose`
channels at their default zero values. Fully populated databases continue
through the existing generic `pose_at_frame` path; mixed empty/nonempty
skipped-vector states fail with `std::invalid_argument`.

The regression verifies compact selection and Contact indexing, accepted
Contact shaping, retained transform/path equality with the full database, and
zero default velocity, angular velocity, hand-DOF, and foot-contact channels.

## Verification

```text
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
  exit 0

make -B build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_trajectory_database
  exit 0

python3 -m unittest tests.python.test_interaction_sources tests.python.test_interaction_build_cli tests.python.test_interaction_artifacts tests.python.test_hand_trajectory_viewer -v
Ran 103 tests in 4.753s
OK

make hand_trajectory_viewer
  exit 0
```
