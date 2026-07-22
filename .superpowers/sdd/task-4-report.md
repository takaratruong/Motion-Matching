# Task 4 report — focused low-memory trajectory database loader

## Commit

- Commit: `c133e9eaa66b7d7a978fefc67d6093d7612c9f5f`
- Message: `perf: load compact trajectory viewer data`
- Commit contents: only `interaction_trajectory_database.h`,
  `tests/cpp/test_interaction_trajectory_database.cpp`, and `Makefile`.

## RED evidence

The equivalence/malformed fixture test and its direct Makefile rule were added
before the loader header existed. The prescribed RED command was run:

```text
$ make -B build/tests/test_interaction_trajectory_database
make: *** No rule to make target 'interaction_trajectory_database.h', needed by 'build/tests/test_interaction_trajectory_database'.  Stop.
```

This is the expected missing-loader failure: the test target was recognized,
but its required compact-loader header had not yet been created.

## GREEN implementation and verification evidence

`interaction_trajectory_database.h` is header-only and reuses safe helpers
from `interaction_database.h`. It reads the unchanged `G1INTDB1` header,
computes the exact schema-v1 byte count with checked multiplication and checked
addition before any array access, rejects a size mismatch, and requires both
the expected final stream position and EOF.

It retains exactly these vectors:

```text
parents, range_starts, range_stops, positions, rotations, phases,
active_hands, object_positions, object_rotations, table_positions,
table_rotations, table_sizes, object_dimensions,
grasp_positions_object, grasp_rotations_object,
approach_directions_object
```

It skips with guarded bounds checks and leaves empty:

```text
velocities, angular_velocities, foot_contacts, hand_contacts,
hand_dof, hand_dof_velocities, time_to_contact,
object_velocities, object_angular_velocities, source_frames
```

The compact test creates a valid schema-v1 fixture, compares every retained
field with `load_database`, confirms every skipped vector is empty, and rejects
truncation inside skipped `velocities`, trailing data, invalid ranges,
nonmonotonic phases, missing CONTACT/LIFT phases, invalid retained rotations,
and invalid active hands.

The required verification command completed with exit code 0:

```text
$ make -B build/tests/test_interaction_trajectory_database \
    build/tests/test_interaction_database
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_interaction_trajectory_database.cpp \
  -o build/tests/test_interaction_trajectory_database
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_interaction_database.cpp \
  -o build/tests/test_interaction_database

$ ./build/tests/test_interaction_trajectory_database
$ ./build/tests/test_interaction_database
$ git diff --check
```

The same build/test/diff checks were run again immediately before staging and
committing, and `git diff --cached --check` was also clean.

## Changed files

- `interaction_trajectory_database.h`: compact schema-v1 loader, checked
  skips, exact-size/EOF checks, and retained-data validation.
- `tests/cpp/test_interaction_trajectory_database.cpp`: valid equivalence
  fixture plus required malformed-input coverage.
- `Makefile`: one compact-loader test binary entry and direct rule using
  `CPP_TEST_FLAGS`.

## Caveats

- The compact loader intentionally does not call `validate_database`, because
  that validator requires skipped vectors to be populated.
- As a result, semantics that depend solely on skipped payloads (hand/foot
  contact contents, source-frame ordering, and time-to-contact values) are not
  validated by this focused viewer loader. Their schema byte ranges are still
  overflow- and bounds-checked before they are skipped.
- An input truncated inside a skipped array or with extra trailing bytes is
  rejected by the pre-read exact schema-v1 byte-count check; the guarded skip
  logic remains as a second defense for each skipped field.

## Follow-up fix — retained approach-direction semantics

### Commit

- Commit: `4626814d65c8f25a8232f42141dfa12dd25f808c`
- Message: `fix: validate compact approach directions`
- Commit contents: only `interaction_trajectory_database.h` and
  `tests/cpp/test_interaction_trajectory_database.cpp`; `Makefile` was not
  changed.

### Fix details

The compact loader now matches the full loader's retained-field contract for
every `approach_directions_object` entry. For each clip it:

- rejects a non-horizontal direction when
  `abs(y) > detail::kFloatTolerance`; and
- computes the three-dimensional norm and rejects a non-unit direction when
  `abs(norm - 1.0) > detail::kFloatTolerance`.

The checks use the same indexing, tolerance constant, order, and error
contracts as `interaction_database.h`.

### RED evidence

The two malformed behaviors were introduced in separate TDD cycles.

First, a retained direction with `y = 0.5` was added while the compact loader
still performed only finiteness validation:

```text
$ make -B build/tests/test_interaction_trajectory_database && \
    ./build/tests/test_interaction_trajectory_database
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_interaction_trajectory_database.cpp \
  -o build/tests/test_interaction_trajectory_database
test_interaction_trajectory_database: ... expect_format_error(...):
Assertion `threw' failed.
```

After the horizontal check was implemented and passed, a horizontal retained
direction with length `0.5` was added. It produced the same expected RED at
`expect_format_error`: the loader did not throw because unit length was not yet
validated.

### GREEN evidence

After adding the full-loader-equivalent norm check, the focused test compiled
and exited 0. Final required verification was then run:

```text
$ make -B build/tests/test_interaction_trajectory_database \
    build/tests/test_interaction_database
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_interaction_trajectory_database.cpp \
  -o build/tests/test_interaction_trajectory_database
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_interaction_database.cpp \
  -o build/tests/test_interaction_database

$ ./build/tests/test_interaction_trajectory_database
$ ./build/tests/test_interaction_database
$ git diff --check
```

Both binaries exited 0 and `git diff --check` produced no output. The staged
fix also passed `git diff --cached --check` before commit.
