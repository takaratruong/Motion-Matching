# Task 3: Load Contact-Bounded Clips in C++

## Summary

- Upgraded reach-pack loading to V2 (`G1RCHD2` / `G1RCHF2`, version 2).
- Loaded `contact_frames` immediately after range stops and validated exactly one
  contact per clip with `range_start <= contact_frame < range_stop`.
- Added checked contact/return range accessors and bounded outbound shaping at
  the contact frame. Return shaping and episode state were not changed.

## Changed files

- `reach_database.h`, `reach_database.cpp`
- `reach_motion.h`, `reach_motion.cpp`
- `reach_coverage.cpp`
- `tests/cpp/test_reach_database.cpp`
- `tests/cpp/test_reach_coverage.cpp`
- `tests/cpp/test_reach_search.cpp` (direct `Database` fixture updated for V2)

## TDD evidence

1. RED: after adding the loader/accessor tests, the focused build failed because
   `Database::contact_frames` and the three requested accessors did not exist.
2. GREEN for loading/accessors: added the V2 field, parsing, validation, and
   checked accessors; `test_reach_database` passed.
3. The first outbound fixture was too short for the existing 0.6-second
   correction-ramp guard and rejected before return-frame behavior was reached.
   The fixture was lengthened to 21 outbound frames. With `shape_candidate`
   temporarily restored to `range_stops`, it failed at the expected outbound
   assertion after observing wild post-contact frames. Restoring
   `contact_frame + 1` made it pass.

## Tests

```bash
make build/tests/test_reach_database build/tests/test_reach_coverage
./build/tests/test_reach_database
./build/tests/test_reach_coverage
make build/tests/test_reach_search
./build/tests/test_reach_search
git diff --check
```

Result: all commands exited 0; the test binaries were silent on success and
`git diff --check` produced no output.

## Self-review

- Confirmed V2 magic/version constants and `contact_frames` wire ordering.
- Confirmed count/range validation rejects contact-before-start and
  contact-at-stop, while contact-at-`stop - 1` is accepted as return-unavailable.
- Confirmed outbound shaping uses `clip_contact_frame(...) + 1`; no return
  shaping or episode-state logic was added.
- Confirmed no generated packs or binaries are staged for this task.

## Concerns

None. V1 packs are intentionally rejected; the Python publisher contract now
produces only V2 packs.
