# Reach Vertical Endpoint Gate Design

## Goal

Do not retarget, compute, or display reach motions whose recorded wrist
endpoint height differs from the requested grasp height by more than 0.10 m.

## Design

Add `maximum_endpoint_vertical_delta_m = 0.10F` to `CoverageConfig`.
Compatibility is the absolute difference between the requested wrist target Y
and the recorded endpoint Y. Yaw placement does not alter height.

Filter incompatible candidates before exhaustive-search worker dispatch so
they do not consume IK/collision work and are absent from visualization.
Apply the same check in candidate selection and direct shaping as a
defense-in-depth guarantee for non-viewer callers. Treat a direct bypass as
`OutsideEnvelope`.

The threshold is inclusive: exactly 0.10 m is allowed; anything greater is
excluded.

## Verification

- Search retains candidates at 0.10 m and excludes candidates above it.
- Excluded candidates do not contribute to search total or displayed paths.
- Direct shaping rejects an incompatible endpoint before producing poses.
- Invalid negative or non-finite thresholds are rejected.
