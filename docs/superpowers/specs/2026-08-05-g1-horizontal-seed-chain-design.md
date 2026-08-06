# G1 Horizontal Seed Chain Design

## Goal

Package approved step-up, uneven-travel, and step-down motions as unchanged
terrain-valid seeds, with explicit path gaps for MotionBricks to in-between.

## Scope

Build chains for the three lanes that have visually approved or strictly
certified edge actions: `lane-pos-0p6`, `lane-pos-0p8`, and `lane-pos-1p0`.
The `+0.6 m` chain uses the visually approved drop reference; `+0.8 m` and
`+1.0 m` use strictly certified dismounts.

## Design

Each seed record contains its NPZ path, source identity, source-frame interval,
and covered path interval. Adjacent intervals produce a boundary record:

- `uncovered_m = max(0, next.start_m - current.stop_m)`
- `overlap_m = max(0, current.stop_m - next.start_m)`

MotionBricks may optimize only boundaries. It must preserve seed interiors and
may choose a splice within overlap. Any uncovered boundary greater than
0.40 m invalidates the chain.

The viewer playlist concatenates seeds with holds and teleport boundaries. It
does not blend them, so visual evaluation cannot mistake a hidden invalid join
for valid source motion.

## Validation

The packager rejects missing arrays, invalid intervals, out-of-order seed
kinds, excessive gaps, and non-finite data. Unit tests cover gap/overlap
calculation and fail-closed chain admission. Generated reports must show all
boundary gaps at or below 0.40 m.

