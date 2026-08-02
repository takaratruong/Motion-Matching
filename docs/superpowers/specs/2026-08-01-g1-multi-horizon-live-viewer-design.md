# G1 Multi-Horizon Live Viewer Design

## Goal

Drive the qualified multi-horizon terrain-skill matcher interactively in the
existing native MuJoCo kinematic viewer with the existing WASD, stop, reset,
and exit controls.

## Architecture

Add an explicit `--multi-horizon` viewer mode. The mode resolves the checked-in
multi-horizon experiment config, builds the plain 27-value base motion database,
then builds the terrain-skill and horizon inventories and wraps the base matcher
with `TerrainSkillHorizonMatcher`. The terrain measurement extension remains
available to the wrapper for commanded-path height targets, full supported-foot
trace validation, terrain markers, and heightfield rendering; it is not appended
to the base motion-search feature vector.

The existing vanilla/contact/foothold viewer modes remain unchanged. Multi-
horizon mode is mutually exclusive with contact-segment and foothold modes so
two independent terrain planners cannot be composed accidentally.

## Runtime behavior

- `W/A/S/D`, `Space`, `Backspace`, and `X` retain their current meanings.
- The window opens at the authenticated stair reset position on `DISPLAY=:1`.
- The overlay identifies multi-horizon mode and shows selected source frame,
  target horizon, endpoint, entry/outcome/total cost, and matcher time.
- Playback remains kinematic: MuJoCo receives qpos and runs forward kinematics;
  no physics or Sonic process is launched.
- A structured horizon search failure is displayed in the overlay without
  committing a pose. A command change or Backspace reset clears the displayed
  fault and permits another search, preventing the viewer process from closing
  on an ordinary coverage failure.

## Testing and launch

Parser tests cover the new flag and incompatible modes. Overlay tests cover both
legacy and horizon diagnostics. A construction test proves the multi-horizon
builder uses the qualified adapter and the 27-value base database. Existing
viewer tests remain green. After focused verification, launch the checked-in
qualified config on `DISPLAY=:1` and verify the process and window remain live.
