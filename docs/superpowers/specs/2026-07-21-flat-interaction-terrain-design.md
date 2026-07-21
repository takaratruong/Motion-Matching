# Flat Interaction Terrain Design

## Goal

Run the native 31-bone G1 diffusion pickup demo on a visually and physically
flat ground plane while retaining the native terrain-aware controller,
interaction runtime, object lifecycle, and G1 mesh/skeleton pose path.

## Design

Add an explicit `MM_INTERACTION_FLAT_TERRAIN=1` launch option. After the
selected terrain scene passes its existing pack validation, the controller
replaces every loaded heightfield sample with `0.0F`, sets the exterior height
to zero, and marks every walkability cell as walkable. This preserves the
validated scene dimensions and coordinate system while removing curb, stair,
and slope behavior from physics and trajectory traversal.

When the option is active, rendering omits the loaded terrain model and draws
a flat plane covering the selected scene's heightfield bounds. The native G1
controller, interaction target, source and destination tables, diffusion
route, placement staging marker, and final-pose publication remain unchanged.

The option is opt-in and parsed strictly: absent or `0` keeps the validated
terrain scene; `1` enables the override; any other value fails at startup with
a clear error. The live acceptance launch will set it to `1`.

## Safety and Testing

- A source-level test requires strict option parsing, physical heightfield and
  walkability flattening, and the flat rendering branch.
- Existing native-G1 architecture, mesh, interaction, carry, placement,
  diffusion, and runtime tests must continue to pass.
- A bounded controller run must start and close cleanly with the override.
- Only one live controller process may run; the prior PID is stopped by exact
  identity before relaunch.

## Deferred Scope

This override does not create or modify a certified terrain-pack artifact. It
also does not add terrain-grid obstacles to diffusion planning because the
selected environment is deliberately flat and fully walkable.
