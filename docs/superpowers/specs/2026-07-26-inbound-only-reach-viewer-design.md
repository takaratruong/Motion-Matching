# Inbound-Only Reach Viewer Design

## Goal

The G1 reach coverage viewer must visualize only the inbound approach toward
the object, ending at the grab/contact frame. It must not animate or draw the
post-contact pullback/return.

## Behavior

- Keep the stored reach pack unchanged, including paired return motion.
- Derive a clip-local inbound frame count from the selected clip's global
  `range_start` and `contact_frame`.
- Loop selected animation samples only over frames `[0, contact_local]`.
- Draw selected, accepted, rejected, preferred, and retargeted wrist paths only
  over the same inbound interval.
- Preserve search, IK, collision, ranking, camera, object controls, and all
  other viewer behavior.

## Verification

A source-contract regression test must prove that playback and path drawing use
the contact boundary instead of the complete pose vector. The viewer must build,
launch against `reach-pack-v5`, complete one search, and visibly animate only
the inbound approach.
