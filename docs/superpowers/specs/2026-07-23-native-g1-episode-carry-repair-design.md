# Native G1 Episode Locomotion and Carry Repair

## Goal

Repair the playable pickup episode so camera-relative controls are correct,
ordinary locomotion, reach, carry, and placement all publish one 31-bone G1
pose, and carrying retains visible motion-matched locomotion instead of sliding
a nearly static carry clip through the scene.

The object-relative exhaustive reach search, collision filtering, attachment
state machine, reversible placement playback, and flat furniture scene remain
unchanged.

## Confirmed failures

The current viewer computes camera-right as `up x forward`. In this camera
convention that points toward screen-left, reversing A and D.

The viewer loads `resources/database.bin` for ordinary locomotion. That database
uses the 23-bone LAFAN skeleton and produces a roughly 1.60 m debug figure.
Reach and carry use the 31-bone G1 databases in the episode pack and produce a
roughly 1.05 m figure. The viewer therefore changes both hierarchy and scale at
interaction ownership.

The episode pack already contains
`build/g1-episode/walking_database.bin`, generated from the same retargeted G1
archive as the bilateral carry databases. The viewer does not currently use
it. The carry sequences have limited root speed and leg variation; using them
as complete locomotion poses makes carrying look like skating.

## Chosen architecture

Use the same-pack native G1 walking database for every locomotion-owned state.
Do not render or run the LAFAN skeleton in the playable episode.

```text
camera-relative WASD
        |
        v
31-bone G1 walking matcher
        |
        +------> Free / pickup approach
        |
        +------> layered carry base
                    |
                    +-- walking root, hips, legs, spine motion
                    +-- final hold active-arm posture
                    +-- posture-aware hand IK
                    +-- wrist-authoritative attached object
        |
        v
reach/place bridge and complete G1 reach playback
        |
        v
one final 31-bone G1 pose
        +-- blue G1 debug skeleton
        +-- optional G1 mesh
```

The terrain-aware database remains a later interchangeable locomotion source.
It is not loaded in this repair because the compact same-pack walking data
avoids the terrain viewer, large terrain artifacts, and cross-source skeleton
seams.

## Controls

Extract camera-relative planar command construction into a deterministic helper
that accepts camera forward and four digital movement inputs. Camera-right is
`forward x up`. W/S add and subtract forward; D/A add and subtract right.
Diagonal input is normalized before applying the configured command speed.

The command speed must be calibrated to the demonstrated native G1 walking
range rather than the previous LAFAN speed. Motion playback remains
data-driven; the runtime must not translate the root independently of the
selected motion.

## One native G1 pose

The episode viewer supplies
`<episode-pack>/walking_database.bin` to `InteractionEpisode`.
`FlatMotionMatcher` already accepts a 31-bone G1 database and publishes a
31-bone pose. In this mode its 23-bone debug pose is invalid and must never be
used for rendering.

`EpisodeOutput::pose` is the sole authoritative visual and interaction pose in
all states. The viewer always converts that pose through
`interaction::world_pose` and draws the G1 parent hierarchy. The optional mesh
consumes the same world transforms. No state-dependent LAFAN/G1 rendering
branch remains.

Construction fails clearly if the selected playable walking database is not
the 31-bone G1 hierarchy. This prevents a silent return to mixed skeletons.

## Layered carry

At pickup contact, preserve the final hold pose and selected hand. Rebase the
native G1 walking matcher to that live root instead of switching the complete
pose source to a carry database.

Each Carry or PlaceApproach update:

1. Advance ordinary native G1 motion matching from the live movement command.
2. Start from the resulting locomotion pose, retaining its root, hips, legs,
   foot contacts, and unselected-arm motion.
3. Blend the three spine rotations partially toward the final hold pose so the
   grasp has a stable torso base without freezing locomotion sway.
4. Seed the selected seven-joint arm from the final hold pose.
5. Solve the selected wrist to the attached object's desired hand transform
   using the existing posture-aware bounded IK path.
6. Publish the solved G1 pose and derive the object transform from the solved
   wrist and frozen hand-in-object transform.

The existing carry blend remains the handoff seam. A failed carry solve must
not detach or teleport the object. It retries from the last accepted active-arm
branch; if no bounded solution exists, the last accepted carry pose is remapped
to the current walking root for that frame and an actionable diagnostic is
reported.

Placement approach uses the same layered carry composition. PlaceBridge then
blends from the final layered G1 pose into the chosen complete G1 reach clip.
After release and reversed reach playback, the native G1 walking matcher is
rebased to the returned pose.

## Data and compatibility

The episode pack continues to contain:

- `walking_database.bin`: native G1 locomotion source;
- `carry_left_database.bin`: left carry posture/reference data;
- `carry_right_database.bin`: mirrored right carry posture/reference data.

Carry databases are no longer complete root/lower-body motion sources. They
remain valid posture references and preserve bilateral hand conventions.

The reach pack, object dimensions, grasp provider, support validation, and
search deadlines are unchanged.

## Verification

Automated tests must prove:

1. Camera-relative D has positive projection onto screen-right and A has the
   opposite projection.
2. The playable walking database has 31 bones with the exact G1 parent table.
3. Free locomotion publishes no valid 23-bone render pose.
4. Free, approach, reach, carry, place, and return poses all contain finite G1
   channels and retain G1 morphology.
5. A carry command produces material root displacement and material alternating
   leg/foot motion.
6. During that displacement the attached object remains within the existing
   position and orientation grasp tolerances.
7. Left- and right-hand carry both retain the selected grasp while the inactive
   arm remains animated.
8. Pickup, shelf placement, shelf re-pick, and placement back to the lower
   table still complete.

Manual acceptance drives A/D/W/S before pickup, initiates pickup from multiple
angles, walks while carrying, places on the shelf, re-picks, and places back.
The blue skeleton must remain the same size and topology throughout, the feet
must visibly animate during carry, and the character must move in the
screen-relative direction requested.

## Deferred work

The terrain-aware locomotion database can replace the compact walking source
through the same matcher interface if the retargeted walking range lacks enough
directional coverage. Training a monolithic locomotion/manipulation model,
speed-warping carry clips, and enabling terrain rendering are outside this
repair.
