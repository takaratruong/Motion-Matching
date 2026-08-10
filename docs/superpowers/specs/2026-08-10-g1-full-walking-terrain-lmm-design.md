# Full Walking-Only Terrain LMM Design

Date: 2026-08-10

Status: approved in conversation

Execution deadline: 2026-08-10 14:00 PDT for the trained, evaluated,
steerable MuJoCo result. The deeper raw-source GRAIL reconstruction audit is
nonblocking for that deadline; all corpus, model, runtime-safety, and formal
viewer gates in this document remain binding.

## Objective

Replace the sparse overnight proof of concept with one steerable Unitree G1
terrain motion matcher trained on the complete admitted walking corpus. The
system must cover standing, walking, stand/walk blends, and terrain
transitions across flat ground, curbs, slopes, and stairs. It must reduce the
visible planted-foot sliding caused by sparse motion coverage and by the
current command-owned planar root.

This is a walking-only system. Jog, run, crouch, crawl, jump-only, sitting,
pickup, and manipulation motion are excluded. Official PFNN mirrors are
included because they provide bilateral turning and foot-phase coverage.

## Relationship to Existing Work

The deep source-native verification reference remains
`2026-08-10-g1-full-terrain-lmm-corpus-design.md`. The first full-model critical
path does not redundantly decode every already-verified GRAIL clip from its raw
Joblib/USD pair. It uses the frozen strict broad bank as a bulk authority,
processes only data missing from that authority, and retains the source-native
pipeline as a later independent audit/fallback.

The current hybrid viewer remains the deployment baseline:

- exact command- and terrain-conditioned search;
- a learned latent table and learned pose decompressor;
- authenticated MuJoCo G1 and terrain scenes; and
- mechanically invalid candidate rejection before commit.

The overnight 3,973,057-row cache and 2,125-row PFNN supplement remain frozen
comparison artifacts. They are not silently extended or relabeled as the full
corpus.

## Complete Admitted Corpus

The full immutable 60 Hz corpus contains:

- every PFNN source row whose authenticated gait is stand, walk, a stand/walk
  blend, or a terrain transition, plus every official mirror;
- all 1,769 GRAIL curb clips;
- all 1,880 GRAIL slope clips;
- all 6,094 GRAIL stair-p1 clips;
- all 6,094 GRAIL stair-p2 clips; and
- the complete Takara walk.

PFNN jog, run, crouch, crawl, and jump-only rows remain counted in exclusion
receipts but never enter a trainable range. GRAIL sitting, pickup, and other
non-locomotion categories never enter the inventory.

The nominal corpus is approximately 9,978,369 rows before PFNN gait,
continuity, and quality filtering.

The accelerated authority path is exact:

- reopen the frozen 3,970,932-row strict broad bank and authenticate its
  database, terrain, support, ranges, features, source maps, and manifest;
- exclude its Takara range and range-locally vector-resample its 1,769 curb,
  1,857 slope, and 12,188 stair ranges from 25 to 60 Hz;
- process the 23 GRAIL slope clips absent from that bank through the already
  verified raw slope adapter;
- process Takara directly from its authenticated 50 Hz source to 60 Hz; and
- process every admitted PFNN walking source/mirror from its authenticated
  source and terrain fit.

Each inherited GRAIL range receives a derived receipt binding its exact parent
bank/range/source identity and new range-local source map. Missing-slope,
Takara, and PFNN sources receive ordinary source-native receipts. A second
deterministic build must reproduce every output digest. The slower raw
per-GRAIL-source reconstruction remains an audit lane and does not block the
first full trained model.

## Canonical Representation

All sources use one support-relative 31-bone G1 representation at exactly
60 Hz. No derivative, contact filter, trajectory horizon, training window, or
successor crosses a source, mirror, continuity, or terrain-fit boundary.

Every row stores:

- local pose, linear/angular velocities, and bilateral contacts;
- the established 27 motion/trajectory matching features;
- four terrain heights at 0.25, 0.50, 0.75, and 1.00 m;
- the PFNN 12-station by 3-lane terrain sidecar;
- root/left/right support heights;
- exact source interpolation indices and alpha; and
- family, terrain class, source, mirror, range, quality, and split identity.

Normalization is fit only on verified `clean + usable` training rows. Mirrors,
GRAIL variants of one terrain identity, and all ranges from one canonical
source remain in the same train, validation, or test split.

## Parallel Build

After the common schema, conversion, storage, and quality interfaces are
frozen, these lanes execute concurrently:

1. verified-bank GRAIL range-local 25-to-60 Hz resampling;
2. the 23 missing GRAIL slopes;
3. PFNN flat/walking/transitions and official mirrors;
4. PFNN terrain walking and official mirrors; and
5. direct Takara 50-to-60 Hz conversion.

Workers publish only lane/range-local immutable artifacts. Verification,
packing, global split/index construction, and normalization are separate
restartable phases. A failed supplemental source receives a structural
rejection or quarantine receipt without stopping unrelated lanes. No worker
completion order affects shard or split identities.

## Training

Train one deterministic learned generator over the verified full index. The
first experiment reuses the accepted visual-articulation architecture:
latent 32, width 512, deterministic AdamW, and the established physical
metrics. Latent 64 is attempted only if the full-corpus held-out metrics or
coverage diagnostics are red. A VAE is not introduced unless deterministic
models fit training rows but fail genuinely held-out source identities.

Sampling is hierarchical rather than row-uniform:

1. balance the four runtime terrain classes `flat|curb|slope|stair`;
2. within a class, balance canonical source identities;
3. within a source, balance speed, turn-rate, contact state, and transition
   bins; and
4. draw only range-safe windows.

This prevents the roughly seven million stair rows from overwhelming flat,
turning, curb, and slope coverage. Model selection uses validation/test source
identities, never training loss alone. The selected architecture is then
refit on all verified `clean + usable` rows while retaining the immutable
selection receipt.

## Motion-Derived Runtime Root

The joystick or keyboard specifies a desired walking velocity and heading; it
does not directly drag the character root.

At each exact search event, the query contains the desired future velocity,
trajectory, facing, and live terrain. Candidate scoring uses normalized feature
distance, contact compatibility, and the frozen continuity/transition cost.
Between searches, the range-safe successor supplies both articulation and its
canonical planar `Simulation`-root displacement.

The committed world-root displacement is the selected motion displacement
rotated into the commanded heading. It is not independently integrated at a
fixed speed and is not arbitrarily scaled. Stick magnitude selects among the
walking speeds represented by the verified corpus; requests above the corpus
95th-percentile walking speed clamp visibly to that supported envelope.

This makes foot motion and root travel come from the same source interval,
addressing the principal runtime cause of visible sliding. Steering remains
responsive because command/facing updates trigger an exact search at most
100 ms later. Mechanically invalid learned candidates are retried in exact
score order; native limits are never relaxed.

## Evaluation Gates

### Corpus and split gates

- every inventory identity has a terminal outcome;
- every inherited range independently reconstructs from its authenticated
  parent-bank range, and every supplemental range independently reconstructs
  from its authenticated source;
- no mirror/source/terrain identity crosses splits;
- no boundary-sensitive operation crosses a range;
- every terrain query used for features/support is valid; and
- exclusion and quality counts are published by family and reason.

### Learned generator gates

- all outputs and receipts are finite;
- held-out joint geodesic MAE is at most 0.03 rad;
- held-out joint frame-max p95 is at most 0.10 rad;
- held-out FK p95 is at most 0.08 m;
- held-out support-foot p95 is at most 0.05 m; and
- bilateral contact F1 is at least 0.85.

The full row-aligned native-limit audit is published as a diagnostic. Runtime
acceptance still requires every committed scripted pose to be native-valid
with zero clamp and zero fallback.

### Coverage and sliding gates

Run identical scripted commands against the frozen overnight baseline and the
full model. On authenticated flat, curb, slope, and stair scenes, the full
model must:

- reduce p95 exact-search query distance by at least 30 percent;
- reduce planted-foot planar-slip p95 by at least 50 percent;
- keep planted-foot median slip at or below 0.02 m/s and p95 at or below
  0.08 m/s;
- keep desired-versus-realized planar speed MAE at or below 0.12 m/s;
- select at least two canonical source identities per terrain class; and
- complete at least 10,000 aggregate MuJoCo frames with zero committed native
  violation, clamp, fallback, nonfinite state, or out-of-range successor.

Planted-foot slip is measured at the authenticated native ankle support probes
during contact segments, after world-root placement and MuJoCo forwarding.
The same reducer and command scripts evaluate baseline and candidate.

## Viewer and Controls

The current viewer contract remains:

- W/S or left-stick Y selects desired supported walking speed;
- A/D or left-stick X selects steering;
- Space is a hard stop;
- R resets to the authenticated scene spawn; and
- unsupported terrain or command speed is visibly diagnostic.

The accepted label remains a hybrid claim: exact terrain-aware search plus a
learned generator on supported walking terrain. It is not a dynamics policy,
learned projector, arbitrary-terrain claim, or evidence of jogging/running.

## Delivery Sequence and Estimate

1. Finish and review the shared accelerated-authority schema/storage layer.
2. Run the five source-processing lanes in parallel.
3. Independently verify, pack, split, normalize, and index the full corpus.
4. Train and select the full-corpus generator.
5. Implement motion-derived root travel and exact transition scoring.
6. Run baseline-versus-full coverage/slip evaluation and formal MuJoCo routes.
7. Publish a new immutable viewer/model/evidence set; never overwrite the
   overnight PoC.

Because representative conversion from every source family is already proven
and 15,814 GRAIL ranges come from the frozen strict bank, the deadline plan
overlaps the five data lanes, runtime-root work, and corpus verification:

- 09:03-09:30 PDT: freeze interfaces and the executable task plan;
- 09:30-10:45 PDT: implement and run verified-bank resampling, PFNN, Takara,
  and missing-slope lanes in parallel;
- 10:45-11:30 PDT: verify, pack, split, normalize, and index;
- 11:30-12:30 PDT: train and select the full-corpus generator while runtime
  root integration completes independently;
- 12:30-13:15 PDT: immutable all-row refit and baseline comparison; and
- 13:15-14:00 PDT: formal MuJoCo routes, viewer inspection, and evidence
  publication.

This is an aggressive five-hour execution target. The conservative estimate
remains 4-7 hours for the first full trained model and 6-10 hours end to end;
meeting 14:00 requires the already-verified adapters to remain mechanical and
all major phases to finish near their low bound. Source corruption,
unavailable PFNN provenance, or failed held-out/slip gates can extend the
finish. They do not authorize silently dropping sources, weakening the gates,
or relabeling a partial corpus as complete.

## Completion Claim

Completion means every admitted walking identity is represented either by an
authenticated frozen-bank range receipt or an explicit supplemental terminal
outcome, the combined build is byte-reproducible, the selected learned
generator is trained/evaluated on the full index, and the motion-root viewer
passes the coverage, slip, and formal terrain gates. A partial PFNN set,
canary overfit, or unreceipted append cannot be substituted for this claim.
