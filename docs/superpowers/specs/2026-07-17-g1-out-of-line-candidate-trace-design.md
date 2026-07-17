# Out-of-Line Candidate Trace Design

## Status and purpose

The committed unscheduled-incumbent repair at `922e092` passes its unit,
sanitizer, privacy, native, Python, and independent-review gates. The first
paired live gate then found that the fast dual-seam trace executable and the
fast no-seam release executable do not emit byte-identical production CSVs.
This design repairs the evidence boundary before any candidate visualizer is
launched.

## Root-cause evidence

The failure is deterministic and compile-time, not a runtime trace side
effect:

- two no-seam release runs emit the same CSV SHA-256;
- two dual-seam runs emit the same, different CSV SHA-256;
- enabling or omitting `MM_CANDIDATE_TRACE` in the dual-seam executable does
  not change its production CSV;
- enabling either test-seam macro alone preserves release CSV bytes, while
  enabling both includes the large inline trace implementation and changes
  query words by a few ULPs; scheduled search rows can consequently change the
  selected-cost word and accepted-state digest;
- equivalent strict controller builds are byte-identical across no-seam and
  dual-seam configurations.

`-ffast-math` therefore permits compilation-unit-dependent reassociation when
the trace serializer/oracle implementation is present in the same translation
unit as the controller. Comparing those two binaries cannot certify that the
instrumentation is observationally neutral.

## Non-negotiable constraints

- Keep the production no-seam controller build and its `-ffast-math` behavior
  unchanged.
- Keep exact 25 Hz runtime behavior, matcher cadence, thresholds, candidate
  capacity, heading/travel independence, terrain/database assets, and log
  schemas unchanged.
- Keep the strict recovery provider and same-process exhaustive oracle as the
  only recovery evidence owners.
- Do not relax byte equality, compare rounded values, or omit query/cost/state
  fields.
- A release executable must expose no trace/oracle symbols or strings.
- Do not inspect, discover, signal, replace, or otherwise touch the existing
  visualizer. A second candidate window remains gated on the repaired 18/32
  proof.

## Chosen architecture

Split the trace implementation from its interface:

1. `g1_candidate_certification_trace.h` retains the seam-only
   `G1CandidateTraceFile` type and declarations for
   `g1_candidate_trace_open`,
   `g1_candidate_trace_append_after_transaction`, and
   `g1_candidate_trace_close`. It contains no serializer, validation, oracle,
   or file-I/O function body.
2. New `g1_candidate_certification_trace.cpp` owns the three external
   definitions and all existing validation, fixed-buffer serialization,
   same-database exhaustive-oracle comparison, write, flush, and close logic.
   It is compilable only with both test-seam macros and is built with strict
   floating-point flags.
3. `controller.cpp` keeps its existing include, environment opt-in, test-seam
   pointer, append call, and cleanup call. No production control-flow or math
   source changes are required.
4. The trace executable links the strictly compiled trace implementation and
   seam recovery objects. The release executable neither compiles nor links
   the trace implementation.

This boundary leaves the fast controller translation unit with only the small
trace interface, preventing the serializer/oracle implementation from
participating in its optimization decisions while preserving the exact live
trace data flow.

## Error and ownership behavior

All existing fail-closed behavior remains literal: invalid paths, overlapping
memory, malformed recovery evidence, invalid enum/provenance words, oracle
disagreement, fixed-buffer overflow, write failure, and flush failure stop the
trace process. The trace object never influences release publication. The
same loaded `database` pointer is still passed from the completed transaction
to the strict exhaustive oracle in the same trace process.

## Test-first proof

Before moving implementation code:

- add a focused production source/linkage test that requires the interface
  header to be implementation-free, the new strict owner to contain the three
  definitions and exhaustive-oracle call, and `controller.cpp` to include only
  the interface;
- preserve the already observed fast 18-frame cross-build CSV mismatch as the
  live RED integration evidence.

After the split:

- build focused strict and fast production tests with the new trace object;
- run trace serializer grammar/mutation/oracle tests and sanitizer coverage;
- require the frozen scheduled trace transcript SHA-256 to remain unchanged;
- rebuild fast trace and no-seam release executables and require exact
  18-frame and 32-frame production-CSV equality, IK-off/on candidate-trace
  equality, and release privacy;
- require canonical frame 17 to retain incumbent provenance
  `117711 -> 117712`, range `402`, cost bits `40f14831`, zero legacy
  traversals, one lazy provider call, accelerated/exhaustive equality, and one
  first dual-certified strict winner;
- only then launch the separately hashed no-seam release candidate on
  `mixed-multilevel` for visual inspection.

## Rejected alternatives

- Compiling both live controllers without fast math proves a different runtime
  and can change the 25 Hz performance envelope.
- Comparing only selected invariants or allowing ULP tolerance weakens the
  instrumentation-neutrality contract and can hide a changed matcher choice.
- Linking seam code into the release executable violates production privacy.

## Completion criteria

The repair is complete when the original no-seam fast runtime remains
unchanged, the trace implementation is a strict out-of-line owner, release
privacy passes, all focused/full regressions are green, the paired fast 18/32
CSVs are byte-identical, canonical frame 17 recovers exactly, and a separate
candidate visualizer is launched without interacting with the existing one.
