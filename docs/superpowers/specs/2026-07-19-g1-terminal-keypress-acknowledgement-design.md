# G1 terminal keypress acknowledgement design

## Goal

Make terminal focus and input receipt unambiguous during the interactive G1
SONIC demo. Every input batch received by the controller must produce an
immediate, human-readable acknowledgement before the next motion boundary.
If no acknowledgement appears, the controller did not receive the key.

## Scope

This change affects only interactive terminal input. It does not change motion
matching, command sampling, lookahead, SONIC physics, scripted runs, evidence
formats, or scored paths.

## Behavior

- Accepted input prints an action, for example `KEY W -> forward` and
  `KEY <SPACE> -> stand`.
- An unsupported byte sequence prints a safely escaped acknowledgement ending
  in `-> ignored`.
- An unsupported sequence is rejected atomically so an ANSI escape sequence
  such as an arrow key cannot accidentally activate a valid letter within it.
- Unsupported input does not terminate the reader; later valid controls remain
  usable.
- Carriage return and line feed remain no-op input but are acknowledged as
  ignored, so every received key still produces visible evidence.

## Design

`TerminalInputReader` receives an optional event sink. After decoding one raw
read, it first validates and feeds the complete batch into the existing strict
`TerminalKeyBuffer`. On success it reports each accepted control through the
sink. On validation failure it reports the safely escaped complete batch as
ignored and resumes reading. Genuine terminal failures such as EOF, I/O
failure, or invalid UTF-8 remain fatal and surface through the existing error
path.

The interactive manual demo supplies a sink that prints one flushed line per
event. Non-interactive users and existing callers omit the sink, preserving
their current API behavior and output.

## Alternatives rejected

- Adding keys only to the 0.4-second boundary line is delayed and cannot prove
  immediate receipt.
- Capturing keys in the MuJoCo viewer requires new cross-process input plumbing
  and is outside this focused diagnostic change.

## Verification

PTY-based tests will establish a red-green regression for:

1. a valid key being fed and acknowledged;
2. an unsupported sequence being acknowledged as ignored without stopping the
   reader;
3. a valid key still working after the unsupported sequence;
4. genuine terminal closure remaining a surfaced failure.

The replacement live session must then visibly print an injected `W`, show the
corresponding nonzero boundary command, print Space, return to zero velocity,
and remain upright in the MuJoCo window.
