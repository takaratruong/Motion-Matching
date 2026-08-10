# Arrow-key driving controls

## Scope

The interactive terrain viewer will accept the arrow keys as its primary keyboard driving controls while retaining the existing WASD aliases:

- Up or W: forward
- Down or S: backward
- Left or A: steer left
- Right or D: steer right
- Space: hard stop
- R: reset
- X, Q, or Escape: exit

No runtime motion, search, model, terrain, or gamepad behavior changes.

## Design

The existing `pynput` listener receives printable keys through `key.char`, but arrow keys arrive as `Key.up`, `Key.down`, `Key.left`, and `Key.right` with no character. A small listener normalization seam will map those special keys to the same internal tokens already consumed by `KeyboardCommandSource`. Press and release will use the same mapping so held-key level state cannot stick.

WASD remains supported as an alias. This keeps compatibility and provides a fallback if desktop keyboard routing behaves differently between X11 sessions. The on-screen overlay will list arrow keys first and WASD second.

## Failure handling

Unknown special keys remain ignored. Escape retains its current exit behavior. Simultaneous opposing directions retain the existing neutralization behavior in `CommandState.from_keyboard`.

## Verification

Tests will first demonstrate that the current listener ignores arrow keys. The implementation must then prove:

1. Each arrow press produces the expected speed or steering command.
2. Each arrow release clears the corresponding command.
3. Opposing arrow keys neutralize exactly like opposing WASD keys.
4. WASD, Space, reset, and exit behavior remain unchanged.
5. The overlay advertises the controls actually accepted by the listener.
6. The restarted real MuJoCo viewer remains live after the change.
