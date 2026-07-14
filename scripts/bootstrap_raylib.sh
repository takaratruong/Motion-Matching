#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
deps="${DEPS_DIR:-$root/.deps}"
raylib_commit=dbc56a87da87d973a9c5baa4e7438a9d20121d28
raygui_commit=25c8c65a6e5f0f4d4b564a0343861898c6f2778b

if [[ "${1:-}" == "--print-lock" ]]; then
    printf 'raylib %s\nraygui %s\n' "$raylib_commit" "$raygui_commit"
    exit 0
fi

checkout_exact() {
    local name="$1" url="$2" commit="$3" path="$deps/$1"
    if [[ ! -d "$path/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$path"
    fi
    git -C "$path" fetch --depth 1 origin "$commit"
    git -C "$path" checkout --detach "$commit"
    test "$(git -C "$path" rev-parse HEAD)" = "$commit"
    test -z "$(git -C "$path" status --porcelain)"
}

mkdir -p "$deps"
checkout_exact raylib https://github.com/raysan5/raylib.git "$raylib_commit"
checkout_exact raygui https://github.com/raysan5/raygui.git "$raygui_commit"
make -C "$deps/raylib/src" PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=STATIC
test -f "$deps/raylib/src/libraylib.a"
test -f "$deps/raygui/src/raygui.h"
