"""Plot reference and tracked root routes from a saved SONIC rollout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import zarr


def _rollout_zarr(path: Path) -> Path:
    source = path.expanduser().resolve()
    if source.is_dir() and source.suffix != ".zarr":
        matches = sorted(source.glob("*.zarr"))
        if len(matches) != 1:
            raise ValueError(f"expected one rollout zarr in {source}, found {len(matches)}")
        return matches[0]
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        default=[],
        help="Episode to plot; repeat to make a compact selected sheet.",
    )
    arguments = parser.parse_args(argv)

    rollout = zarr.open_group(str(_rollout_zarr(arguments.rollout)), mode="r")
    data = rollout["data"]
    meta = rollout["meta"]
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    clips = np.asarray(meta["episode_clip"][:]).astype(str)
    origins = np.asarray(meta["episode_env_origin"][:], dtype=np.float64)
    failed = np.asarray(meta["episode_failed"][:], dtype=bool)
    mpjpe = np.asarray(meta["episode_mpjpe_mm"][:], dtype=np.float64)

    episodes = tuple(arguments.episode) or tuple(range(len(clips)))
    if any(index < 0 or index >= len(clips) for index in episodes):
        raise ValueError("requested route episode is outside the rollout")
    columns = min(3, len(episodes))
    rows = int(math.ceil(len(episodes) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.6 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    bundle = arguments.bundle.expanduser().resolve()

    for panel, axis in enumerate(axes):
        if panel >= len(episodes):
            axis.axis("off")
            continue
        episode = episodes[panel]
        start, stop = int(starts[episode]), int(ends[episode])
        clip = str(clips[episode])
        blob = joblib.load(bundle / "robot" / f"{clip}.pkl")
        reference = np.asarray(blob[clip]["root_trans_offset"], dtype=np.float64)
        actual = np.asarray(data["root_pos"][start:stop], dtype=np.float64) - origins[episode]
        count = min(len(reference), len(actual))
        reference = reference[:count, :2]
        actual = actual[:count, :2]

        # Remove only the simulator's small spawn offset. The route shape and
        # accumulated tracking drift remain visible.
        actual = actual - (actual[0] - reference[0])
        axis.plot(reference[:, 0], reference[:, 1], "--", color="#d62728", lw=2.3, label="reference")
        axis.plot(actual[:, 0], actual[:, 1], color="#1f77b4", lw=2.0, label="SONIC")
        axis.scatter(*reference[0], marker="o", s=35, color="#2ca02c", zorder=4)
        axis.scatter(*reference[-1], marker="X", s=55, color="#111111", zorder=4)
        short = clip.replace("manifests_", "").replace("_left_right", " L→R").replace("_right_left", " R→L")
        if len(short) > 58:
            short = short[:55] + "…"
        state = "FAIL" if failed[episode] else "full rollout"
        axis.set_title(f"{episode}: {short}\n{state}, peak MPJPE {mpjpe[episode]:.1f} mm", fontsize=9)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.22)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        if panel == 0:
            axis.legend(loc="best")

    figure.suptitle(
        "Directional stairs: intended versus physically tracked root route",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
