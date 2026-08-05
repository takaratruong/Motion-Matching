"""Build route profiles and contact fragments for the clean C490 curb bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_stairs500_fragment_bank import build_stairs500_fragment_bank
from .terrain_oracle.stair_geometry_warp import build_motion_route_profile_catalog


DEFAULT_ROUTE_CATALOG = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-c490-curb-slope-motion-route-profiles-v1.npz"
)
DEFAULT_FRAGMENT_DIR = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-c490-curb-fragments-v1"
)
DEFAULT_FRAGMENT_BANK = DEFAULT_FRAGMENT_DIR / "fragments.jsonl"


def main(argv: list[str] | None = None) -> int:
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--route-catalog", type=Path, default=DEFAULT_ROUTE_CATALOG)
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT_DIR)
    arguments = parser.parse_args(argv)
    archive_path = arguments.archive.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    curb_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            np.asarray(archive["clip_family"][:]).astype(str) == "c490_curb"
        )
    )
    route_path = arguments.route_catalog.expanduser().resolve()
    if route_path.is_file():
        with np.load(route_path, allow_pickle=False) as route_data:
            level_count = np.asarray(route_data["level_count"])
            route_report = {
                "schema": str(route_data["schema"]),
                "archive_path": str(route_data["archive_path"]),
                "output_path": str(route_path),
                "clip_count": int(len(level_count)),
                "valid_clip_count": int(np.count_nonzero(level_count)),
                "maximum_level_count": int(np.max(level_count, initial=0)),
                "reused": True,
            }
    else:
        route_report = build_motion_route_profile_catalog(
            archive_path,
            route_path,
            worker_count=1,
        )
    fragment_report = build_stairs500_fragment_bank(
        archive_path,
        arguments.fragment_dir.expanduser().resolve(),
        model_path=arguments.model.expanduser().resolve(),
        clip_indices=curb_indices,
        workers=1,
    )
    result = {
        "archive": str(archive_path),
        "curb_clip_count": len(curb_indices),
        "route_catalog": route_report,
        "fragment_bank": fragment_report,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
