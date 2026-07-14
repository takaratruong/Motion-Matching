from pathlib import Path
from typing import Sequence

import numpy as np
from pxr import Usd, UsdGeom

from .schema import SourcePaths


def read_usd_dimensions(path: Path) -> np.ndarray:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open object USD: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    dimensions = np.asarray(world.GetSize(), np.float32)
    if (
        dimensions.shape != (3,)
        or not np.isfinite(dimensions).all()
        or np.any(dimensions <= 0)
    ):
        raise ValueError(f"invalid object bounds in {path}: {dimensions}")
    return dimensions


def build_dimension_catalog(
    paths: Sequence[SourcePaths],
) -> dict[str, np.ndarray]:
    catalog: dict[str, np.ndarray] = {}
    for source in paths:
        size = read_usd_dimensions(source.object_usd)
        if source.object_id in catalog and not np.allclose(
            catalog[source.object_id], size, rtol=0.0, atol=1e-3
        ):
            raise ValueError(
                f"inconsistent USD bounds for object {source.object_id}"
            )
        catalog[source.object_id] = size
    return catalog
