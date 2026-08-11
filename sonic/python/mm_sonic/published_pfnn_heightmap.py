"""Export one released PFNN demo heightmap as a z-up metre NPZ mesh."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


HORIZONTAL_SCALE_CM = 3.937007874
VERTICAL_SCALE_CM = 3.0


def _released_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        values = [float(value) for value in line.split()]
        if not values:
            continue
        # Preserve the release's `while (iss) { iss >> f; push_back(f); }`
        # behavior: the final value is appended once more after EOF.
        rows.append([*values, values[-1]])
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("PFNN heightmap must be a nonempty rectangular text grid")
    data = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("PFNN heightmap must be finite")
    return data


def build_mesh(
    path: Path, *, uniform_scale: float = 1.0, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(uniform_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("uniform_scale must be finite and positive")
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    data = _released_rows(Path(path))
    width, height = data.shape
    x_indices = np.arange(0, width, stride, dtype=np.int64)
    z_indices = np.arange(0, height, stride, dtype=np.int64)
    if x_indices[-1] != width - 1:
        x_indices = np.append(x_indices, width - 1)
    if z_indices[-1] != height - 1:
        z_indices = np.append(z_indices, height - 1)
    xs = HORIZONTAL_SCALE_CM * x_indices.astype(np.float64)
    xs -= HORIZONTAL_SCALE_CM * width / 2.0
    zs = HORIZONTAL_SCALE_CM * z_indices.astype(np.float64)
    zs -= HORIZONTAL_SCALE_CM * height / 2.0
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")
    source_height = VERTICAL_SCALE_CM * (data - float(data.mean()))
    sampled_height = source_height[np.ix_(x_indices, z_indices)]
    vertices = scale * np.column_stack(
        (
            grid_x.reshape(-1) / 100.0,
            -grid_z.reshape(-1) / 100.0,
            sampled_height.T.reshape(-1) / 100.0,
        )
    )
    mesh_width = len(x_indices)
    mesh_height = len(z_indices)
    faces = np.asarray(
        [
            triangle
            for y in range(mesh_height - 1)
            for x in range(mesh_width - 1)
            for triangle in (
                (
                    x + y * mesh_width,
                    x + (y + 1) * mesh_width,
                    x + 1 + y * mesh_width,
                ),
                (
                    x + 1 + (y + 1) * mesh_width,
                    x + 1 + y * mesh_width,
                    x + (y + 1) * mesh_width,
                ),
            )
        ],
        dtype=np.int32,
    )
    return vertices, faces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heightmap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform-scale", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)
    arguments = parser.parse_args(argv)
    vertices, faces = build_mesh(
        arguments.heightmap,
        uniform_scale=arguments.uniform_scale,
        stride=arguments.stride,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, vertices=vertices, faces=faces)
    print(
        f"wrote {arguments.output} with {len(vertices)} vertices and {len(faces)} faces"
    )


if __name__ == "__main__":
    main()
