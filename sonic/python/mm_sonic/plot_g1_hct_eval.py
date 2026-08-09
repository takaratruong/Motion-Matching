"""Plot intended-versus-realized routes and coverage from a G1 HCT eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quaternion_yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def intended_path(root_quaternion: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
    yaw = quaternion_yaw_wxyz(root_quaternion)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    velocity = np.empty((len(command), 2), dtype=np.float64)
    velocity[:, 0] = cosine * command[:, 0] - sine * command[:, 1]
    velocity[:, 1] = sine * command[:, 0] + cosine * command[:, 1]
    return np.vstack((np.zeros((1, 2)), np.cumsum(velocity[:-1] * dt, axis=0)))


def _best_nonflat_by_program(data: np.lib.npyio.NpzFile, program: str) -> int | None:
    program_name = np.asarray(data["program_name"]).astype(str)
    terrain_name = np.asarray(data["terrain_name"]).astype(str)
    clean = np.asarray(data["clean_envs"], dtype=bool)
    selected = (program_name == program) & (terrain_name != "flat") & clean
    indices = np.flatnonzero(selected)
    if not len(indices):
        return None
    score = np.asarray(data["command_rmse"], dtype=np.float64)[indices]
    return int(indices[int(np.argmin(score))])


def plot_routes(data: np.lib.npyio.NpzFile, output: Path) -> dict:
    programs = list(dict.fromkeys(np.asarray(data["program_name"]).astype(str).tolist()))
    root = np.asarray(data["root_pos"], dtype=np.float64)
    quaternion = np.asarray(data["root_quat"], dtype=np.float64)
    command = np.asarray(data["commands"], dtype=np.float64)
    terrain = np.asarray(data["terrain_name"]).astype(str)
    rmse = np.asarray(data["command_rmse"], dtype=np.float64)
    dt = float(data["dt"])

    figure, axes = plt.subplots(2, 5, figsize=(19, 7.6), constrained_layout=True)
    rows = []
    for axis, program in zip(axes.flat, programs, strict=True):
        env = _best_nonflat_by_program(data, program)
        axis.set_title(program.replace("_", " "), fontsize=10)
        if env is None:
            axis.text(0.5, 0.5, "no clean example", ha="center", va="center")
            axis.set_axis_off()
            rows.append({"program": program, "status": "no_clean_example"})
            continue
        actual = root[:, env, :2] - root[0, env, :2]
        desired = intended_path(quaternion[:, env], command[:, env], dt)
        axis.plot(desired[:, 0], desired[:, 1], "--", color="#d62728", linewidth=2.0, label="intended")
        axis.plot(actual[:, 0], actual[:, 1], color="#1f77b4", linewidth=2.0, label="actual")
        axis.scatter([0.0], [0.0], s=35, color="black", marker="o", zorder=4)
        axis.scatter([actual[-1, 0]], [actual[-1, 1]], s=45, color="#1f77b4", marker="x", zorder=4)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.set_xlabel("world x from start (m)")
        axis.set_ylabel("world y from start (m)")
        axis.text(
            0.02, 0.98, f"{terrain[env]}\nRMSE {rmse[env]:.2f}",
            transform=axis.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        rows.append({
            "program": program,
            "status": "plotted",
            "env": env,
            "terrain": terrain[env],
            "command_rmse": float(rmse[env]),
        })
    axes.flat[0].legend(loc="lower right", fontsize=8)
    level = int(np.asarray(data["terrain_level_initial"])[0])
    figure.suptitle(
        f"Robot-local joystick intent versus actual G1 path | unseen terrain seed | level {level}/9",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {"routes": rows, "output": str(output.resolve())}


def plot_summary(data: np.lib.npyio.NpzFile, output: Path) -> dict:
    terrain = np.asarray(data["terrain_name"]).astype(str)
    clean = np.asarray(data["clean_envs"], dtype=bool)
    rmse = np.asarray(data["command_rmse"], dtype=np.float64)
    names = list(dict.fromkeys(terrain.tolist()))
    pass_rate = []
    clean_rmse = []
    for name in names:
        selected = terrain == name
        passed = selected & clean
        pass_rate.append(float(passed.sum() / selected.sum()))
        clean_rmse.append(float(rmse[passed].mean()) if passed.any() else np.nan)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    positions = np.arange(len(names))
    axes[0].bar(positions, pass_rate, color="#2ca02c")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("no-reset fraction")
    axes[0].set_title("12-second survival at level 9/9")
    axes[1].bar(positions, clean_rmse, color="#1f77b4")
    axes[1].set_ylabel("mixed velocity/yaw RMSE")
    axes[1].set_title("command tracking among clean runs")
    for axis in axes:
        axis.set_xticks(positions, [name.replace("_", "\n") for name in names], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {
        "output": str(output.resolve()),
        "terrain": {
            name: {"clean_fraction": pass_rate[index], "clean_command_rmse": clean_rmse[index]}
            for index, name in enumerate(names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.source, allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.source.resolve()),
        "routes": plot_routes(data, args.output_dir / "intended_vs_actual_routes.png"),
        "summary": plot_summary(data, args.output_dir / "terrain_performance.png"),
    }
    path = args.output_dir / "figures.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
