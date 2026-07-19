#!/usr/bin/env python3
"""Create the deterministic pre-IK motion-quality baseline report."""

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from resources.check_g1_runtime_log import check_rows, read_rows


BASELINE_ROUTES = (
    "stairs-shallow/flat-positive-z",
    "stairs-shallow/flat-positive-x",
    "stairs-standard/ascent-landing-descent",
    "ramp-10-up-down/up-landing-down",
    "stairs-standard/ascent-landing-descent@positive-x",
    "mixed-multilevel/tangent-level-boundary",
)
SOURCE_FAMILIES = ("flat", "curb", "slope", "stair")


def _number(row, name, index):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {index}: invalid {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"row {index}: non-finite {name}")
    return value


def _integer(row, name, index):
    value = _number(row, name, index)
    if value != int(value):
        raise ValueError(f"row {index}: non-integer {name}")
    return int(value)


def _metric(value):
    if not math.isfinite(value):
        raise ValueError("motion quality metric is non-finite")
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0.0 else rounded


def _percentile(values, percentile):
    if not values:
        raise ValueError("motion quality percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return _metric(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    alpha = position - lower
    value = ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha
    return _metric(value)


def _source_family(source_terrain):
    normalized = source_terrain.strip().lower()
    if normalized == "flat" or normalized.startswith("flat_"):
        return "flat"
    if "stair" in normalized:
        return "stair"
    if "slope" in normalized or "ramp" in normalized:
        return "slope"
    if "curb" in normalized:
        return "curb"
    raise ValueError(
        f"unknown source terrain family {source_terrain!r}")


def _clearance_metrics(values):
    minimum = min(values)
    return {
        "minimum_m": _metric(minimum),
        "p05_m": _percentile(values, 0.05),
        "penetrating_frames": sum(value < 0.0 for value in values),
        "worst_penetration_m": _metric(max(0.0, -minimum)),
    }


def summarize_route(rows):
    """Validate one exact runtime log and return deterministic metrics."""
    check_rows(rows)
    family_counts = {family: 0 for family in SOURCE_FAMILIES}
    selected_costs = []
    rendered_clearances = []
    sole_clearances = []
    penetrating_probes = 0
    contact_frames = [0, 0]
    maximum_stance = [0.0, 0.0]
    total_slip = [0.0, 0.0]
    previous_slip = [0.0, 0.0]

    for index, row in enumerate(rows):
        family_counts[_source_family(row["source_terrain"])] += 1
        selected_costs.append(_number(row, "selected_cost", index))
        rendered_clearances.append(
            _number(row, "rendered_min_clearance", index))
        sole_clearances.append(_number(row, "sole_min_clearance", index))
        for side in ("left", "right"):
            for probe in range(4):
                if _number(
                        row, f"{side}_sole_clearance_{probe}", index) < 0.0:
                    penetrating_probes += 1

        for foot, side in enumerate(("left", "right")):
            contact = _integer(row, f"{side}_contact", index)
            reset = _integer(row, f"{side}_stance_slip_reset", index)
            slip = _number(row, f"{side}_stance_slip", index)
            contact_frames[foot] += contact
            maximum_stance[foot] = max(maximum_stance[foot], slip)
            if reset:
                previous_slip[foot] = 0.0
            else:
                delta = slip - previous_slip[foot]
                if delta < 0.0:
                    raise ValueError(
                        f"row {index}: {side} cumulative stance slip decreased")
                total_slip[foot] += delta
            previous_slip[foot] = slip

    sole_metrics = _clearance_metrics(sole_clearances)
    sole_metrics["penetrating_probes"] = penetrating_probes
    return {
        "frames": len(rows),
        "source_family_frames": family_counts,
        "search_count": sum(_integer(row, "searched", index)
                            for index, row in enumerate(rows)),
        "transition_count": sum(_integer(row, "transitioned", index)
                                for index, row in enumerate(rows)),
        "transition_rate": _metric(
            sum(_integer(row, "transitioned", index)
                for index, row in enumerate(rows)) / len(rows)),
        "selected_cost": {
            "p50": _percentile(selected_costs, 0.50),
            "p90": _percentile(selected_costs, 0.90),
            "p95": _percentile(selected_costs, 0.95),
            "max": _metric(max(selected_costs)),
        },
        "rendered_joint_clearance": _clearance_metrics(
            rendered_clearances),
        "sole_clearance": sole_metrics,
        "stance_slip": {
            "contact_frames": {
                "left": contact_frames[0], "right": contact_frames[1],
            },
            "maximum_stance_m": {
                "left": _metric(maximum_stance[0]),
                "right": _metric(maximum_stance[1]),
            },
            "total_m": {
                "left": _metric(total_slip[0]),
                "right": _metric(total_slip[1]),
            },
        },
        "route_complete": bool(_integer(rows[-1], "route_complete", len(rows) - 1)),
        "ik_enabled_frames": sum(_integer(row, "ik_enabled", index)
                                 for index, row in enumerate(rows)),
    }


def build_baseline_report(logs):
    """Freeze old-pack metrics and candidate comparison directions."""
    missing = sorted(set(BASELINE_ROUTES) - set(logs))
    unexpected = sorted(set(logs) - set(BASELINE_ROUTES))
    if missing or unexpected:
        raise ValueError(
            f"baseline route set mismatch: missing={missing} unexpected={unexpected}")
    routes = {
        route: summarize_route(logs[route])
        for route in sorted(BASELINE_ROUTES)
    }
    lateral = routes[BASELINE_ROUTES[1]]
    stance_total = sum(
        metrics["stance_slip"]["total_m"][side]
        for metrics in routes.values()
        for side in ("left", "right"))
    worst_sole_penetration = max(
        metrics["sole_clearance"]["worst_penetration_m"]
        for metrics in routes.values())
    return {
        "schema": "g1-motion-quality-baseline/v1",
        "ik_enabled": False,
        "sample_rate_hz": 25,
        "metric_definitions": {
            "selected_cost_percentiles":
                "Type-7 linear interpolation over every finite selected_cost row",
            "transition_rate": "transition_count / frame_count",
            "rendered_joint_clearance":
                "existing rendered_min_clearance across seven joint probes",
            "sole_clearance":
                "minimum of eight authoritative physical sole clearances per frame",
            "stance_slip_total_m":
                "sum of positive increments in cumulative per-stance sole-centroid slip",
            "penetration": "strictly negative vertical clearance",
        },
        "routes": routes,
        "candidate_thresholds": {
            "flat_wrong_family_frames": {
                "comparison": "equal",
                "target": 0,
                "routes": list(BASELINE_ROUTES[:2]),
            },
            "activated_terrain_wrong_family_frames": {
                "comparison": "equal",
                "target": 0,
                "routes": list(BASELINE_ROUTES[2:]),
                "activation": "first accepted non-flat classifier activation",
            },
            "lateral_selected_cost_p95": {
                "comparison": "strictly_less_than_baseline",
                "baseline": lateral["selected_cost"]["p95"],
                "route": BASELINE_ROUTES[1],
            },
            "lateral_transition_rate": {
                "comparison": "strictly_less_than_baseline",
                "baseline": lateral["transition_rate"],
                "route": BASELINE_ROUTES[1],
            },
            "stance_slip_total_m": {
                "comparison": "strictly_less_than_baseline",
                "baseline": _metric(stance_total),
                "aggregation": "sum over both feet and all routes",
            },
            "sole_worst_penetration_m": {
                "comparison": "strictly_less_than_baseline",
                "baseline": _metric(worst_sole_penetration),
                "aggregation": "maximum over all routes",
            },
        },
    }


def canonical_json(report):
    return json.dumps(
        report,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ) + "\n"


def _parse_logs(values):
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--log must be ROUTE=PATH, got {value!r}")
        route, path = value.split("=", 1)
        if not route or not path:
            raise ValueError(f"--log must be ROUTE=PATH, got {value!r}")
        if route in paths:
            raise ValueError(f"duplicate baseline route {route!r}")
        paths[route] = path
    return paths


def _write_transactionally(path, text):
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize deterministic pre-IK G1 runtime logs")
    parser.add_argument(
        "--log", action="append", default=[], metavar="ROUTE=PATH",
        help="exact route identifier and runtime CSV (repeat six times)")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        paths = _parse_logs(arguments.log)
        logs = {route: read_rows(path) for route, path in paths.items()}
        report = build_baseline_report(logs)
        _write_transactionally(arguments.output, canonical_json(report))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
