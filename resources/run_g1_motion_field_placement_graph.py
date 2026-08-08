#!/usr/bin/env python3
"""Combine hard-gated edge-placement scans into one explicit lookup graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from mm_sonic.joints import ContractError


def build_placement_graph(
    *, reports: Sequence[dict[str, object]]
) -> dict[str, object]:
    owned = tuple(reports)
    if not owned or any(not isinstance(item, dict) for item in owned):
        raise ContractError("placement graph reports are invalid")
    scenes = {str(item.get("query_scene", "")) for item in owned}
    templates = [str(item.get("template_edge_id", "")) for item in owned]
    if "" in scenes or len(scenes) != 1 or "" in templates or len(set(templates)) != len(templates):
        raise ContractError("placement graph reports are incompatible")
    cells = []
    edges = []
    for report in owned:
        records = report.get("records")
        if not isinstance(records, list) or len(records) != report.get("candidate_count"):
            raise ContractError("placement graph report counts are invalid")
        admitted_count = 0
        for record in records:
            cell = dict(record)
            cell["template_edge_id"] = report["template_edge_id"]
            if record.get("validated") is True:
                if (
                    not record.get("artifact")
                    or not isinstance(record.get("artifact_sha256"), str)
                    or len(record["artifact_sha256"]) != 64
                    or record.get("rejections") != []
                ):
                    raise ContractError("admitted placement cell is invalid")
                cell["status"] = "admitted"
                edges.append(cell)
                admitted_count += 1
            else:
                if not record.get("rejections"):
                    raise ContractError("rejected placement cell lacks reason")
                cell["status"] = "rejected"
            cells.append(cell)
        if admitted_count != report.get("accepted_count"):
            raise ContractError("placement graph report counts are invalid")
    occupied = sorted(
        {
            line_id
            for edge in edges
            for line_id in (edge["from_line_id"], edge["to_line_id"])
        }
    )
    return {
        "schema": "g1-object-motion-field-placement-graph/v1",
        "query_scene": next(iter(scenes)),
        "template_count": len(owned),
        "candidate_cell_count": len(cells),
        "edge_count": len(edges),
        "rejected_cell_count": len(cells) - len(edges),
        "occupied_line_count": len(occupied),
        "occupied_line_ids": occupied,
        "edges": edges,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        reports = tuple(
            json.loads(path.read_text("utf-8")) for path in args.scan_reports
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("placement scan report is invalid") from error
    graph = build_placement_graph(reports=reports)
    for edge in graph["edges"]:
        artifact = Path(edge["artifact"])
        try:
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except OSError as error:
            raise ContractError("placement graph artifact is unavailable") from error
        if digest != edge["artifact_sha256"]:
            raise ContractError("placement graph artifact hash mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_cell_count": graph["candidate_cell_count"],
                "edge_count": graph["edge_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
