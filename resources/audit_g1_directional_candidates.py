#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import struct
import tempfile


MANIFEST_SCHEMA = "g1-terrain-artifacts/v2"
AUDIT_SCHEMA = "g1-directional-candidate-audit/v1"
SUMMARY_SCHEMA = "g1-directional-candidate-summary/v1"
FEATURE_DIMENSIONS = 31
TOP_COUNT = 16
UINT32_MAX = 2 ** 32 - 1
HEADINGS = {
    "forward", "backward", "positive-x", "negative-x",
    "diagonal-positive-x", "diagonal-negative-x",
    "relative",
}
SOURCE_KEYS = {
    "name", "terrain_id", "source_fps", "source_frames", "output_frames",
    "range_start", "range_stop", "source_frame_map",
}
RECORD_KEYS = {
    "schema", "requested_frame", "query_bits_hex", "scene_id", "route",
    "heading", "accepted_frame", "eligible_count",
    "within_best_plus_0_25", "within_best_plus_1", "within_best_plus_4",
    "top_count", "top",
}
TOP_KEYS = {"frame", "range", "cost_bits"}
LOWER_HEX_WORD = re.compile(r"[0-9a-f]{8}")
LOWER_QUERY_BITS = re.compile(r"[0-9a-f]{248}")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key}")
        output[key] = value
    return output


def _load_json_text(text, label):
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeError, TypeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _integer(value, label, minimum=0, maximum=UINT32_MAX):
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{label} is not an integer in range")
    return value


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return float(value)


def _text(value, label):
    if (type(value) is not str or not value or not value.isprintable() or
            any(character in value for character in "\r\n")):
        raise ValueError(f"{label} is not safe nonempty text")
    return value


def _float_from_word(word, label, nonnegative=False):
    if type(word) is not str or LOWER_HEX_WORD.fullmatch(word) is None:
        raise ValueError(f"{label} is not one lowercase binary32 word")
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    if not math.isfinite(value) or (nonnegative and (int(word, 16) >> 31)):
        raise ValueError(f"{label} is not a finite nonnegative binary32 word")
    return value


def _query_bits(value, label):
    if type(value) is not str or LOWER_QUERY_BITS.fullmatch(value) is None:
        raise ValueError(f"{label} is not 31 lowercase binary32 words")
    for dimension in range(FEATURE_DIMENSIONS):
        word = value[dimension * 8:(dimension + 1) * 8]
        _float_from_word(word, f"{label} dimension {dimension}")
    return value


def _read_text(path, label):
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {label}") from error


def load_manifest(path):
    value = _load_json_text(_read_text(path, "manifest"), "manifest")
    if type(value) is not dict:
        raise ValueError("manifest root must be an object")
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema is invalid")
    if _integer(
            value.get("feature_dimensions"),
            "manifest feature_dimensions") != FEATURE_DIMENSIONS:
        raise ValueError("manifest must contain 31 matching features")
    frames = _integer(
        value.get("database_frames"), "manifest database_frames", 1)
    sources = value.get("sources")
    if type(sources) is not list or not sources:
        raise ValueError("manifest sources must be nonempty")
    if _integer(
            value.get("total_clips"), "manifest total_clips", 1) != len(sources):
        raise ValueError("manifest total_clips disagrees with sources")

    cursor = 0
    names = set()
    ranges = []
    for ordinal, source in enumerate(sources):
        label = f"manifest sources[{ordinal}]"
        if type(source) is not dict or set(source) != SOURCE_KEYS:
            raise ValueError(f"{label} keys are invalid")
        name = _text(source["name"], f"{label} name")
        if name in names:
            raise ValueError(f"duplicate manifest source name {name}")
        names.add(name)
        _text(source["terrain_id"], f"{label} terrain_id")
        source_fps = _finite_number(
            source["source_fps"], f"{label} source_fps")
        if source_fps not in (25.0, 50.0):
            raise ValueError(f"{label} source_fps is not canonical")
        source_frames = _integer(
            source["source_frames"], f"{label} source_frames", 1)
        output_frames = _integer(
            source["output_frames"], f"{label} output_frames", 1)
        start = _integer(source["range_start"], f"{label} range_start")
        stop = _integer(source["range_stop"], f"{label} range_stop", 1)
        if start != cursor or stop != start + output_frames or stop > frames:
            raise ValueError(f"{label} range is not the manifest partition")
        frame_map = source["source_frame_map"]
        if (type(frame_map) is not list or len(frame_map) != output_frames or
                any(type(frame) is not int or frame < 0 or
                    frame >= source_frames for frame in frame_map)):
            raise ValueError(f"{label} source_frame_map is invalid")
        ranges.append({
            "range": ordinal,
            "name": name,
            "range_start": start,
            "range_stop": stop,
        })
        cursor = stop
    if cursor != frames:
        raise ValueError("manifest source ranges do not cover database_frames")
    return {
        "schema": MANIFEST_SCHEMA,
        "database_frames": frames,
        "ranges": ranges,
    }


def _range_for_frame(ranges, frame, label):
    # Manifest ranges are validated ascending and contiguous. The bounded
    # linear lookup keeps the ordinal mapping explicit and mutation-sensitive.
    for source in ranges:
        if source["range_start"] <= frame < source["range_stop"]:
            return source
    raise ValueError(f"{label} is outside every manifest range")


def _validate_record(value, index, manifest_value):
    label = f"audit record {index}"
    if type(value) is not dict or set(value) != RECORD_KEYS:
        raise ValueError(f"{label} keys are invalid")
    if value["schema"] != AUDIT_SCHEMA:
        raise ValueError(f"{label} schema is invalid")
    requested_frame = _integer(
        value["requested_frame"], f"{label} requested_frame")
    query = _query_bits(value["query_bits_hex"], f"{label} query_bits_hex")
    scene = _text(value["scene_id"], f"{label} scene_id")
    route = _text(value["route"], f"{label} route")
    heading = value["heading"]
    if heading not in HEADINGS:
        raise ValueError(f"{label} heading is invalid")
    accepted_frame = _integer(
        value["accepted_frame"], f"{label} accepted_frame", 0,
        manifest_value["database_frames"] - 1)
    accepted_source = _range_for_frame(
        manifest_value["ranges"], accepted_frame,
        f"{label} accepted_frame")

    eligible = _integer(value["eligible_count"], f"{label} eligible_count")
    count_0_25 = _integer(
        value["within_best_plus_0_25"],
        f"{label} within_best_plus_0_25")
    count_1 = _integer(
        value["within_best_plus_1"], f"{label} within_best_plus_1")
    count_4 = _integer(
        value["within_best_plus_4"], f"{label} within_best_plus_4")
    if not (TOP_COUNT <= eligible and
            1 <= count_0_25 <= count_1 <= count_4 <= eligible):
        raise ValueError(f"{label} candidate counts are inconsistent")
    if _integer(value["top_count"], f"{label} top_count") != TOP_COUNT:
        raise ValueError(f"{label} must contain the full top-16")
    top = value["top"]
    if type(top) is not list or len(top) != TOP_COUNT:
        raise ValueError(f"{label} top must contain exactly 16 entries")

    validated_top = []
    seen_frames = set()
    previous_key = None
    for ordinal, entry in enumerate(top):
        entry_label = f"{label} top[{ordinal}]"
        if type(entry) is not dict or set(entry) != TOP_KEYS:
            raise ValueError(f"{entry_label} keys are invalid")
        frame = _integer(
            entry["frame"], f"{entry_label} frame", 0,
            manifest_value["database_frames"] - 1)
        range_ordinal = _integer(
            entry["range"], f"{entry_label} range", 0,
            len(manifest_value["ranges"]) - 1)
        if frame in seen_frames:
            raise ValueError(f"{label} contains a duplicate top frame")
        seen_frames.add(frame)
        source = _range_for_frame(
            manifest_value["ranges"], frame, f"{entry_label} frame")
        if source["range"] != range_ordinal:
            raise ValueError(
                f"{entry_label} range disagrees with manifest source")
        cost_word = entry["cost_bits"]
        cost = _float_from_word(
            cost_word, f"{entry_label} cost_bits", nonnegative=True)
        key = (cost, frame)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"{label} top entries are not stable sorted")
        previous_key = key
        validated_top.append({
            "frame": frame,
            "range": range_ordinal,
            "source_name": source["name"],
            "cost_bits": cost_word,
            "cost": cost,
        })
    return {
        "sort_key": (scene, route, heading, requested_frame),
        "requested_frame": requested_frame,
        "query_bits_hex": query,
        "scene_id": scene,
        "route": route,
        "heading": heading,
        "accepted_frame": accepted_frame,
        "accepted_source_name": accepted_source["name"],
        "eligible_count": eligible,
        "within_best_plus_0_25": count_0_25,
        "within_best_plus_1": count_1,
        "within_best_plus_4": count_4,
        "top": validated_top,
    }


def read_audit(path, manifest_value):
    text = _read_text(path, "candidate audit")
    lines = text.splitlines()
    if not lines:
        raise ValueError("candidate audit is empty")
    output = []
    previous_key = None
    previous_requested_frame = None
    chronological_scene_cycle = None
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"audit record {index} is empty")
        value = _load_json_text(line, f"audit record {index}")
        record = _validate_record(value, index, manifest_value)
        record_is_chronological_scene_cycle = (
            record["route"] == "scene-cycle" and
            record["heading"] == "relative")
        if chronological_scene_cycle is None:
            chronological_scene_cycle = record_is_chronological_scene_cycle
        elif record_is_chronological_scene_cycle != chronological_scene_cycle:
            raise ValueError("audit mixes canonical and scene-cycle ordering")
        if chronological_scene_cycle:
            if (previous_requested_frame is not None and
                    record["requested_frame"] <= previous_requested_frame):
                raise ValueError("audit records are duplicate or unsorted")
        elif (previous_key is not None and
              record["sort_key"] <= previous_key):
            raise ValueError("audit records are duplicate or unsorted")
        previous_key = record["sort_key"]
        previous_requested_frame = record["requested_frame"]
        output.append(record)
    return output


def _count_summary(records, field):
    values = [record[field] for record in records]
    return {
        "minimum": min(values),
        "median": float(statistics.median(values)),
    }


def _cell_summary(key, records, ranges):
    first_costs = [record["top"][0]["cost"] for record in records]
    first_cost_words = [record["top"][0]["cost_bits"] for record in records]
    minimum_index = min(
        range(len(first_costs)), key=lambda index: first_costs[index])
    covered_ranges = sorted({
        entry["range"]
        for record in records
        for entry in record["top"]
    })
    range_by_ordinal = {source["range"]: source for source in ranges}
    return {
        "scene_id": key[0],
        "route": key[1],
        "heading": key[2],
        "samples": len(records),
        "candidate_counts": {
            "eligible": _count_summary(records, "eligible_count"),
            "within_best_plus_0_25": _count_summary(
                records, "within_best_plus_0_25"),
            "within_best_plus_1": _count_summary(
                records, "within_best_plus_1"),
            "within_best_plus_4": _count_summary(
                records, "within_best_plus_4"),
        },
        "top_costs": {
            "bits": first_cost_words,
            "top16_bits": [
                [entry["cost_bits"] for entry in record["top"]]
                for record in records
            ],
            "minimum_bits": first_cost_words[minimum_index],
            "minimum": first_costs[minimum_index],
            "median": float(statistics.median(first_costs)),
        },
        "top16_source_coverage": {
            "unique_ranges": covered_ranges,
            "unique_source_count": len(covered_ranges),
            "source_names": [
                range_by_ordinal[ordinal]["name"]
                for ordinal in covered_ranges
            ],
        },
    }


def summarize(manifest_path, audit_path):
    manifest_value = load_manifest(manifest_path)
    records = read_audit(audit_path, manifest_value)
    records_by_cell = {}
    for record in records:
        cell_key = record["sort_key"][:3]
        records_by_cell.setdefault(cell_key, []).append(record)
    grouped = [
        _cell_summary(
            cell_key, records_by_cell[cell_key], manifest_value["ranges"])
        for cell_key in sorted(records_by_cell)
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "manifest_schema": manifest_value["schema"],
        "database_frames": manifest_value["database_frames"],
        "source_count": len(manifest_value["ranges"]),
        "record_count": len(records),
        "ranges": manifest_value["ranges"],
        "cells": grouped,
    }


def _absolute(path, label):
    if type(path) is not str or not os.path.isabs(path):
        raise ValueError(f"{label} must be an absolute path")
    return path


def write_summary(manifest_path, audit_path, output_path):
    _absolute(manifest_path, "manifest")
    _absolute(audit_path, "audit")
    _absolute(output_path, "output")
    value = summarize(manifest_path, audit_path)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":")) + "\n").encode("ascii")
    directory = os.path.dirname(output_path)
    if not os.path.isdir(directory):
        raise ValueError("output directory does not exist")
    descriptor = None
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".g1-candidate-summary-", dir=directory)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise ValueError("cannot publish candidate summary") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and summarize behavior-inert G1 candidates")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    write_summary(
        _absolute(arguments.manifest, "manifest"),
        _absolute(arguments.audit, "audit"),
        _absolute(arguments.output, "output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
