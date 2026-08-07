"""Materialize an order-preserving subset of an existing SONIC terrain bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil


def subset_bundle(
    source: Path,
    output: Path,
    *,
    include_pilot_regexes: tuple[str, ...],
    exclude_pilot_regexes: tuple[str, ...],
) -> dict[str, object]:
    import joblib

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")

    clip_records = json.loads((source / "clips.json").read_text())
    provenance = json.loads((source / "bundle_provenance.json").read_text())
    provenance_rows = provenance["clips"]
    if len(clip_records) != len(provenance_rows):
        raise ValueError("clips.json and bundle_provenance.json disagree on clip count")

    include_patterns = tuple(re.compile(value) for value in include_pilot_regexes)
    exclude_patterns = tuple(re.compile(value) for value in exclude_pilot_regexes)
    selected_indices = [
        index
        for index, row in enumerate(provenance_rows)
        if (
            not include_patterns
            or any(
                pattern.search(str(row.get("pilot_label", "")))
                for pattern in include_patterns
            )
        )
        and not any(
            pattern.search(str(row.get("pilot_label", "")))
            for pattern in exclude_patterns
        )
    ]
    if not selected_indices:
        raise ValueError("subset filter rejected every clip")

    selected_records = [clip_records[index] for index in selected_indices]
    selected_provenance = [provenance_rows[index] for index in selected_indices]
    selected_stems = [str(row["stem"]) for row in selected_records]
    if len(selected_stems) != len(set(selected_stems)):
        raise ValueError("source bundle contains duplicate stems")

    merged = joblib.load(source / "motion_lib_merged.pkl")
    objects = joblib.load(source / "objects.pkl")
    missing = [stem for stem in selected_stems if stem not in merged or stem not in objects]
    if missing:
        raise ValueError(f"source bundle dictionaries are missing stems: {missing[:3]}")

    robot_output = output / "robot"
    terrain_output = output / "object_usd"
    robot_output.mkdir(parents=True)
    terrain_output.mkdir(parents=True)
    for stem in selected_stems:
        shutil.copy2(source / "robot" / f"{stem}.pkl", robot_output / f"{stem}.pkl")
        shutil.copy2(
            source / "object_usd" / f"{stem}.usd",
            terrain_output / f"{stem}.usd",
        )

    joblib.dump({stem: merged[stem] for stem in selected_stems}, output / "motion_lib_merged.pkl")
    joblib.dump({stem: objects[stem] for stem in selected_stems}, output / "objects.pkl")
    shutil.copy2(
        terrain_output / f"{selected_stems[0]}.usd",
        output / "flat_placeholder.usd",
    )
    (output / "clips.json").write_text(
        json.dumps(selected_records, indent=2, sort_keys=True) + "\n"
    )

    provenance["clip_count"] = len(selected_provenance)
    provenance["clips"] = selected_provenance
    provenance["subset"] = {
        "source": str(source),
        "include_pilot_regexes": list(include_pilot_regexes),
        "exclude_pilot_regexes": list(exclude_pilot_regexes),
        "source_clip_count": len(provenance_rows),
        "dropped_clip_count": len(provenance_rows) - len(selected_provenance),
    }
    (output / "bundle_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return {
        "clip_count": len(selected_provenance),
        "dropped_clip_count": len(provenance_rows) - len(selected_provenance),
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-pilot-regex", action="append", default=[])
    parser.add_argument("--exclude-pilot-regex", action="append", default=[])
    arguments = parser.parse_args(argv)
    result = subset_bundle(
        arguments.source,
        arguments.output,
        include_pilot_regexes=tuple(arguments.include_pilot_regex),
        exclude_pilot_regexes=tuple(arguments.exclude_pilot_regex),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
