"""Generate deterministic runtime metadata for the fourteen G1 arm hinges."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree


ARM_JOINTS = {
    "Left": (
        (17, "left_shoulder_pitch_joint"),
        (18, "left_shoulder_roll_joint"),
        (19, "left_shoulder_yaw_joint"),
        (20, "left_elbow_joint"),
        (21, "left_wrist_roll_joint"),
        (22, "left_wrist_pitch_joint"),
        (23, "left_wrist_yaw_joint"),
    ),
    "Right": (
        (24, "right_shoulder_pitch_joint"),
        (25, "right_shoulder_roll_joint"),
        (26, "right_shoulder_yaw_joint"),
        (27, "right_elbow_joint"),
        (28, "right_wrist_roll_joint"),
        (29, "right_wrist_pitch_joint"),
        (30, "right_wrist_yaw_joint"),
    ),
}


def _numbers(value: str | None, count: int, description: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"missing {description}")
    try:
        numbers = tuple(float(component) for component in value.split())
    except ValueError as error:
        raise ValueError(f"invalid {description}: {value!r}") from error
    if len(numbers) != count or not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"invalid {description}: {value!r}")
    return numbers


def _canonical(value: float) -> float:
    return 0.0 if abs(value) < 5.0e-13 else value


def _zup_to_yup_vector(value: tuple[float, float, float]) -> tuple[float, ...]:
    x, y, z = value
    return tuple(_canonical(component) for component in (x, z, -y))


def _zup_to_yup_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, ...]:
    norm = math.sqrt(sum(component * component for component in value))
    if norm < 1.0e-12:
        raise ValueError("body quaternion has zero length")
    w, x, y, z = (component / norm for component in value)
    return tuple(_canonical(component) for component in (w, x, z, -y))


def _format_float(value: float) -> str:
    value = _canonical(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-4:
        return f"{value:.10f}"
    return f"{value:.9g}"


def _format_quaternion(value: tuple[float, ...]) -> str:
    components = []
    for item in value:
        text = _format_float(item)
        components.append(text if text in {"-1", "0", "1"} else f"{text}F")
    return "{" + ", ".join(components) + "}"


def _format_axis(value: tuple[float, ...]) -> str:
    return "{" + ",".join(_format_float(item) for item in value) + "}"


def _joint_records(root: ElementTree.Element):
    records = {}
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name is None:
                continue
            if name in records:
                raise ValueError(f"duplicate joint {name}")
            records[name] = (body, joint)
    return records


def _descriptor(
    records,
    bone: int,
    joint_name: str,
) -> str:
    if joint_name not in records:
        raise ValueError(f"missing G1 arm joint {joint_name}")
    body, joint = records[joint_name]
    rest = _zup_to_yup_quaternion(
        _numbers(
            body.get("quat", "1 0 0 0"),
            4,
            f"body quaternion for {joint_name}",
        )
    )
    axis = _zup_to_yup_vector(
        _numbers(joint.get("axis"), 3, f"axis for {joint_name}")
    )
    axis_length = math.sqrt(sum(component * component for component in axis))
    if abs(axis_length - 1.0) > 1.0e-9:
        raise ValueError(f"non-unit axis for {joint_name}")
    lower, upper = _numbers(
        joint.get("range"), 2, f"range for {joint_name}"
    )
    if lower > upper:
        raise ValueError(f"reversed range for {joint_name}")
    return (
        f"    {{{bone}, {_format_quaternion(rest)}, {_format_axis(axis)}, "
        f"{_format_float(lower)}F, {_format_float(upper)}F}},"
    )


def generate_header(g1_xml: Path) -> str:
    """Return the canonical C++ header derived from ``g1_xml``."""
    root = ElementTree.parse(Path(g1_xml)).getroot()
    records = _joint_records(root)
    lines = [
        "#pragma once",
        "",
        '#include "quat.h"',
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace interaction {",
        "",
        "struct HingeJoint {",
        "    int32_t bone;",
        "    quat rest_rotation;",
        "    vec3 axis;",
        "    float lower;",
        "    float upper;",
        "};",
        "",
    ]
    for side, joints in ARM_JOINTS.items():
        lines.append(
            f"inline constexpr std::array<HingeJoint, 7> k{side}Arm = {{{{"
        )
        lines.extend(
            _descriptor(records, bone, joint_name)
            for bone, joint_name in joints
        )
        lines.extend(("}};", ""))
    lines.extend(("}  // namespace interaction", ""))
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.write_text(
        generate_header(args.g1_xml), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
