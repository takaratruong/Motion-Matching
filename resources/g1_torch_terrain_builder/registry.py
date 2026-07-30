"""Checked source registry for the expanded G1 terrain-motion corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from resources.g1_torch_stair_builder.corpus import STAIR_BASES


SourceFamily = Literal["grail", "stair-local", "stair-karen", "curb-chair"]
SourceAdapter = Literal["grail-record", "native-npz"]
TerrainAdapter = Literal[
    "grail-usd",
    "fixed-staircase",
    "scaled-staircase-084",
    "karen-metadata",
    "chair-object",
]

_FAMILIES = frozenset(("grail", "stair-local", "stair-karen", "curb-chair"))
_SOURCE_ADAPTERS = frozenset(("grail-record", "native-npz"))
_TERRAIN_ADAPTERS = frozenset(
    (
        "grail-usd",
        "fixed-staircase",
        "scaled-staircase-084",
        "karen-metadata",
        "chair-object",
    )
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class SourceSpec:
    logical_name: str
    family: SourceFamily
    source_adapter: SourceAdapter
    motion_relative_path: str
    motion_sha256: str
    terrain_adapter: TerrainAdapter
    geometry_relative_paths: tuple[str, ...]
    geometry_sha256: tuple[str, ...] = ()
    motion_to_terrain_xy_yaw: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    expected_layout: str = "g1-29dof-isaaclab-v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logical_name, str)
            or not self.logical_name
            or self.logical_name.strip() != self.logical_name
        ):
            raise ValueError("logical_name must be a non-empty trimmed string")
        if self.family not in _FAMILIES:
            raise ValueError("source family is invalid")
        if self.source_adapter not in _SOURCE_ADAPTERS:
            raise ValueError("source adapter is invalid")
        if self.terrain_adapter not in _TERRAIN_ADAPTERS:
            raise ValueError("terrain adapter is invalid")
        if (
            not isinstance(self.motion_relative_path, str)
            or not self.motion_relative_path
            or Path(self.motion_relative_path).is_absolute()
        ):
            raise ValueError("motion_relative_path must be non-empty and relative")
        if (
            not isinstance(self.motion_sha256, str)
            or len(self.motion_sha256) != 64
            or any(character not in _HEX for character in self.motion_sha256)
        ):
            raise ValueError("motion SHA-256 must be 64 lowercase hex characters")
        if (
            type(self.geometry_relative_paths) is not tuple
            or any(
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                for path in self.geometry_relative_paths
            )
        ):
            raise ValueError("geometry paths must be a tuple of relative paths")
        if (
            type(self.geometry_sha256) is not tuple
            or len(self.geometry_sha256) != len(self.geometry_relative_paths)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in _HEX for character in digest)
                for digest in self.geometry_sha256
            )
        ):
            raise ValueError(
                "geometry SHA-256 values must match the geometry paths"
            )
        if self.expected_layout != "g1-29dof-isaaclab-v1":
            raise ValueError("expected layout must be g1-29dof-isaaclab-v1")
        transform = self.motion_to_terrain_xy_yaw
        if (
            type(transform) is not tuple
            or len(transform) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in transform
            )
        ):
            raise ValueError(
                "motion-to-terrain transform must be a finite numeric tuple"
            )


@dataclass(frozen=True)
class ResolvedSource:
    spec: SourceSpec
    motion_path: Path
    geometry_paths: tuple[Path, ...]
    source_sha256: Mapping[str, str]


_FIXED_GEOMETRY = (
    "staircase/box_environment.xml",
    "staircase/box_models/box1.obj",
    "staircase/box_models/box2.obj",
    "staircase/box_models/box3.obj",
)
_FIXED_GEOMETRY_HASHES = (
    "d521bf511bce8511652af961d624579a312ec4350fbb2e5b5fab717e9b7965ca",
    "412cab2eb06f0501d20009823c7127c5293257d87b1e86ffdb3da2c7697ec5de",
    "8d3e9e204fa2646dbac9726ffe0da163f0b5811259981a76ee49c420079ff22c",
    "dc8679af4fa022512613f42e340567fe9ec27697e71613f9d1b9695a767b1ca4",
)
_SCALED_084_GEOMETRY = (
    "staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf",
    "staircase/box_models/box1.obj",
    "staircase/box_models/box2.obj",
    "staircase/box_models/box3.obj",
)
_SCALED_084_GEOMETRY_HASHES = (
    "78b560b01880e7202f5d36cf55cb66ab8ab98189fdb3f152c4406bf17557ac2f",
    "412cab2eb06f0501d20009823c7127c5293257d87b1e86ffdb3da2c7697ec5de",
    "8d3e9e204fa2646dbac9726ffe0da163f0b5811259981a76ee49c420079ff22c",
    "dc8679af4fa022512613f42e340567fe9ec27697e71613f9d1b9695a767b1ca4",
)

_GRAIL_ROBOT_HASHES = (
    "5d7b39d9d1386c4c152e05a9f4036d59b9bb14b2b2adfcb3c4bf1f2c4f012162",
    "d04040e0df9856d3778a3391505bfbd25cfeae43823df99a74dd1125039e1cc0",
    "fd5b5112d98f1645896195ff56b45c86570543a8402d167dbc85fb83f554f530",
    "b370a8cdbc25f52d1c86761ef702dddb0009a84e0486b7fe1cdeda3dd3283311",
)
_GRAIL_LOGICAL_NAMES = (
    "grail-stair-0000",
    "grail-stair-0001",
    "grail-stair-0002",
    "grail-stair-updown-0000",
)
_GRAIL_GEOMETRY_HASHES = (
    (
        "76dbb85eb46f98552294a54198c3b72cb8081a8df1e1440d2bf9382d94893efa",
        "3ec79c8726fcfaac82d3c4a4674b7cc2164bc6a9c6e2ee79b712c183727c619e",
    ),
    (
        "96e5aaea29483927ffd970562caa4fc0e8de32efe3301ee2d10d30d20465830d",
        "003f2ac8a9a28b6fc32526d9a59f30a8a433dd68f73c23aabedc90648948a0e9",
    ),
    (
        "b4cb01773a93473f893edf45baf28ec14b72e7e16884073dc5c7c2aab508e08e",
        "ed00df18a5c469ea959b0a552bc6ba8e4c86938b4f79b1bed09ef2d50982c3dc",
    ),
    (
        "2e8eb47306f86e730e853a3ea9ba0b650ce1fd36c5085b5370ec163ddb76649e",
        "2ae97849cdfcaa500be3632edb4133d5b864341fd633edf58a6931f5188e33f1",
    ),
)


def _grail_specs() -> tuple[SourceSpec, ...]:
    return tuple(
        SourceSpec(
            logical_name=name,
            family="grail",
            source_adapter="grail-record",
            motion_relative_path=f"robot/{base}.pkl",
            motion_sha256=digest,
            terrain_adapter="grail-usd",
            geometry_relative_paths=(
                f"objects/{base}.pkl",
                f"object_usd/{base}.usd",
            ),
            geometry_sha256=geometry_hashes,
        )
        for name, base, digest, geometry_hashes in zip(
            _GRAIL_LOGICAL_NAMES,
            STAIR_BASES,
            _GRAIL_ROBOT_HASHES,
            _GRAIL_GEOMETRY_HASHES,
        )
    )


_LOCAL_IDENTITIES = (
    ("staircase-v0", "stair-local", "staircase:v0/motion.npz", "265133ea0b1e460f40a7e15921cb0ae0ca270cadbe73ad22709761c35198b925", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("staircase-final", "stair-local", "staircase_final:v0/motion.npz", "846fc49ef5a6a628353f4644fb989b5341ee0a8f73f26aa72f1c09346a69065b", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("staircase-final-2", "stair-local", "staircase_final_2:v0/motion.npz", "72ed7871b8ef1feffc650bd36107707bfcc293b79741edda1455894d0d6ffb67", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("staircase-final-v3", "stair-local", "staircase_final_v3:v0/motion.npz", "22e80e8787a13a9c946b745a58fba6a0a4ffba658bbd50d8ecd9ffc2159af1b6", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("staircase-side-stepto", "stair-local", "staircase_side_stepto:v1/motion.npz", "e92ad315d8ed212a9fe9cd656199e5bf1389993530ddc2850a4dbe113e0ef18c", "scaled-staircase-084", _SCALED_084_GEOMETRY, _SCALED_084_GEOMETRY_HASHES),
    ("up-continuous-33", "stair-local", "up_continuous_33.npz:v0/motion.npz", "03da1a99a161327fd6650e8b304934e182ab34b3ff9ec8cc0978c69700a6696c", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("down-continuous-33", "stair-local", "down_continuous_33.npz:v0/motion.npz", "ca05856c8fe06f8d01e8a91bd300e118c28852b641a1c1154415c06a99c5eb3b", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("walk-up-33", "stair-local", "walk_up_33.npz:v0/motion.npz", "1158fb751c8d6ee1be19e878e12ab35042b7127559e44cb941279e0dbf0b378f", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("walk-down-33", "stair-local", "walk_down_33.npz:v0/motion.npz", "62bd560bd2d00bf6cae97e33bc8d1dea299f8a57a147acf0a08c7f1b79814704", "fixed-staircase", _FIXED_GEOMETRY, _FIXED_GEOMETRY_HASHES),
    ("up-continuous-karen", "stair-karen", "up_continuous_v2_karen_stairs:v0/motion.npz", "fc6df7b83c41923675d64933267ec9c46ec9c6d8330f58159f9e04740376299b", "karen-metadata", ("up_continuous_v2_karen_stairs/staircase_metadata.json",), ("31de02af9fc3747cde33d52cf15caa5ada890e2b54f9486c5d7de9142ce144dc",)),
    ("down-continuous-karen", "stair-karen", "down_continuous_v2_karen_stairs:v0/motion.npz", "693bd8f1469eab388404e824d3079cde7533a42540fb9d47c03d3bee01c12068", "karen-metadata", ("down_continuous_v2_karen_stairs/staircase_metadata.json",), ("f19f26fcfd94413ac303c07e23c9b5c1840533283056aef73fb7281ea9988ae3",)),
    ("walk-up-karen", "stair-karen", "walk_up_karen_stairs:v0/motion.npz", "4cee3698372634ab00352e112d121523f5db0ea412074fdc7e70279b295e33e5", "karen-metadata", ("walk_up_karen_stairs/staircase_metadata.json",), ("16dfb74939e1f521cb20c6c21e309811b0e2e40cc1a4ca1b8cf1aad5b6e71a57",)),
    ("walk-down-karen", "stair-karen", "walk_down_karen_stairs:v0/motion.npz", "cb4b352fc4595cb5014859e40a18bdbb6db27ca7f1157e44b0fa69f0197c97da", "karen-metadata", ("walk_down_karen_stairs/staircase_metadata.json",), ("902db01f84a8d0967aaa35b72036091cb9dc00e0218a944da855cb54094faf16",)),
    ("chair-step-v0", "curb-chair", "chair_step:v0/motion.npz", "25dd0116ae03fc0d6b5d6f9759e452a841f7c9c105035b3850864560101f3a85", "chair-object", (), ()),
    ("chair-step-v2", "curb-chair", "chair_step:v2/motion.npz", "e145bb8e6ed9a82fb4194b5f0dae805c2d6762caf9b2c725ae72a10b0debfdc5", "chair-object", (), ()),
    ("chair-step-v3", "curb-chair", "chair_step:v3/motion.npz", "ce3398dd6038766181ab10390d2f12a5f5b0f3fca7c05856f4b39f02afc18659", "chair-object", (), ()),
    ("chair-step-climbing-final", "curb-chair", "chair_step_climbing_final:v1/motion.npz", "69c545d85349033d1dbdc41041ea2baec730e3700f16dd1075c53fd08c99dc76", "chair-object", (), ()),
    ("chair-step-tracking-2", "curb-chair", "chair_step_tracking_2:v0/motion.npz", "5d5f78a6da8552db5b1297b8bd5ed566d00335e8b0d5db3d634e33069d118a3d", "chair-object", (), ()),
    ("chair-step-tracking-final", "curb-chair", "chair_step_tracking_final:v6/motion.npz", "80cb05c3a94503084d24152cbd8f7367f0f5b8c36c2a0eb19d4ebb5ce5b6b283", "chair-object", (), ()),
    ("chair-step-truncated", "curb-chair", "chair_step_truncated_converted.npz:v0/motion.npz", "f9a858ae4d9e0b097255bbb1dffcc6c214e3c8a49ae8701645a87a91c65b5256", "chair-object", (), ()),
)

_MOTION_TO_TERRAIN_XY_YAW = {
    "down-continuous-33": (-0.29, 3.99, -math.pi / 2.0),
    "staircase-final-v3": (-0.24, -0.10, math.pi / 2.0),
}


def _local_specs() -> tuple[SourceSpec, ...]:
    return tuple(
        SourceSpec(
            logical_name=name,
            family=family,
            source_adapter="native-npz",
            motion_relative_path=path,
            motion_sha256=digest,
            terrain_adapter=terrain_adapter,
            geometry_relative_paths=geometry,
            geometry_sha256=geometry_hashes,
            motion_to_terrain_xy_yaw=_MOTION_TO_TERRAIN_XY_YAW.get(
                name, (0.0, 0.0, 0.0)
            ),
        )
        for (
            name,
            family,
            path,
            digest,
            terrain_adapter,
            geometry,
            geometry_hashes,
        ) in _LOCAL_IDENTITIES
    )


CANDIDATE_SPECS = _grail_specs() + _local_specs()
if len(CANDIDATE_SPECS) != len(
    {spec.logical_name for spec in CANDIDATE_SPECS}
):
    raise RuntimeError("expanded terrain registry contains duplicate logical names")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(root: Path, relative: str, expected_sha256: str | None) -> Path:
    lexical = root / relative
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"registered path must remain inside source root: {relative}") from error
    if lexical.is_symlink() or not resolved.is_file():
        raise ValueError(f"registered source must be a real file: {relative}")
    if expected_sha256 is not None and _sha256(resolved) != expected_sha256:
        raise ValueError(f"registered source SHA-256 changed: {relative}")
    return resolved


def resolve_source_spec(
    spec: SourceSpec,
    *,
    source_root: str | Path,
    grail_root: str | Path,
) -> ResolvedSource:
    if not isinstance(spec, SourceSpec):
        raise TypeError("spec must be a SourceSpec")
    local = Path(source_root).resolve()
    grail = Path(grail_root).resolve()
    root = grail if spec.family == "grail" else local
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"source root must be a real directory: {root}")
    motion = _checked_file(root, spec.motion_relative_path, spec.motion_sha256)
    geometry = tuple(
        _checked_file(root, relative, digest)
        for relative, digest in zip(
            spec.geometry_relative_paths, spec.geometry_sha256
        )
    )
    hashes = {"motion": spec.motion_sha256}
    hashes.update(
        {
            f"geometry:{relative}": _sha256(path)
            for relative, path in zip(spec.geometry_relative_paths, geometry)
        }
    )
    return ResolvedSource(
        spec=spec,
        motion_path=motion,
        geometry_paths=geometry,
        source_sha256=MappingProxyType(hashes),
    )


def resolve_registered_sources(
    *,
    source_root: str | Path,
    grail_root: str | Path,
) -> tuple[ResolvedSource, ...]:
    return tuple(
        resolve_source_spec(
            spec, source_root=source_root, grail_root=grail_root
        )
        for spec in CANDIDATE_SPECS
    )
