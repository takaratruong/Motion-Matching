import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
)
from resources.g1_terrain_builder.schema import SourceClip

from .schema import GMRSource, ReviewCorpus


_MOTIONS_NAME = "review_motions.npz"
_MANIFEST_NAME = "manifest.json"
_PROPOSALS_NAME = "proposals.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_review_sources(
    sources: list[GMRSource],
    kinematics: G1Kinematics,
    *,
    excluded_sequence_ids: tuple[str, ...] = (),
) -> ReviewCorpus:
    if not sources:
        raise ValueError("review conversion requires included sources")
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    source_frames: list[np.ndarray] = []
    starts: list[int] = []
    stops: list[int] = []
    reports: list[dict[str, float]] = []
    offset = 0
    for source in sources:
        source.validate()
        if source.disposition != "included":
            raise ValueError(f"excluded source {source.sequence_id} cannot convert")
        motion, skeleton, report = convert_source_clip(
            SourceClip(
                source.sequence_id,
                source.fps,
                source.qpos,
                source.source_frames,
                "reach",
            ),
            kinematics,
            target_fps=25.0,
        )
        if skeleton.signature() != G1_SKELETON.signature():
            raise ValueError(
                f"{source.sequence_id}: canonical skeleton signature mismatch"
            )
        starts.append(offset)
        offset += len(motion.positions)
        stops.append(offset)
        positions.append(motion.positions.astype(np.float32, copy=False))
        rotations.append(motion.rotations.astype(np.float32, copy=False))
        source_frames.append(motion.source_frames.astype(np.int32, copy=False))
        reports.append({key: float(value) for key, value in report.items()})

    corpus = ReviewCorpus(
        fps=25.0,
        positions=np.concatenate(positions, axis=0),
        rotations=np.concatenate(rotations, axis=0),
        range_starts=np.asarray(starts, np.int32),
        range_stops=np.asarray(stops, np.int32),
        source_frames=np.concatenate(source_frames).astype(np.int32, copy=False),
        sequence_ids=tuple(source.sequence_id for source in sources),
        archive_members=tuple(source.archive_member for source in sources),
        archive_paths=tuple(str(source.archive_path) for source in sources),
        archive_sha256=tuple(source.archive_sha256 for source in sources),
        source_fps=np.asarray([source.fps for source in sources], np.float64),
        fps_overridden=tuple(source.fps_overridden for source in sources),
        conversion_reports=tuple(reports),
        skeleton_signature=G1_SKELETON.signature(),
        excluded_sequence_ids=tuple(sorted(excluded_sequence_ids)),
    )
    corpus.validate()
    return corpus


def _manifest(
    corpus: ReviewCorpus,
    motions_sha256: str,
    proposals_sha256: str,
    proposal_count: int,
    proposal_strategy: str,
) -> dict:
    return {
        "schema": "g1-reach-review",
        "version": 1,
        "target_fps": corpus.fps,
        "skeleton_names": list(G1_SKELETON.names),
        "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
        "skeleton_signature": corpus.skeleton_signature,
        "source_count": len(corpus.sequence_ids),
        "frame_count": len(corpus.positions),
        "sequence_ids": list(corpus.sequence_ids),
        "archive_members": list(corpus.archive_members),
        "archive_paths": list(corpus.archive_paths),
        "archive_sha256": list(corpus.archive_sha256),
        "source_fps": corpus.source_fps.tolist(),
        "fps_overridden": list(corpus.fps_overridden),
        "conversion_reports": list(corpus.conversion_reports),
        "excluded_sequence_ids": list(corpus.excluded_sequence_ids),
        "review_motions_sha256": motions_sha256,
        "proposals_sha256": proposals_sha256,
        "proposal_count": proposal_count,
        "proposal_strategy": proposal_strategy,
    }


def _remove_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"expected private directory, got {path}")
        shutil.rmtree(path)


def write_review_corpus(
    output: Path,
    corpus: ReviewCorpus,
    *,
    pause_bounded: bool = False,
) -> None:
    corpus.validate()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    previous = output.parent / f".{output.name}.previous-{os.getpid()}"
    _remove_directory(temporary)
    _remove_directory(previous)
    temporary.mkdir()
    try:
        np.savez_compressed(
            temporary / _MOTIONS_NAME,
            positions=corpus.positions.astype(np.float32, copy=False),
            rotations=corpus.rotations.astype(np.float32, copy=False),
            range_starts=corpus.range_starts.astype(np.int32, copy=False),
            range_stops=corpus.range_stops.astype(np.int32, copy=False),
            source_frames=corpus.source_frames.astype(np.int32, copy=False),
        )
        from .segmentation import (
            propose_pause_bounded_reaches,
            propose_reaches,
        )

        proposals = (
            propose_pause_bounded_reaches(corpus)
            if pause_bounded
            else propose_reaches(corpus)
        )
        proposal_document = {
            "schema": "g1-reach-proposals",
            "version": 1,
            "review_motions_sha256": _sha256(temporary / _MOTIONS_NAME),
            "proposals": [dataclasses.asdict(value) for value in proposals],
        }
        (temporary / _PROPOSALS_NAME).write_text(
            json.dumps(proposal_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = _manifest(
            corpus,
            proposal_document["review_motions_sha256"],
            _sha256(temporary / _PROPOSALS_NAME),
            len(proposals),
            "pause-bounded" if pause_bounded else "radial",
        )
        (temporary / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        read_review_corpus(temporary)
        if output.exists():
            if not output.is_dir():
                raise ValueError(f"review output is not a directory: {output}")
            os.replace(output, previous)
        try:
            os.replace(temporary, output)
        except BaseException:
            if previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        _remove_directory(previous)
    except BaseException:
        _remove_directory(temporary)
        raise


def read_review_corpus(output: Path) -> ReviewCorpus:
    output = Path(output)
    manifest = json.loads(
        (output / _MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "g1-reach-review" or manifest.get("version") != 1:
        raise ValueError("unsupported reach review manifest")
    motions = output / _MOTIONS_NAME
    if _sha256(motions) != manifest.get("review_motions_sha256"):
        raise ValueError("review motions checksum mismatch")
    with np.load(motions, allow_pickle=False) as arrays:
        expected = {
            "positions", "rotations", "range_starts", "range_stops",
            "source_frames",
        }
        if set(arrays.files) != expected:
            raise ValueError("review motion array names mismatch")
        values = {name: arrays[name].copy() for name in expected}
    corpus = ReviewCorpus(
        fps=float(manifest["target_fps"]),
        positions=values["positions"],
        rotations=values["rotations"],
        range_starts=values["range_starts"],
        range_stops=values["range_stops"],
        source_frames=values["source_frames"],
        sequence_ids=tuple(manifest["sequence_ids"]),
        archive_members=tuple(manifest["archive_members"]),
        archive_paths=tuple(manifest["archive_paths"]),
        archive_sha256=tuple(manifest["archive_sha256"]),
        source_fps=np.asarray(manifest["source_fps"], np.float64),
        fps_overridden=tuple(bool(value) for value in manifest["fps_overridden"]),
        conversion_reports=tuple(manifest["conversion_reports"]),
        skeleton_signature=str(manifest["skeleton_signature"]),
        excluded_sequence_ids=tuple(manifest["excluded_sequence_ids"]),
    )
    if corpus.skeleton_signature != G1_SKELETON.signature():
        raise ValueError("review skeleton signature mismatch")
    corpus.validate()
    if manifest.get("source_count") != len(corpus.sequence_ids):
        raise ValueError("review manifest source count mismatch")
    if manifest.get("frame_count") != len(corpus.positions):
        raise ValueError("review manifest frame count mismatch")
    proposals = read_reach_proposals(output)
    if manifest.get("proposal_count") != len(proposals):
        raise ValueError("review manifest proposal count mismatch")
    return corpus


def read_reach_proposals(output: Path):
    from .segmentation import ReachProposal

    output = Path(output)
    manifest = json.loads(
        (output / _MANIFEST_NAME).read_text(encoding="utf-8")
    )
    proposals_path = output / _PROPOSALS_NAME
    if _sha256(proposals_path) != manifest.get("proposals_sha256"):
        raise ValueError("reach proposals checksum mismatch")
    document = json.loads(proposals_path.read_text(encoding="utf-8"))
    if (
        document.get("schema") != "g1-reach-proposals"
        or document.get("version") != 1
    ):
        raise ValueError("unsupported reach proposal document")
    if document.get("review_motions_sha256") != manifest.get(
        "review_motions_sha256"
    ):
        raise ValueError("reach proposals are bound to different review motions")
    proposals = [ReachProposal(**value) for value in document.get("proposals", [])]
    if len({proposal.proposal_id for proposal in proposals}) != len(proposals):
        raise ValueError("duplicate reach proposal id")
    if any(proposal.status != "pending" for proposal in proposals):
        raise ValueError("generated reach proposals must remain pending")
    return proposals
