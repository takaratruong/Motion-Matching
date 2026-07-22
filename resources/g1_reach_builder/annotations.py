from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_terrain_builder.kinematics import forward_local_hierarchy

from .review import read_reach_proposals, read_review_corpus
from .schema import ReviewCorpus
from .segmentation import ReachProposal, outbound_approach_delta


@dataclass(frozen=True)
class ReachAnnotation:
    proposal_id: str
    sequence_id: str
    active_hand: str
    departure_frame: int
    grab_frame: int
    source_departure_frame: int
    source_grab_frame: int
    status: str
    note: str
    endpoint_position_root: tuple[float, float, float]
    endpoint_rotation_root_wxyz: tuple[float, float, float, float]
    approach_direction_root: tuple[float, float, float]

    @property
    def frame_count(self) -> int:
        return self.grab_frame - self.departure_frame + 1


@dataclass(frozen=True)
class AnnotationDocument:
    version: int
    review_motions_sha256: str
    proposals_sha256: str
    annotations: tuple[ReachAnnotation, ...]


def _manifest(review: Path) -> dict:
    return json.loads(
        (Path(review) / "manifest.json").read_text(encoding="utf-8")
    )


def _source_index(corpus: ReviewCorpus, sequence_id: str) -> int:
    try:
        return corpus.sequence_ids.index(sequence_id)
    except ValueError as error:
        raise ValueError(f"unknown annotation sequence {sequence_id}") from error


def _world_source(
    corpus: ReviewCorpus,
    source_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    start = int(corpus.range_starts[source_index])
    stop = int(corpus.range_stops[source_index])
    positions, rotations = forward_local_hierarchy(
        corpus.positions[start:stop].astype(np.float64),
        corpus.rotations[start:stop].astype(np.float64),
        G1_SKELETON.parents,
    )
    return positions, rotations, start


def _annotation(
    corpus: ReviewCorpus,
    proposal_id: str,
    sequence_id: str,
    departure_frame: int,
    grab_frame: int,
    status: str,
    note: str,
    world: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> ReachAnnotation:
    if status not in {"pending", "accepted", "rejected"}:
        raise ValueError(f"invalid annotation status {status}")
    source_index = _source_index(corpus, sequence_id)
    if world is None:
        world = _world_source(corpus, source_index)
    positions, rotations, global_start = world
    if not (0 <= departure_frame <= grab_frame < len(positions)):
        raise ValueError(
            f"{sequence_id}: annotation frames outside source range"
        )
    simulation = 0
    wrist = G1_SKELETON.names.index("LeftWrist")
    root_position = positions[departure_frame, simulation]
    root_rotation_inv = holden_quat.inv(
        rotations[departure_frame, simulation]
    )
    endpoint_position = holden_quat.mul_vec(
        root_rotation_inv,
        positions[grab_frame, wrist] - root_position,
    )
    endpoint_rotation = holden_quat.normalize(holden_quat.mul(
        root_rotation_inv,
        rotations[grab_frame, wrist],
    ))
    approach_delta_world = outbound_approach_delta(
        positions[:, wrist], departure_frame, grab_frame
    )
    approach_delta = holden_quat.mul_vec(
        root_rotation_inv,
        approach_delta_world,
    )
    approach_length = float(np.linalg.norm(approach_delta))
    if status == "accepted" and approach_length < 0.01:
        raise ValueError(
            f"{sequence_id}: approach displacement {approach_length} m is below 0.01"
        )
    approach = (
        approach_delta / approach_length
        if approach_length >= 0.01
        else np.zeros(3, np.float64)
    )
    global_departure = global_start + departure_frame
    global_grab = global_start + grab_frame
    return ReachAnnotation(
        proposal_id=proposal_id,
        sequence_id=sequence_id,
        active_hand="left",
        departure_frame=int(departure_frame),
        grab_frame=int(grab_frame),
        source_departure_frame=int(corpus.source_frames[global_departure]),
        source_grab_frame=int(corpus.source_frames[global_grab]),
        status=status,
        note=str(note),
        endpoint_position_root=tuple(float(value) for value in endpoint_position),
        endpoint_rotation_root_wxyz=tuple(
            float(value) for value in endpoint_rotation
        ),
        approach_direction_root=tuple(float(value) for value in approach),
    )


def create_annotation_document(review: Path) -> AnnotationDocument:
    review = Path(review)
    corpus = read_review_corpus(review)
    proposals = read_reach_proposals(review)
    manifest = _manifest(review)
    world_by_source: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    annotations: list[ReachAnnotation] = []
    for proposal in proposals:
        source_index = _source_index(corpus, proposal.sequence_id)
        world = world_by_source.setdefault(
            proposal.sequence_id,
            _world_source(corpus, source_index),
        )
        annotations.append(_annotation(
            corpus,
            proposal.proposal_id,
            proposal.sequence_id,
            proposal.departure_frame,
            proposal.grab_frame,
            "pending",
            "",
            world,
        ))
    document = AnnotationDocument(
        version=1,
        review_motions_sha256=manifest["review_motions_sha256"],
        proposals_sha256=manifest["proposals_sha256"],
        annotations=tuple(annotations),
    )
    _validate_document(document)
    return document


def update_annotation(
    review: Path | ReviewCorpus,
    current: ReachAnnotation,
    *,
    departure_frame: int,
    grab_frame: int,
    status: str,
    note: str,
) -> ReachAnnotation:
    corpus = review if isinstance(review, ReviewCorpus) else read_review_corpus(review)
    return _annotation(
        corpus,
        current.proposal_id,
        current.sequence_id,
        departure_frame,
        grab_frame,
        status,
        note,
    )


def _validate_document(document: AnnotationDocument) -> None:
    if document.version != 1:
        raise ValueError(f"unsupported annotation version {document.version}")
    if len(document.review_motions_sha256) != 64:
        raise ValueError("invalid annotation review checksum")
    if len(document.proposals_sha256) != 64:
        raise ValueError("invalid annotation proposal checksum")
    ids = [value.proposal_id for value in document.annotations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate annotation proposal id")
    for value in document.annotations:
        if value.active_hand != "left":
            raise ValueError("this corpus only accepts left-hand annotations")
        if value.status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid annotation status {value.status}")
        if value.departure_frame < 0 or value.grab_frame < value.departure_frame:
            raise ValueError("invalid annotation frame interval")
        endpoint = np.asarray(value.endpoint_position_root)
        rotation = np.asarray(value.endpoint_rotation_root_wxyz)
        approach = np.asarray(value.approach_direction_root)
        if not all(np.isfinite(array).all() for array in (endpoint, rotation, approach)):
            raise ValueError("annotation contains non-finite endpoint values")
        if abs(float(np.linalg.norm(rotation)) - 1.0) > 1e-4:
            raise ValueError("annotation endpoint quaternion is not unit length")
        if value.status == "accepted" and abs(
            float(np.linalg.norm(approach)) - 1.0
        ) > 1e-4:
            raise ValueError("accepted annotation approach is not unit length")


def write_annotations_atomic(
    path: Path,
    document: AnnotationDocument,
) -> None:
    _validate_document(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "g1-reach-annotations",
        "version": document.version,
        "review_motions_sha256": document.review_motions_sha256,
        "proposals_sha256": document.proposals_sha256,
        "annotations": [asdict(value) for value in document.annotations],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_annotations(path: Path, review: Path) -> AnnotationDocument:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "g1-reach-annotations":
        raise ValueError("unsupported annotation document")
    document = AnnotationDocument(
        version=int(value["version"]),
        review_motions_sha256=str(value["review_motions_sha256"]),
        proposals_sha256=str(value["proposals_sha256"]),
        annotations=tuple(
            ReachAnnotation(
                **{
                    **annotation,
                    "endpoint_position_root": tuple(
                        annotation["endpoint_position_root"]
                    ),
                    "endpoint_rotation_root_wxyz": tuple(
                        annotation["endpoint_rotation_root_wxyz"]
                    ),
                    "approach_direction_root": tuple(
                        annotation["approach_direction_root"]
                    ),
                }
            )
            for annotation in value.get("annotations", [])
        ),
    )
    _validate_document(document)
    manifest = _manifest(Path(review))
    if document.review_motions_sha256 != manifest.get("review_motions_sha256"):
        raise ValueError("annotations are bound to different review motions")
    if document.proposals_sha256 != manifest.get("proposals_sha256"):
        raise ValueError("annotations are bound to different proposals")
    proposals = read_reach_proposals(review)
    if [value.proposal_id for value in document.annotations] != [
        value.proposal_id for value in proposals
    ]:
        raise ValueError("annotation proposal ids do not match review proposals")
    return document
