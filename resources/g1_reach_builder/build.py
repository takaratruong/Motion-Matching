import numpy as np

from .annotations import AnnotationDocument
from .artifacts import VERSION, ReachArtifact, ReachFeatures, assemble_reach_pack
from .mirror import build_bilateral_reaches
from .motions import CanonicalReach, build_captured_reach
from .schema import ReviewCorpus


def prepare_reach_pack(
    corpus: ReviewCorpus,
    annotations: AnnotationDocument,
) -> tuple[list[CanonicalReach], ReachArtifact, ReachFeatures, dict]:
    captured = [
        build_captured_reach(corpus, annotation)
        for annotation in annotations.annotations
        if annotation.status == "accepted"
    ]
    if not captured:
        raise ValueError("no accepted reach annotations")
    reaches = build_bilateral_reaches(captured)
    artifact, features = assemble_reach_pack(reaches)
    pending = sum(
        annotation.status == "pending" for annotation in annotations.annotations
    )
    rejected = sum(
        annotation.status == "rejected" for annotation in annotations.annotations
    )
    return_frames = artifact.range_stops - artifact.contact_frames - 1
    manifest = {
        "version": VERSION,
        "captured_reaches": len(captured),
        "mirrored_reaches": len(captured),
        "paired_returns": int(np.sum(return_frames > 0)),
        "unavailable_returns": int(np.sum(return_frames == 0)),
        "return_frame_count": int(np.sum(return_frames)),
        "minimum_return_frames": int(np.min(return_frames)),
        "maximum_return_frames": int(np.max(return_frames)),
        "pending_annotations": pending,
        "rejected_annotations": rejected,
        "excluded_sequence_ids": list(corpus.excluded_sequence_ids),
        "source_archive_paths": list(corpus.archive_paths),
        "source_archive_sha256": list(corpus.archive_sha256),
        "reach_records": [
            {
                "reach_id": reach.reach_id,
                "original_reach_id": reach.original_reach_id,
                "proposal_id": reach.proposal_id,
                "sequence_id": reach.sequence_id,
                "active_hand": int(reach.active_hand),
                "augmentation": int(reach.augmentation),
                "contact_index": reach.contact_index,
                "return_available": bool(reach.return_available),
            }
            for reach in reaches
        ],
    }
    return reaches, artifact, features, manifest
