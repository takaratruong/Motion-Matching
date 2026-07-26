from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .annotations import AnnotationDocument
from .artifacts import VERSION, ReachArtifact, ReachFeatures, assemble_reach_pack
from .mirror import build_bilateral_reaches
from .motions import CanonicalReach, build_captured_reach
from .schema import ReviewCorpus


@dataclass(frozen=True)
class ReachCorpusInput:
    label: str
    corpus: ReviewCorpus
    annotations: AnnotationDocument


def prepare_combined_reach_pack(
    inputs: Sequence[ReachCorpusInput],
) -> tuple[list[CanonicalReach], ReachArtifact, ReachFeatures, dict]:
    if not inputs:
        raise ValueError("combined reach pack requires at least one corpus")
    captured: list[CanonicalReach] = []
    corpus_records: list[dict] = []
    archive_paths: list[str] = []
    archive_sha256: list[str] = []
    excluded_sequence_ids: list[str] = []
    total_pending = 0
    total_rejected = 0
    for item in inputs:
        current = [
            build_captured_reach(item.corpus, annotation)
            for annotation in item.annotations.annotations
            if annotation.status == "accepted"
        ]
        captured.extend(current)
        pending = sum(
            value.status == "pending"
            for value in item.annotations.annotations
        )
        rejected = sum(
            value.status == "rejected"
            for value in item.annotations.annotations
        )
        total_pending += pending
        total_rejected += rejected
        archive_paths.extend(item.corpus.archive_paths)
        archive_sha256.extend(item.corpus.archive_sha256)
        excluded_sequence_ids.extend(item.corpus.excluded_sequence_ids)
        corpus_records.append({
            "label": item.label,
            "captured_reaches": len(current),
            "pending_annotations": pending,
            "rejected_annotations": rejected,
            "sequence_ids": list(item.corpus.sequence_ids),
        })
    if not captured:
        raise ValueError("no accepted reach annotations")
    ids = [reach.reach_id for reach in captured]
    proposals = [reach.proposal_id for reach in captured]
    if len(ids) != len(set(ids)) or len(proposals) != len(set(proposals)):
        raise ValueError("duplicate reach or proposal id across corpora")
    reaches = build_bilateral_reaches(captured)
    artifact, features = assemble_reach_pack(reaches)
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
        "pending_annotations": total_pending,
        "rejected_annotations": total_rejected,
        "excluded_sequence_ids": excluded_sequence_ids,
        "source_archive_paths": archive_paths,
        "source_archive_sha256": archive_sha256,
        "corpora": corpus_records,
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


def prepare_reach_pack(
    corpus: ReviewCorpus,
    annotations: AnnotationDocument,
) -> tuple[list[CanonicalReach], ReachArtifact, ReachFeatures, dict]:
    return prepare_combined_reach_pack(
        [ReachCorpusInput(label="reach", corpus=corpus, annotations=annotations)]
    )
