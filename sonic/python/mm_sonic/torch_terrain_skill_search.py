"""Entry-only motion matching for PHP-style terrain skills."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import torch

from .joints import ContractError
from .torch_motion_features import TorchMotionDatabase
from .torch_motion_matcher import MatcherConfig, rank_exact_transition_candidates
from .torch_terrain_skills import TerrainSkill, TerrainSkillInventory


TerrainValidator = Callable[[TerrainSkill, int], bool]


@dataclass(frozen=True)
class TerrainSkillSearchResult:
    skill: TerrainSkill
    selected_row: int
    selected_feature_cost: float
    rejected_by_reason: Mapping[str, int]


def skill_entry_eligibility(
    database: TorchMotionDatabase,
    inventory: TerrainSkillInventory,
) -> torch.Tensor:
    """Return a device-owned mask containing only annotated skill-entry rows."""

    if not isinstance(database, TorchMotionDatabase):
        raise ContractError("terrain skill search requires a TorchMotionDatabase")
    if not isinstance(inventory, TerrainSkillInventory):
        raise ContractError("terrain skill search requires a TerrainSkillInventory")
    row_count = int(database._search_features.shape[0])
    eligible = torch.zeros(row_count, dtype=torch.bool, device=database.device)
    for skill in inventory.skills:
        for row in skill.entry_rows:
            if type(row) is not int or not 0 <= row < row_count:
                raise ContractError("terrain skill entry row is outside database")
            if inventory.row_to_skill.get(row) != skill.skill_index:
                raise ContractError("terrain skill row mapping is inconsistent")
            eligible[row] = True
    return eligible


def select_terrain_skill(
    database: TorchMotionDatabase,
    inventory: TerrainSkillInventory,
    normalized_query: torch.Tensor,
    *,
    current_clip_index: int,
    current_frame_index: int,
    terrain_validator: TerrainValidator,
    config: MatcherConfig = MatcherConfig(),
) -> TerrainSkillSearchResult:
    """Hard-gate terrain compatibility, then rank entries by exact MM cost."""

    if not callable(terrain_validator):
        raise ContractError("terrain validator must be callable")
    eligible = skill_entry_eligibility(database, inventory)
    rejected = 0
    for row in torch.nonzero(eligible, as_tuple=False).flatten().cpu().tolist():
        row = int(row)
        skill_index = inventory.row_to_skill[row]
        if not 0 <= skill_index < len(inventory.skills):
            raise ContractError("terrain skill index is invalid")
        skill = inventory.skills[skill_index]
        if not bool(terrain_validator(skill, row)):
            eligible[row] = False
            rejected += 1
    if not bool(eligible.any().item()):
        raise ContractError("no terrain-compatible skill entry exists")

    ranked = rank_exact_transition_candidates(
        database,
        normalized_query,
        current_clip_index=current_clip_index,
        current_frame_index=current_frame_index,
        config=config,
        transition_eligible_rows=eligible,
    )
    if not ranked:
        raise ContractError("no terrain-compatible skill entry exists")
    decision = ranked[0]
    skill_index = inventory.row_to_skill.get(decision.selected_row)
    if skill_index is None:
        raise ContractError("selected terrain skill row has no owner")
    return TerrainSkillSearchResult(
        skill=inventory.skills[skill_index],
        selected_row=decision.selected_row,
        selected_feature_cost=decision.selected_feature_cost,
        rejected_by_reason=MappingProxyType({"terrain": rejected}),
    )
