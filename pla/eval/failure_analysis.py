"""Failure case analysis.

Categorize per-episode failures and dump qualitative videos with proximity
sensor readings overlaid (PROJECT.md §4.4).

Categories:
  approach_collision  — arm hits obstacle (proximity should help)
  grasp_miss          — gripper misses target (dense EE sensors may help)
  place_failure       — object dropped (proximity unlikely to help)
  language_failure    — wrong object selected (VLM error, proximity doesn't help)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureType(str, Enum):
    APPROACH_COLLISION = "approach_collision"
    GRASP_MISS = "grasp_miss"
    PLACE_FAILURE = "place_failure"
    LANGUAGE_FAILURE = "language_failure"
    SUCCESS = "success"


@dataclass
class EpisodeOutcome:
    seed: int
    task: str
    success: bool
    failure_type: FailureType
    proximity_min_mm: float
    notes: str = ""


def categorize(*, episode_log) -> FailureType:
    """Map raw env-side log to a FailureType.

    Stub — implement against MolmoSpaces FrankaPickandPlace once the env
    contract for collision / grasp / place events is wired in.
    """
    raise NotImplementedError("hook to MolmoSpaces episode log")
