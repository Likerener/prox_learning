"""Eval task definitions.

Four tasks, all built on MolmoSpaces FrankaPickandPlace (procthor-objaverse,
randomized cameras):

  pnp           — open workspace, no nearby obstacles (small expected delta)
  near_contact  — fixed obstacle 5–8 cm from expert path (PRIMARY)
  pnp_color     — language-specified object among distractors
  pnp_next_to   — spatial relation: place next to a reference

The primary scientific claim (PROJECT.md §8) is tested on ``near_contact``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    obstacle: bool
    language_relation: bool

    # Filled in by the registry below; pointers to scene/config builders.
    scene_builder: Callable | None = None


REGISTRY: dict[str, TaskSpec] = {
    "pnp": TaskSpec(
        name="pnp",
        description="Open workspace pick-and-place. Baseline competence test.",
        obstacle=False,
        language_relation=False,
    ),
    "near_contact": TaskSpec(
        name="near_contact",
        description="Pick-and-place with a fixed obstacle 5–8 cm from the "
        "expert arm path. Expected to show the largest proximity advantage.",
        obstacle=True,
        language_relation=False,
    ),
    "pnp_color": TaskSpec(
        name="pnp_color",
        description="Language-specified object among colored distractors.",
        obstacle=False,
        language_relation=True,
    ),
    "pnp_next_to": TaskSpec(
        name="pnp_next_to",
        description="Place object next to a reference object. Most challenging "
        "language task.",
        obstacle=False,
        language_relation=True,
    ),
}


def get(name: str) -> TaskSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {list(REGISTRY)}")
    return REGISTRY[name]
