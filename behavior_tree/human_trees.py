"""Human behavior-tree presets by personality."""

from __future__ import annotations

from .actions import AvoidObstacles, FollowCrowd, IncreasePanic, RunAwayFromEvent, WalkToGoal
from .conditions import FireVisible, NearbyPeopleRunning
from .core import Selector, Sequence


def build_human_tree(personality: str = "normal"):
    if personality == "cautious":
        return Selector(Sequence(FireVisible(), IncreasePanic(0.35), RunAwayFromEvent()), Sequence(AvoidObstacles(), WalkToGoal()))
    if personality == "social":
        return Selector(Sequence(NearbyPeopleRunning(), IncreasePanic(0.18), FollowCrowd()), Sequence(AvoidObstacles(), WalkToGoal()))
    if personality == "curious":
        return Selector(Sequence(FireVisible(), IncreasePanic(0.05), WalkToGoal()), Sequence(AvoidObstacles(), WalkToGoal()))
    return Selector(
        Sequence(FireVisible(), IncreasePanic(0.25), RunAwayFromEvent()),
        Sequence(NearbyPeopleRunning(), IncreasePanic(0.1), FollowCrowd()),
        Sequence(AvoidObstacles(), WalkToGoal()),
    )

