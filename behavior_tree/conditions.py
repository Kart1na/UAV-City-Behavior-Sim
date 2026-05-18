"""Behavior-tree condition nodes."""

from __future__ import annotations

from math import dist

from .core import Node, Status


class FireVisible(Node):
    def tick(self, agent, world) -> Status:
        events = world.get("events")
        if events and events.is_event_visible_to_agent(agent):
            return Status.SUCCESS
        return Status.FAILURE


class NearbyPeopleRunning(Node):
    def __init__(self, radius: float = 8.0):
        self.radius = radius

    def tick(self, agent, world) -> Status:
        humans = world.get("humans")
        if not humans:
            return Status.FAILURE
        for other in humans.humans:
            if other is agent:
                continue
            if other.state == "running" and dist(agent.position, other.position) <= self.radius:
                return Status.SUCCESS
        return Status.FAILURE


class CrowdDensityHigh(Node):
    def __init__(self, radius: float = 6.0, threshold: int = 5):
        self.radius = radius
        self.threshold = threshold

    def tick(self, agent, world) -> Status:
        humans = world.get("humans")
        if not humans:
            return Status.FAILURE
        count = sum(1 for other in humans.humans if dist(agent.position, other.position) <= self.radius)
        return Status.SUCCESS if count >= self.threshold else Status.FAILURE

