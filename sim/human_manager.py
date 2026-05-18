"""Manage the crowd and simple social-behavior contagion."""

from __future__ import annotations

from math import dist
from typing import Any

from .human_agent import HumanAgent, Position


class HumanManager:
    def __init__(self, humans: list[HumanAgent] | None = None, contagion_radius: float = 6.0):
        self.humans = humans or []
        self.contagion_radius = contagion_radius

    def add(self, human: HumanAgent) -> None:
        self.humans.append(human)

    def tick_all(self, world: Any, dt: float = 1.0) -> None:
        self._apply_panic_contagion()
        for human in self.humans:
            human.tick(world, dt=dt)

    def _apply_panic_contagion(self) -> None:
        running = [human for human in self.humans if human.state == "running" or human.panic > 0.6]
        for human in self.humans:
            if human in running:
                continue
            if any(dist(human.position, other.position) <= self.contagion_radius for other in running):
                human.panic = min(1.0, human.panic + 0.1)

    def get_all_positions(self) -> list[Position]:
        return [human.get_position() for human in self.humans]

    def get_all_states(self) -> list[str]:
        return [human.get_state() for human in self.humans]

