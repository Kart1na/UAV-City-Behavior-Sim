"""Single human agent with behavior-tree driven movement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any


Position = tuple[float, float]


@dataclass
class HumanAgent:
    id: str
    position: Position
    goal: Position
    personality: str = "normal"
    scene_manager: Any | None = None
    behavior_tree: Any | None = None
    speed: float = 1.2
    panic: float = 0.0
    state: str = "walking"
    velocity: Position = (0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tick(self, world: Any, dt: float = 1.0) -> None:
        if self.behavior_tree:
            self.behavior_tree.tick(self, world)
        self._move_towards(self.goal, dt)

    def _move_towards(self, target: Position, dt: float) -> None:
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        length = hypot(dx, dy)
        if length < 1e-6:
            self.velocity = (0.0, 0.0)
            self.state = "arrived"
            return

        move_speed = self.speed * (1.0 + min(self.panic, 1.0))
        step = min(move_speed * dt, length)
        candidate = (self.position[0] + dx / length * step, self.position[1] + dy / length * step)
        if self.scene_manager and not self.scene_manager.is_position_free(candidate):
            self.velocity = (0.0, 0.0)
            self.state = "blocked"
            return
        self.velocity = ((candidate[0] - self.position[0]) / dt, (candidate[1] - self.position[1]) / dt)
        self.position = candidate
        self.state = "running" if self.panic > 0.55 else "walking"

    def run_away_from(self, threat_position: Position) -> None:
        dx = self.position[0] - threat_position[0]
        dy = self.position[1] - threat_position[1]
        length = hypot(dx, dy) or 1.0
        self.goal = (self.position[0] + dx / length * 25.0, self.position[1] + dy / length * 25.0)
        self.panic = min(1.0, self.panic + 0.25)
        self.state = "running"

    def get_position(self) -> Position:
        return self.position

    def get_state(self) -> str:
        return self.state

    def set_goal(self, new_goal: Position) -> None:
        self.goal = new_goal

