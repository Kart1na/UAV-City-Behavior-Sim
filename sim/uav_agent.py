"""UAV observation and next-best-view movement."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from typing import Any


Position3 = tuple[float, float, float]


@dataclass
class UAVAgent:
    position: Position3 = (20.0, 20.0, 20.0)
    yaw: float = 0.0
    speed: float = 8.0
    target: Position3 | None = None
    camera_params: dict[str, Any] | None = None

    def tick(self, dt: float = 1.0) -> None:
        if self.target is not None:
            self.move_to(self.target, dt=dt)

    def move_to(self, position: Position3, dt: float = 1.0) -> None:
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        dz = position[2] - self.position[2]
        length = hypot(hypot(dx, dy), dz)
        if length < 1e-6:
            self.target = None
            return
        step = min(self.speed * dt, length)
        self.position = (
            self.position[0] + dx / length * step,
            self.position[1] + dy / length * step,
            self.position[2] + dz / length * step,
        )

    def hover(self) -> None:
        self.target = None

    def look_at(self, target_position: tuple[float, float]) -> None:
        self.yaw = atan2(target_position[1] - self.position[1], target_position[0] - self.position[0])

    def capture_frame(self) -> dict[str, Any]:
        """Return placeholder RGB/depth/semantic payloads for local runs."""
        return {"rgb": None, "depth": None, "semantic": None, "pose": self.pose_dict()}

    def pose_dict(self) -> dict[str, Any]:
        return {"position": self.position, "yaw": self.yaw}

