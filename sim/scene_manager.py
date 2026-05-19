"""City scene management for UAV crowd-behavior simulation.

The class in this module intentionally works without Isaac Sim installed. When
an Isaac Sim stage API is provided by the caller, object creation can be routed
through that integration later; otherwise it runs as a lightweight 2D map.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any, Iterable

from .config_loader import load_yaml


Position = tuple[float, float]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangular object in the 2D city map."""

    x: float
    y: float
    width: float
    height: float
    kind: str = "obstacle"
    id: str | None = None

    def contains(self, position: Position) -> bool:
        px, py = position
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def center(self) -> Position:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


class SceneManager:
    """Load city geometry and answer spatial queries used by agents."""

    def __init__(self, config_path: str | Path | None = None, config: dict[str, Any] | None = None):
        self.config_path = Path(config_path) if config_path else None
        self.width = 100.0
        self.height = 100.0
        self.buildings: list[Rect] = []
        self.roads: list[Rect] = []
        self.obstacles: list[Rect] = []
        self.exits: list[Position] = [(0.0, 50.0), (100.0, 50.0)]
        self.stage = None
        self.raw_config: dict[str, Any] = {}

        data = config or self._load_config(self.config_path)
        self.load_city(data)

    def _load_config(self, path: Path | None) -> dict[str, Any]:
        if not path or not path.exists():
            return {}
        return load_yaml(path)

    def load_city(self, config: dict[str, Any]) -> None:
        """Load map bounds, buildings, roads, obstacles, and exits."""
        self.raw_config = config
        bounds = config.get("bounds", {})
        self.width = float(bounds.get("width", self.width))
        self.height = float(bounds.get("height", self.height))
        self.buildings = self._load_rects(config.get("buildings", []), "building")
        self.roads = self._load_rects(config.get("roads", []), "road")
        self.obstacles = self._load_rects(config.get("obstacles", []), "obstacle") + self.buildings
        self.exits = [tuple(map(float, item)) for item in config.get("exits", self.exits)]

    def _load_rects(self, rows: Iterable[dict[str, Any]], kind: str) -> list[Rect]:
        rects: list[Rect] = []
        for idx, row in enumerate(rows):
            rects.append(
                Rect(
                    x=float(row.get("x", 0.0)),
                    y=float(row.get("y", 0.0)),
                    width=float(row.get("width", 1.0)),
                    height=float(row.get("height", 1.0)),
                    kind=row.get("kind", kind),
                    id=row.get("id", f"{kind}_{idx}"),
                )
            )
        return rects

    def create_prims(self, stage: Any | None = None) -> None:
        """Hook point for Isaac Sim prim creation.

        In Isaac Sim, callers can pass a stage and adapt this method to call
        get_stage()/create_prim(). In local mode this method simply records the
        stage object and leaves geometry in Python data structures.
        """
        self.stage = stage

    def is_position_free(self, position: Position) -> bool:
        x, y = position
        if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
            return False
        return not any(obstacle.contains(position) for obstacle in self.obstacles)

    def get_nearest_exit(self, position: Position) -> Position:
        return min(self.exits, key=lambda exit_pos: dist(position, exit_pos))

    def get_obstacles_in_radius(self, position: Position, radius: float) -> list[Rect]:
        return [obstacle for obstacle in self.obstacles if dist(position, obstacle.center()) <= radius]
