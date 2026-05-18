"""Hidden event manager for fire, accident, and blockage events."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any

from .config_loader import load_yaml


@dataclass
class Event:
    id: str
    kind: str
    position: tuple[float, float]
    radius: float = 12.0
    active: bool = True
    visible: bool = False


class EventManager:
    def __init__(self, config_path: str | Path | None = None, config: dict[str, Any] | None = None):
        data = config or self._load_config(Path(config_path) if config_path else None)
        self.events = self._load_events(data.get("events", []))

    def _load_config(self, path: Path | None) -> dict[str, Any]:
        if not path or not path.exists():
            return {}
        return load_yaml(path)

    def _load_events(self, rows: list[dict[str, Any]]) -> list[Event]:
        if not rows:
            rows = [{"id": "fire_0", "kind": "fire", "position": [70, 45], "radius": 14}]
        return [
            Event(
                id=row.get("id", f"event_{idx}"),
                kind=row.get("kind", "fire"),
                position=tuple(map(float, row.get("position", [0, 0]))),
                radius=float(row.get("radius", 12.0)),
                active=bool(row.get("active", True)),
            )
            for idx, row in enumerate(rows)
        ]

    def tick(self, world: Any | None = None) -> None:
        for event in self.events:
            event.visible = bool(event.visible and event.active)

    def is_event_visible_to_agent(self, agent: Any, view_radius: float = 25.0) -> bool:
        return any(event.active and dist(agent.get_position(), event.position) <= view_radius for event in self.events)

    def get_event_position(self, event_id: str) -> tuple[float, float] | None:
        for event in self.events:
            if event.id == event_id:
                return event.position
        return None

    def nearest_active_event(self, position: tuple[float, float]) -> Event | None:
        active = [event for event in self.events if event.active]
        return min(active, key=lambda event: dist(position, event.position), default=None)
