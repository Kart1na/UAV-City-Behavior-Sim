"""Frame logger for simulation episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrameRecord:
    frame: int
    humans: list[dict[str, Any]]
    uav_pose: dict[str, Any]
    events: list[dict[str, Any]]


class DataLogger:
    def __init__(self, output_dir: str | Path = "outputs/episodes/default"):
        self.output_dir = Path(output_dir)
        self.frames: list[FrameRecord] = []

    def log_frame(self, frame: int, human_manager: Any, uav_agent: Any, event_manager: Any) -> None:
        humans = [
            {
                "id": human.id,
                "position": human.position,
                "velocity": human.velocity,
                "state": human.state,
                "panic": human.panic,
            }
            for human in human_manager.humans
        ]
        events = [asdict(event) for event in event_manager.events]
        self.frames.append(FrameRecord(frame=frame, humans=humans, uav_pose=uav_agent.pose_dict(), events=events))

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "frames.json"
        path.write_text(json.dumps([asdict(frame) for frame in self.frames], indent=2), encoding="utf-8")
        return path

