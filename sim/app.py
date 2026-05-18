"""Application entry point for a complete UAV crowd simulation."""

from __future__ import annotations

from pathlib import Path

from behavior_tree.human_trees import build_human_tree
from inference.hazard_heatmap import predict_hazard_heatmap
from inference.next_best_view import choose_next_best_view
from sim.data_logger import DataLogger
from sim.config_loader import load_yaml
from sim.event_manager import EventManager
from sim.human_agent import HumanAgent
from sim.human_manager import HumanManager
from sim.scene_manager import SceneManager
from sim.uav_agent import UAVAgent


def build_default_world(
    scene_config: str | Path | None = None,
    humans_config: str | Path | None = None,
    events_config: str | Path | None = None,
    uav_config: str | Path | None = None,
) -> tuple[SceneManager, HumanManager, EventManager, UAVAgent, DataLogger]:
    scene_data = load_yaml(scene_config) or {
        "bounds": {"width": 100, "height": 100},
        "buildings": [{"x": 38, "y": 35, "width": 12, "height": 22}],
        "exits": [[0, 50], [100, 50]],
    }
    events_data = load_yaml(events_config) or {"events": [{"id": "fire_0", "kind": "fire", "position": [72, 45], "radius": 14}]}
    humans_data = load_yaml(humans_config)
    uav_data = load_yaml(uav_config)

    scene = SceneManager(config=scene_data)
    events = EventManager(config=events_data)
    human_rows = humans_data.get("humans") or [
        {"id": f"h{i}", "position": [20.0 + i * 2, 45.0 + i % 3], "goal": [95.0, 50.0], "personality": "normal"}
        for i in range(12)
    ]
    humans = HumanManager(
        [
            HumanAgent(
                id=row.get("id", f"h{i}"),
                position=tuple(map(float, row.get("position", [20.0, 50.0]))),
                goal=tuple(map(float, row.get("goal", [95.0, 50.0]))),
                personality=row.get("personality", "normal"),
                scene_manager=scene,
            )
            for i, row in enumerate(human_rows)
        ]
    )
    for human in humans.humans:
        human.behavior_tree = build_human_tree(human.personality)
    uav = UAVAgent(
        position=tuple(map(float, uav_data.get("position", [20.0, 20.0, 20.0]))),
        yaw=float(uav_data.get("yaw", 0.0)),
        speed=float(uav_data.get("speed", 8.0)),
        camera_params=uav_data.get("camera"),
    )
    logger = DataLogger()
    return scene, humans, events, uav, logger


def run_simulation(
    frames: int = 120,
    output_dir: str | Path = "outputs/episodes/default",
    scene_config: str | Path | None = None,
    humans_config: str | Path | None = None,
    events_config: str | Path | None = None,
    uav_config: str | Path | None = None,
) -> Path:
    scene, humans, events, uav, logger = build_default_world(scene_config, humans_config, events_config, uav_config)
    logger.output_dir = Path(output_dir)
    for frame in range(frames):
        world = {"scene": scene, "events": events, "humans": humans, "uav": uav}
        humans.tick_all(world)
        events.tick(world)
        heatmap = predict_hazard_heatmap(humans.get_all_positions(), [human.velocity for human in humans.humans])
        next_view = choose_next_best_view(heatmap, uav.position, scene)
        uav.target = (next_view[0], next_view[1], uav.position[2])
        uav.look_at(next_view)
        uav.tick()
        logger.log_frame(frame, humans, uav, events)
    return logger.save()


if __name__ == "__main__":
    print(run_simulation())
