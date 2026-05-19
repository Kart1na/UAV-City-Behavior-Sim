"""Run a city anomaly episode inside Isaac Sim."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from inference.hazard_heatmap import predict_hazard_heatmap
from inference.next_best_view import choose_next_best_view
from sim.app import build_default_world
from sim.isaac_runtime import IsaacCityRuntime, IsaacRuntimeConfig, IsaacUnavailableError


def run_isaac_simulation(
    frames: int = 240,
    output_dir: str | Path = "outputs/isaac/default",
    scene_config: str | Path = "configs/scene_city.yaml",
    humans_config: str | Path = "configs/humans.yaml",
    events_config: str | Path = "configs/events.yaml",
    uav_config: str | Path = "configs/uav.yaml",
    headless: bool = False,
    save_usd: bool = True,
) -> Path:
    """Run the project simulation loop while rendering it in Isaac Sim."""
    output_dir = Path(output_dir)
    scene, humans, events, uav, logger = build_default_world(scene_config, humans_config, events_config, uav_config)
    logger.output_dir = output_dir

    runtime = IsaacCityRuntime(IsaacRuntimeConfig(headless=headless)).launch()
    try:
        runtime.build_scene(scene, events, humans, uav)
        for frame in range(frames):
            world = {"scene": scene, "events": events, "humans": humans, "uav": uav}
            humans.tick_all(world)
            events.tick(world)

            heatmap = predict_hazard_heatmap(humans.get_all_positions(), [human.velocity for human in humans.humans])
            next_view = choose_next_best_view(heatmap, uav.position, scene)
            uav.target = (next_view[0], next_view[1], uav.position[2])
            uav.look_at(next_view)
            uav.tick()

            runtime.sync(humans, events, uav)
            runtime.step(render=True)
            logger.log_frame(frame, humans, uav, events)

        log_path = logger.save()
        if save_usd:
            runtime.save_stage(output_dir / "city_anomaly_episode.usd")
        return log_path
    finally:
        runtime.close()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--output-dir", default="outputs/isaac/default")
    parser.add_argument("--scene-config", default="configs/scene_city.yaml")
    parser.add_argument("--humans-config", default="configs/humans.yaml")
    parser.add_argument("--events-config", default="configs/events.yaml")
    parser.add_argument("--uav-config", default="configs/uav.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-save-usd", action="store_true")
    args = parser.parse_args()

    try:
        print(
            run_isaac_simulation(
                frames=args.frames,
                output_dir=args.output_dir,
                scene_config=args.scene_config,
                humans_config=args.humans_config,
                events_config=args.events_config,
                uav_config=args.uav_config,
                headless=args.headless,
                save_usd=not args.no_save_usd,
            )
        )
    except IsaacUnavailableError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
