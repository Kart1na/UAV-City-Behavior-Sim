"""Run one complete episode."""

from __future__ import annotations

from argparse import ArgumentParser

from sim.app import run_simulation


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs/episodes/default")
    parser.add_argument("--scene-config", default="configs/scene_city.yaml")
    parser.add_argument("--humans-config", default="configs/humans.yaml")
    parser.add_argument("--events-config", default="configs/events.yaml")
    parser.add_argument("--uav-config", default="configs/uav.yaml")
    args = parser.parse_args()
    print(
        run_simulation(
            frames=args.frames,
            output_dir=args.output_dir,
            scene_config=args.scene_config,
            humans_config=args.humans_config,
            events_config=args.events_config,
            uav_config=args.uav_config,
        )
    )


if __name__ == "__main__":
    main()
