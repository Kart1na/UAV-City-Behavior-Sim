"""Generate a batch of simulation episodes."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from random import seed

from sim.app import run_simulation


def generate_dataset(num_episodes: int, output_root: str | Path, random_seed: int = 7) -> list[Path]:
    seed(random_seed)
    output_root = Path(output_root)
    paths = []
    for idx in range(num_episodes):
        paths.append(
            run_simulation(
                frames=120,
                output_dir=output_root / f"episode_{idx:04d}",
                scene_config="configs/scene_city.yaml",
                humans_config="configs/humans.yaml",
                events_config="configs/events.yaml",
                uav_config="configs/uav.yaml",
            )
        )
    return paths


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-n", "--num-episodes", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/episodes")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for path in generate_dataset(args.num_episodes, args.output_root, args.seed):
        print(path)


if __name__ == "__main__":
    main()
