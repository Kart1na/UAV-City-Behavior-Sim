"""Feature extraction from human trajectories."""

from __future__ import annotations

from math import atan2, hypot
from statistics import mean, pvariance


def compute_crowd_features(
    positions: list[tuple[float, float]],
    velocities: list[tuple[float, float]],
    panic: list[float] | None = None,
) -> dict[str, float]:
    speeds = [hypot(vx, vy) for vx, vy in velocities]
    angles = [atan2(vy, vx) for vx, vy in velocities if abs(vx) + abs(vy) > 1e-9]
    panic = panic or [0.0 for _ in positions]
    return {
        "average_speed": mean(speeds) if speeds else 0.0,
        "velocity_variance": pvariance(speeds) if len(speeds) > 1 else 0.0,
        "flow_entropy": _angle_entropy(angles),
        "exit_density": _edge_density(positions),
        "average_panic": mean(panic) if panic else 0.0,
    }


def _angle_entropy(angles: list[float], bins: int = 8) -> float:
    if not angles:
        return 0.0
    counts = [0 for _ in range(bins)]
    for angle in angles:
        idx = int(((angle + 3.141592653589793) / (2 * 3.141592653589793)) * bins) % bins
        counts[idx] += 1
    total = sum(counts)
    return -sum((count / total) * __import__("math").log(count / total + 1e-12) for count in counts if count)


def _edge_density(positions: list[tuple[float, float]], map_size: float = 100.0, edge_band: float = 10.0) -> float:
    if not positions:
        return 0.0
    edge_count = sum(1 for x, y in positions if x <= edge_band or y <= edge_band or x >= map_size - edge_band or y >= map_size - edge_band)
    return edge_count / len(positions)

