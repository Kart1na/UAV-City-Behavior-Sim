"""Select a UAV next-best-view location from the hazard heatmap."""

from __future__ import annotations

from math import dist

from .hazard_heatmap import Grid, peak_position


def choose_next_best_view(
    heatmap: Grid,
    uav_pose: tuple[float, float, float],
    scene,
    altitude: float = 20.0,
) -> tuple[float, float]:
    peak = peak_position(heatmap)
    candidates = _candidate_ring(peak, radius=18.0)
    valid = [candidate for candidate in candidates if scene.is_position_free(candidate)]
    if not valid:
        return peak
    return max(valid, key=lambda candidate: _information_gain(candidate, peak, uav_pose, altitude))


def _candidate_ring(center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    cx, cy = center
    offsets = [(radius, 0), (-radius, 0), (0, radius), (0, -radius), (radius * 0.7, radius * 0.7), (-radius * 0.7, radius * 0.7)]
    return [(cx + dx, cy + dy) for dx, dy in offsets]


def _information_gain(candidate: tuple[float, float], peak: tuple[float, float], uav_pose: tuple[float, float, float], altitude: float) -> float:
    travel_cost = dist((uav_pose[0], uav_pose[1]), candidate)
    view_distance = dist(candidate, peak)
    return 100.0 / (1.0 + view_distance) - 0.05 * travel_cost + 0.01 * altitude

