"""Infer a hazard heatmap from crowd motion and panic."""

from __future__ import annotations

from math import hypot


Grid = list[list[float]]


def predict_hazard_heatmap(
    positions: list[tuple[float, float]],
    velocities: list[tuple[float, float]],
    panic: list[float] | None = None,
    grid_size: int = 64,
    map_size: float = 100.0,
) -> Grid:
    panic = panic or [0.5 for _ in positions]
    heatmap = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for (x, y), (vx, vy), p in zip(positions, velocities, panic):
        speed = hypot(vx, vy)
        if speed < 1e-6:
            continue
        # People tend to move away from danger, so vote backwards along motion.
        source_x = x - vx / speed * 15.0
        source_y = y - vy / speed * 15.0
        gx = max(0, min(grid_size - 1, int(source_x / map_size * grid_size)))
        gy = max(0, min(grid_size - 1, int(source_y / map_size * grid_size)))
        heatmap[gy][gx] += 1.0 + p
    return _smooth(heatmap)


def peak_position(heatmap: Grid, map_size: float = 100.0) -> tuple[float, float]:
    best_y, best_x, best_value = 0, 0, float("-inf")
    for y, row in enumerate(heatmap):
        for x, value in enumerate(row):
            if value > best_value:
                best_y, best_x, best_value = y, x, value
    grid_size = len(heatmap) or 1
    return ((best_x + 0.5) / grid_size * map_size, (best_y + 0.5) / grid_size * map_size)


def _smooth(heatmap: Grid) -> Grid:
    height = len(heatmap)
    width = len(heatmap[0]) if height else 0
    out = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            values = []
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    values.append(heatmap[yy][xx])
            out[y][x] = sum(values) / len(values)
    return out

