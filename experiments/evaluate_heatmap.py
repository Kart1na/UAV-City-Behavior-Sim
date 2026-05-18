"""Evaluate hazard heatmap localization."""

from __future__ import annotations

from math import dist

from inference.hazard_heatmap import Grid, peak_position


def evaluate_heatmap(heatmap: Grid, ground_truth_fire_position: tuple[float, float], top_k: int = 5) -> dict[str, float]:
    prediction = peak_position(heatmap)
    flat = sorted(
        ((value, x, y) for y, row in enumerate(heatmap) for x, value in enumerate(row)),
        reverse=True,
    )
    grid_size = len(heatmap) or 1
    gt_cell = (
        int(ground_truth_fire_position[0] / 100.0 * grid_size),
        int(ground_truth_fire_position[1] / 100.0 * grid_size),
    )
    top_cells = {(x, y) for _, x, y in flat[:top_k]}
    return {
        "localization_error_m": dist(prediction, ground_truth_fire_position),
        "top_k_hit_rate": 1.0 if gt_cell in top_cells else 0.0,
        "heatmap_iou": _threshold_iou(heatmap, gt_cell),
    }


def _threshold_iou(heatmap: Grid, gt_cell: tuple[int, int], threshold_ratio: float = 0.5) -> float:
    max_value = max((value for row in heatmap for value in row), default=0.0)
    if max_value <= 0:
        return 0.0
    predicted = {
        (x, y)
        for y, row in enumerate(heatmap)
        for x, value in enumerate(row)
        if value >= max_value * threshold_ratio
    }
    truth = {gt_cell}
    return len(predicted & truth) / len(predicted | truth) if predicted else 0.0

