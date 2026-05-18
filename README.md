# UAV City Behavior Sim

UAV City Behavior Sim is a lightweight Python project skeleton for simulating crowd behavior in a city scene, observing the crowd with a UAV, and inferring hidden hazards from human motion.

The current implementation can run locally without Isaac Sim. Isaac Sim-specific calls are kept as future integration points, so the project can first be tested as a simple 2D simulation loop.

## Features

- City scene manager with buildings, roads, obstacles, and exits
- Human agents with position, velocity, panic, state, and goals
- Crowd manager with simple panic contagion
- Hidden event manager for fire, accidents, or blocked areas
- UAV agent with pose, movement, camera placeholder, and next-best-view targeting
- Behavior tree framework for human decision logic
- Crowd feature extraction and hazard heatmap inference
- Episode runner, dataset generator, and heatmap evaluation script
- YAML configuration files for scene, humans, events, and UAV settings

## Project Structure

```text
.
|-- behavior_tree/
|   |-- core.py              # Node, Selector, Sequence, Status
|   |-- conditions.py        # FireVisible, NearbyPeopleRunning, CrowdDensityHigh
|   |-- actions.py           # Panic, escape, follow-crowd, walk, avoid-obstacle actions
|   `-- human_trees.py       # Personality-based human behavior trees
|-- configs/
|   |-- scene_city.yaml      # City bounds, buildings, roads, obstacles, exits
|   |-- humans.yaml          # Initial human positions, goals, personalities
|   |-- events.yaml          # Hidden events such as fire sources
|   `-- uav.yaml             # UAV pose, speed, and camera parameters
|-- experiments/
|   |-- run_episode.py       # Run one complete simulation episode
|   |-- generate_dataset.py  # Batch-generate episodes
|   `-- evaluate_heatmap.py  # Evaluate predicted hazard heatmaps
|-- inference/
|   |-- crowd_features.py    # Crowd motion feature extraction
|   |-- hazard_heatmap.py    # Hazard heatmap prediction
|   `-- next_best_view.py    # UAV next-best-view selection
`-- sim/
    |-- app.py               # Main simulation assembly and loop
    |-- scene_manager.py     # City map and spatial queries
    |-- human_agent.py       # Single human agent
    |-- human_manager.py     # Crowd-level updates
    |-- event_manager.py     # Hidden event lifecycle and visibility
    |-- data_logger.py       # Frame logging to JSON
    |-- uav_agent.py         # UAV motion and camera placeholder
    `-- config_loader.py     # YAML loader with a simple fallback parser
```

## Requirements

- Python 3.10+
- No required third-party package for the default demo
- Optional: `PyYAML` for richer YAML parsing
- Optional later: Isaac Sim for real scene primitives, sensors, and rendering

## Quick Start

Run one short smoke test:

```powershell
python -m experiments.run_episode --frames 5 --output-dir outputs/episodes/smoke_test
```

Run a default episode:

```powershell
python -m experiments.run_episode --frames 120 --output-dir outputs/episodes/default
```

The script prints the generated log path, for example:

```text
outputs\episodes\default\frames.json
```

## Configuration

By default, `experiments/run_episode.py` reads:

```text
configs/scene_city.yaml
configs/humans.yaml
configs/events.yaml
configs/uav.yaml
```

You can provide custom config paths:

```powershell
python -m experiments.run_episode `
  --frames 120 `
  --output-dir outputs/episodes/custom `
  --scene-config configs/scene_city.yaml `
  --humans-config configs/humans.yaml `
  --events-config configs/events.yaml `
  --uav-config configs/uav.yaml
```

## Output Format

Each episode writes a `frames.json` file. Every frame contains:

- `frame`: frame index
- `humans`: per-human position, velocity, state, and panic value
- `uav_pose`: UAV position and yaw
- `events`: event metadata and active/visible state

## Batch Dataset Generation

Generate multiple episodes:

```powershell
python -m experiments.generate_dataset --num-episodes 10 --output-root outputs/episodes --seed 7
```

## Evaluation

`experiments/evaluate_heatmap.py` exposes `evaluate_heatmap()`, which returns:

- `localization_error_m`
- `top_k_hit_rate`
- `heatmap_iou`

This is designed to compare the predicted hazard peak against a known ground-truth event position.

## Isaac Sim Integration Points

The current code is intentionally Isaac Sim-friendly but not Isaac Sim-dependent.

Recommended integration targets:

- `sim/scene_manager.py`: connect city geometry to `get_stage()` and `create_prim()`
- `sim/human_agent.py`: replace local 2D position updates with Isaac Sim transforms such as `apply_translation()`
- `sim/uav_agent.py`: connect `CameraSensor()`, `set_world_pose()`, and `capture_image()`
- `sim/app.py`: connect the loop to Isaac Sim stage, viewport, camera capture, and simulation clock

## Suggested Development Order

1. Confirm the local 2D loop runs with `experiments.run_episode`.
2. Add richer scene and human configs.
3. Improve behavior tree actions and conditions.
4. Tune hazard heatmap inference.
5. Connect UAV camera and movement to Isaac Sim.
6. Add rendered image outputs and dataset export formats such as PNG or NPZ.
