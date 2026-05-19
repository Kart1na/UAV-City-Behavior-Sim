# UAV City Behavior Sim

UAV City Behavior Sim simulates abnormal urban events in a 3D Isaac Sim city scene, observes the crowd with a UAV, and infers hidden hazards from human motion.

The project keeps a lightweight local Python loop for fast logic testing, but the primary runtime is now `experiments/run_isaac_episode.py`, which creates a real USD scene in Isaac Sim with city geometry, humans, event markers, a UAV body, and a UAV camera.

## Features

- City scene manager with buildings, roads, obstacles, and exits
- Human agents with position, velocity, panic, state, and goals
- Crowd manager with simple panic contagion
- Hidden event manager for fire, accidents, or blocked areas
- UAV agent with pose, movement, Isaac Sim camera prim, and next-best-view targeting
- Behavior tree framework for human decision logic
- Crowd feature extraction and hazard heatmap inference
- Isaac Sim episode runner, local episode runner, dataset generator, and heatmap evaluation script
- YAML configuration files for scene, humans, events, and UAV settings
- Optional USD city reference support through `configs/scene_city.yaml`

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
|   |-- run_isaac_episode.py # Run and render the episode in Isaac Sim
|   |-- generate_dataset.py  # Batch-generate episodes
|   `-- evaluate_heatmap.py  # Evaluate predicted hazard heatmaps
|-- inference/
|   |-- crowd_features.py    # Crowd motion feature extraction
|   |-- hazard_heatmap.py    # Hazard heatmap prediction
|   `-- next_best_view.py    # UAV next-best-view selection
`-- sim/
    |-- app.py               # Main simulation assembly and loop
    |-- isaac_runtime.py     # Isaac Sim USD scene, prim, camera, and sync adapter
    |-- scene_manager.py     # City map and spatial queries
    |-- human_agent.py       # Single human agent
    |-- human_manager.py     # Crowd-level updates
    |-- event_manager.py     # Hidden event lifecycle and visibility
    |-- data_logger.py       # Frame logging to JSON
    |-- uav_agent.py         # UAV motion and camera placeholder
    `-- config_loader.py     # YAML loader with a simple fallback parser
```

## Requirements

- Python 3.10+ for the local loop
- Isaac Sim for the 3D runtime
- No required third-party package for the default demo
- Optional: `PyYAML` for richer YAML parsing

## Isaac Sim Quick Start

Run the 3D city anomaly simulation from an Isaac Sim shell:

```powershell
isaac-sim.bat --no-window --python experiments/run_isaac_episode.py --headless --frames 240 --output-dir outputs/isaac/default
```

Or run it with the Isaac Sim UI:

```powershell
isaac-sim.bat --python experiments/run_isaac_episode.py --frames 240 --output-dir outputs/isaac/default
```

The Isaac runner writes:

- `outputs/isaac/default/frames.json`: frame-by-frame humans, events, and UAV pose
- `outputs/isaac/default/city_anomaly_episode.usd`: the generated 3D USD scene

If your Isaac installation uses a different launcher path, call the same script with Isaac Sim's bundled Python interpreter. A normal system Python cannot import Isaac Sim modules.

## Local Quick Start

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

By default, both `experiments/run_episode.py` and `experiments/run_isaac_episode.py` read:

```text
configs/scene_city.yaml
configs/humans.yaml
configs/events.yaml
configs/uav.yaml
```

`configs/scene_city.yaml` describes the city footprint used by both runtimes. Isaac Sim turns those rectangles into 3D USD geometry. You can also reference a detailed USD city asset:

```yaml
usd_reference: assets/my_city.usd
use_procedural_geometry: false
```

Set `use_procedural_geometry: true` if you want the referenced city plus the generated roads/buildings/obstacles.

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

## Isaac Sim Runtime

`sim/isaac_runtime.py` is the runtime bridge. It launches Isaac Sim, creates a USD stage, builds city geometry, instantiates event markers, creates human capsules, creates the UAV and camera prim, then synchronizes Python simulation state into Isaac Sim every frame.

The 2D logic remains the source of behavior state. Isaac Sim is responsible for the actual 3D scene, visualization, camera, and exported USD. This keeps behavior-tree and inference work testable without requiring Isaac Sim for every edit.

## Suggested Development Order

1. Replace capsule humans with animated people assets if your Isaac Sim install includes a crowd/people extension.
2. Attach real smoke/fire/traffic accident assets to event kinds.
3. Add camera RGB/depth/segmentation export through Isaac Replicator.
4. Improve behavior-tree navigation with path planning around obstacles.
5. Expand dataset export formats such as PNG, NPZ, or COCO-style annotations.
