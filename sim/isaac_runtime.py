"""Isaac Sim visual runtime adapter for the city anomaly simulation.

The adapter supports two visual paths:

1. Reference real USD assets supplied in ``configs/scene_city.yaml``.
2. Fall back to detailed procedural USD assets so the project still produces a
   recognizable final scene without downloading external content.

All Isaac imports stay inside ``IsaacCityRuntime.launch()`` so normal Python can
still import and test the behavior/inference code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot, radians, tan
from pathlib import Path
from typing import Any

from sim.event_manager import EventManager
from sim.human_manager import HumanManager
from sim.scene_manager import Rect, SceneManager
from sim.uav_agent import UAVAgent


class IsaacUnavailableError(RuntimeError):
    """Raised when Isaac Sim modules are unavailable in the active Python."""


@dataclass
class IsaacRuntimeConfig:
    headless: bool = False
    physics_dt: float = 1.0 / 30.0
    render_dt: float = 1.0 / 30.0
    world_path: str = "/World"
    city_path: str = "/World/City"
    agents_path: str = "/World/Agents"
    events_path: str = "/World/Events"
    uav_path: str = "/World/UAV"
    default_building_height: float = 14.0
    default_obstacle_height: float = 1.5
    road_thickness: float = 0.06
    human_height: float = 1.7
    human_radius: float = 0.22


@dataclass
class IsaacCityRuntime:
    """Create and update a 3D Isaac Sim city scene from project state."""

    config: IsaacRuntimeConfig = field(default_factory=IsaacRuntimeConfig)
    simulation_app: Any | None = None
    stage: Any | None = None
    _omni_usd: Any | None = None
    _timeline: Any | None = None
    _UsdGeom: Any | None = None
    _UsdLux: Any | None = None
    _UsdShade: Any | None = None
    _Sdf: Any | None = None
    _Gf: Any | None = None
    _kit_app: Any | None = None
    _asset_refs: dict[str, Any] = field(default_factory=dict)
    _config_base_dir: Path | None = None
    _human_prims: dict[str, Any] = field(default_factory=dict)
    _human_state_prims: dict[str, list[Any]] = field(default_factory=dict)
    _event_prims: dict[str, Any] = field(default_factory=dict)
    _materials: dict[str, Any] = field(default_factory=dict)

    def launch(self) -> "IsaacCityRuntime":
        """Attach to a running Isaac Sim app or launch one from Isaac Python."""
        running_app = self._attach_to_running_app()
        if not running_app:
            try:
                try:
                    from isaacsim import SimulationApp
                except ImportError:  # Isaac Sim 2023/2024 compatibility
                    from omni.isaac.kit import SimulationApp
            except ImportError as exc:  # pragma: no cover - requires Isaac Sim
                raise IsaacUnavailableError(
                    "Isaac Sim Python modules were not found. Run this script with "
                    "Isaac Sim's bundled Python, for example isaac-sim.sh --no-window "
                    "--python experiments/run_isaac_episode.py"
                ) from exc

            self.simulation_app = SimulationApp({"headless": self.config.headless})

        try:
            import omni.kit.app  # type: ignore[import-not-found]
            import omni.timeline  # type: ignore[import-not-found]
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires Isaac Sim
            raise IsaacUnavailableError(
                "Isaac Sim modules were not available after launch. Run from an "
                "open Isaac Sim Script Editor or Isaac Sim's bundled python.sh."
            ) from exc

        self._kit_app = omni.kit.app.get_app()
        self._omni_usd = omni.usd
        self._timeline = omni.timeline.get_timeline_interface()
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._UsdShade = UsdShade
        self._Sdf = Sdf
        self._Gf = Gf

        context = self._omni_usd.get_context()
        context.new_stage()
        self._update_app()
        self.stage = context.get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.Xform.Define(self.stage, self.config.world_path)
        return self

    def build_scene(self, scene: SceneManager, events: EventManager, humans: HumanManager, uav: UAVAgent) -> None:
        """Create static city geometry plus dynamic agents, events, and UAV."""
        self._require_stage()
        self._asset_refs = dict(scene.raw_config.get("assets", {}) or {})
        self._config_base_dir = scene.config_path.parent if scene.config_path else None
        self._clear_paths(
            [
                f"{self.config.world_path}/Lights",
                f"{self.config.world_path}/Materials",
                f"{self.config.world_path}/OverviewCamera",
                self.config.city_path,
                self.config.agents_path,
                self.config.events_path,
                self.config.uav_path,
            ]
        )
        self._ensure_materials()

        self._UsdGeom.Xform.Define(self.stage, self.config.city_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.agents_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.events_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.uav_path)
        self._create_lighting(scene)
        self._create_overview_camera(scene)

        use_procedural = bool(scene.raw_config.get("use_procedural_geometry", True))
        self._attach_reference(self.config.city_path, scene.raw_config.get("usd_reference") or self._asset_refs.get("city_usd"))
        if use_procedural:
            self._create_ground(scene)
            for road in scene.roads:
                self._create_visual_road(f"{self.config.city_path}/Roads/{road.id}", road)
            for building in scene.buildings:
                self._create_visual_building(f"{self.config.city_path}/Buildings/{building.id}", building)
            for obstacle in scene.obstacles:
                if obstacle.kind == "building":
                    continue
                self._create_visual_obstacle(f"{self.config.city_path}/Obstacles/{obstacle.id}", obstacle)
            self._create_city_dressing(scene)

        self._create_events(events)
        self._create_humans(humans)
        self._create_uav(uav)
        self.sync(humans, events, uav)

    def sync(self, humans: HumanManager, events: EventManager, uav: UAVAgent) -> None:
        """Push current Python simulation state into Isaac Sim transforms."""
        self._sync_humans(humans)
        self._sync_events(events)
        self._sync_uav(uav)

    def step(self, render: bool = True) -> None:
        """Advance Isaac Sim by one app update."""
        self._require_stage()
        if self._timeline and not self._timeline.is_playing():
            self._timeline.play()
        self._update_app()
        if render:
            self._update_app()

    def save_stage(self, path: str | Path) -> Path:
        """Export the generated USD scene for inspection or later replay."""
        self._require_stage()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stage.GetRootLayer().Export(str(output))
        return output

    def close(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.close()

    def _attach_to_running_app(self) -> bool:
        """Return True when executing inside an already-open Isaac Sim UI."""
        try:
            import omni.usd  # type: ignore[import-not-found]
        except ImportError:
            return False

        try:
            import omni.kit.app  # type: ignore[import-not-found]

            self._kit_app = omni.kit.app.get_app()
        except ImportError:
            self._kit_app = None

        return True

    def _update_app(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.update()

    def _require_stage(self) -> None:
        if self.stage is None:
            raise RuntimeError("IsaacCityRuntime.launch() must be called before using the runtime.")

    def _clear_paths(self, paths: list[str]) -> None:
        for path in paths:
            if self.stage.GetPrimAtPath(path):
                self.stage.RemovePrim(path)

    def _ensure_materials(self) -> None:
        material_specs = {
            "ground": ((0.13, 0.18, 0.15), 0.85, None, 1.0, 0.0),
            "grass_dark": ((0.07, 0.22, 0.09), 0.9, None, 1.0, 0.0),
            "road": ((0.025, 0.027, 0.030), 0.55, None, 1.0, 0.0),
            "sidewalk": ((0.45, 0.45, 0.42), 0.8, None, 1.0, 0.0),
            "lane_white": ((0.95, 0.93, 0.82), 0.35, None, 1.0, 0.0),
            "lane_yellow": ((1.0, 0.72, 0.08), 0.4, None, 1.0, 0.0),
            "building": ((0.37, 0.40, 0.43), 0.62, None, 1.0, 0.0),
            "building_alt": ((0.46, 0.39, 0.35), 0.68, None, 1.0, 0.0),
            "roof": ((0.10, 0.11, 0.13), 0.7, None, 1.0, 0.0),
            "window_lit": ((0.95, 0.76, 0.38), 0.2, (0.95, 0.58, 0.18), 1.0, 0.0),
            "window_dark": ((0.04, 0.08, 0.12), 0.2, None, 1.0, 0.0),
            "metal": ((0.36, 0.38, 0.40), 0.35, None, 1.0, 0.25),
            "obstacle": ((0.58, 0.42, 0.24), 0.55, None, 1.0, 0.0),
            "vehicle_red": ((0.75, 0.06, 0.05), 0.42, None, 1.0, 0.0),
            "vehicle_blue": ((0.05, 0.14, 0.52), 0.42, None, 1.0, 0.0),
            "vehicle_black": ((0.015, 0.015, 0.018), 0.55, None, 1.0, 0.0),
            "skin": ((0.80, 0.55, 0.38), 0.58, None, 1.0, 0.0),
            "human_walking": ((0.08, 0.35, 0.95), 0.48, None, 1.0, 0.0),
            "human_running": ((0.95, 0.12, 0.06), 0.42, None, 1.0, 0.0),
            "human_blocked": ((0.95, 0.72, 0.08), 0.5, None, 1.0, 0.0),
            "pants": ((0.08, 0.09, 0.12), 0.62, None, 1.0, 0.0),
            "event_fire": ((1.0, 0.22, 0.03), 0.15, (1.0, 0.18, 0.02), 1.0, 0.0),
            "event_flame_yellow": ((1.0, 0.78, 0.05), 0.18, (1.0, 0.55, 0.02), 1.0, 0.0),
            "event_smoke": ((0.34, 0.34, 0.34), 0.9, None, 0.55, 0.0),
            "event_accident": ((1.0, 0.58, 0.02), 0.4, (1.0, 0.35, 0.02), 1.0, 0.0),
            "event_blockage": ((0.72, 0.08, 0.07), 0.5, None, 1.0, 0.0),
            "hazard_area": ((1.0, 0.18, 0.04), 0.7, None, 0.28, 0.0),
            "uav": ((0.04, 0.70, 0.68), 0.36, None, 1.0, 0.1),
            "camera": ((0.01, 0.01, 0.012), 0.35, None, 1.0, 0.0),
            "tree_trunk": ((0.35, 0.20, 0.10), 0.7, None, 1.0, 0.0),
            "tree_leaf": ((0.08, 0.38, 0.14), 0.85, None, 1.0, 0.0),
            "lamp": ((1.0, 0.86, 0.45), 0.2, (1.0, 0.70, 0.28), 1.0, 0.0),
        }
        for name, (color, roughness, emissive, opacity, metallic) in material_specs.items():
            self._materials[name] = self._create_material(name, color, roughness, emissive, opacity, metallic)

    def _create_material(
        self,
        name: str,
        color: tuple[float, float, float],
        roughness: float = 0.65,
        emissive: tuple[float, float, float] | None = None,
        opacity: float = 1.0,
        metallic: float = 0.0,
    ) -> Any:
        path = f"{self.config.world_path}/Materials/{name}"
        material = self._UsdShade.Material.Define(self.stage, path)
        shader = self._UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", self._Sdf.ValueTypeNames.Float).Set(float(metallic))
        if emissive is not None:
            shader.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set(emissive)
        if opacity < 1.0:
            shader.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind(self, prim: Any, material_name: str) -> None:
        material = self._materials.get(material_name)
        if material:
            self._UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _create_ground(self, scene: SceneManager) -> None:
        self._create_box(
            f"{self.config.city_path}/Ground/Base",
            center=(scene.width / 2.0, scene.height / 2.0, -0.04),
            scale=(scene.width + 16.0, scene.height + 16.0, 0.08),
            material_name="ground",
        )
        self._create_box(
            f"{self.config.city_path}/Ground/GrassInset",
            center=(scene.width / 2.0, scene.height / 2.0, -0.015),
            scale=(scene.width, scene.height, 0.025),
            material_name="grass_dark",
        )

    def _create_lighting(self, scene: SceneManager) -> None:
        dome = self._UsdLux.DomeLight.Define(self.stage, f"{self.config.world_path}/Lights/Dome")
        dome.CreateIntensityAttr(550.0)
        dome.CreateColorAttr(self._Gf.Vec3f(0.92, 0.96, 1.0))

        sun = self._UsdLux.DistantLight.Define(self.stage, f"{self.config.world_path}/Lights/Sun")
        sun.CreateIntensityAttr(2600.0)
        sun.CreateAngleAttr(0.8)
        self._UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(
            self._Gf.Vec3f(-48.0, 0.0, -38.0),
            self._UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    def _create_overview_camera(self, scene: SceneManager) -> None:
        camera = self._UsdGeom.Camera.Define(self.stage, f"{self.config.world_path}/OverviewCamera")
        center_x = scene.width / 2.0
        center_y = scene.height / 2.0
        self._UsdGeom.XformCommonAPI(camera.GetPrim()).SetTranslate(
            self._Gf.Vec3d(center_x, center_y - scene.height * 0.92, max(scene.width, scene.height) * 0.72)
        )
        self._UsdGeom.XformCommonAPI(camera.GetPrim()).SetRotate(
            self._Gf.Vec3f(58.0, 0.0, 0.0),
            self._UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        camera.CreateFocalLengthAttr(18.0)

    def _create_visual_road(self, path: str, road: Rect) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        self._create_rect_box(f"{path}/Asphalt", road, self.config.road_thickness, "road")
        horizontal = road.width >= road.height
        if horizontal:
            self._create_box(
                f"{path}/SidewalkNorth",
                center=(road.x + road.width / 2.0, road.y - 1.0, 0.035),
                scale=(road.width, 1.8, 0.08),
                material_name="sidewalk",
            )
            self._create_box(
                f"{path}/SidewalkSouth",
                center=(road.x + road.width / 2.0, road.y + road.height + 1.0, 0.035),
                scale=(road.width, 1.8, 0.08),
                material_name="sidewalk",
            )
            y_mid = road.y + road.height / 2.0
            for idx, x in enumerate(self._frange(road.x + 4.0, road.x + road.width - 4.0, 10.0)):
                self._create_box(
                    f"{path}/LaneDash_{idx}",
                    center=(x, y_mid, 0.09),
                    scale=(4.6, 0.22, 0.035),
                    material_name="lane_yellow",
                )
            self._create_crosswalk(path, road.x + 8.0, y_mid, horizontal=True)
            self._create_crosswalk(path, road.x + road.width - 8.0, y_mid, horizontal=True)
        else:
            self._create_box(
                f"{path}/SidewalkWest",
                center=(road.x - 1.0, road.y + road.height / 2.0, 0.035),
                scale=(1.8, road.height, 0.08),
                material_name="sidewalk",
            )
            self._create_box(
                f"{path}/SidewalkEast",
                center=(road.x + road.width + 1.0, road.y + road.height / 2.0, 0.035),
                scale=(1.8, road.height, 0.08),
                material_name="sidewalk",
            )
            x_mid = road.x + road.width / 2.0
            for idx, y in enumerate(self._frange(road.y + 4.0, road.y + road.height - 4.0, 10.0)):
                self._create_box(
                    f"{path}/LaneDash_{idx}",
                    center=(x_mid, y, 0.09),
                    scale=(0.22, 4.6, 0.035),
                    material_name="lane_yellow",
                )
            self._create_crosswalk(path, x_mid, road.y + 8.0, horizontal=False)
            self._create_crosswalk(path, x_mid, road.y + road.height - 8.0, horizontal=False)

    def _create_crosswalk(self, path: str, x: float, y: float, horizontal: bool) -> None:
        for idx in range(5):
            offset = (idx - 2) * 0.85
            if horizontal:
                center = (x, y + offset, 0.105)
                scale = (3.6, 0.36, 0.035)
            else:
                center = (x + offset, y, 0.105)
                scale = (0.36, 3.6, 0.035)
            self._create_box(f"{path}/Crosswalk_{int(x)}_{int(y)}_{idx}", center=center, scale=scale, material_name="lane_white")

    def _create_visual_building(self, path: str, building: Rect) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        height = self._building_height(building)
        material_name = "building_alt" if self._stable_index(building.id or path, 2) else "building"
        self._create_rect_box(f"{path}/Mass", building, height, material_name)
        self._create_box(
            f"{path}/Roof",
            center=(building.x + building.width / 2.0, building.y + building.height / 2.0, height + 0.12),
            scale=(building.width + 0.45, building.height + 0.45, 0.24),
            material_name="roof",
        )
        self._create_building_windows(path, building, height)
        self._create_rooftop_details(path, building, height)

    def _create_building_windows(self, path: str, building: Rect, height: float) -> None:
        floors = max(2, int(height // 3.0))
        columns_x = max(2, int(building.width // 3.2))
        columns_y = max(2, int(building.height // 3.2))
        for floor in range(floors):
            z = 1.8 + floor * 2.6
            if z > height - 1.0:
                continue
            for col in range(columns_x):
                x = building.x + 1.6 + col * ((building.width - 3.2) / max(1, columns_x - 1))
                lit = "window_lit" if (floor + col + self._stable_index(building.id or path, 3)) % 3 == 0 else "window_dark"
                self._create_box(
                    f"{path}/Windows/N_{floor}_{col}",
                    center=(x, building.y - 0.035, z),
                    scale=(0.8, 0.06, 1.0),
                    material_name=lit,
                )
                self._create_box(
                    f"{path}/Windows/S_{floor}_{col}",
                    center=(x, building.y + building.height + 0.035, z),
                    scale=(0.8, 0.06, 1.0),
                    material_name=lit,
                )
            for col in range(columns_y):
                y = building.y + 1.6 + col * ((building.height - 3.2) / max(1, columns_y - 1))
                lit = "window_lit" if (floor + col + self._stable_index(building.id or path, 5)) % 4 == 0 else "window_dark"
                self._create_box(
                    f"{path}/Windows/W_{floor}_{col}",
                    center=(building.x - 0.035, y, z),
                    scale=(0.06, 0.8, 1.0),
                    material_name=lit,
                )
                self._create_box(
                    f"{path}/Windows/E_{floor}_{col}",
                    center=(building.x + building.width + 0.035, y, z),
                    scale=(0.06, 0.8, 1.0),
                    material_name=lit,
                )

    def _create_rooftop_details(self, path: str, building: Rect, height: float) -> None:
        cx, cy = building.center()
        self._create_box(f"{path}/Roof/HVAC_A", center=(cx - 1.7, cy, height + 0.55), scale=(1.5, 1.1, 0.55), material_name="metal")
        self._create_box(f"{path}/Roof/HVAC_B", center=(cx + 1.5, cy + 1.2, height + 0.45), scale=(1.0, 1.4, 0.45), material_name="metal")

    def _create_visual_obstacle(self, path: str, obstacle: Rect) -> None:
        name = (obstacle.id or "").lower()
        cx, cy = obstacle.center()
        if "vehicle" in name or "car" in name:
            self._create_vehicle(path, center=(cx, cy, 0.0), material_name="vehicle_blue")
        elif "barrier" in name:
            self._create_barrier_row(path, center=(cx, cy, 0.0), width=max(obstacle.width, obstacle.height))
        elif "debris" in name:
            self._create_debris_pile(path, center=(cx, cy, 0.0))
        else:
            self._create_rect_box(path, obstacle, self.config.default_obstacle_height, "obstacle")

    def _create_city_dressing(self, scene: SceneManager) -> None:
        root = f"{self.config.city_path}/StreetFurniture"
        self._UsdGeom.Xform.Define(self.stage, root)
        tree_idx = 0
        lamp_idx = 0
        for road in scene.roads:
            horizontal = road.width >= road.height
            if horizontal:
                for x in self._frange(road.x + 10.0, road.x + road.width - 8.0, 18.0):
                    for y in (road.y - 4.0, road.y + road.height + 4.0):
                        if self._in_bounds(scene, x, y) and not self._inside_rects((x, y), scene.buildings):
                            self._create_tree(f"{root}/Tree_{tree_idx}", (x, y, 0.0))
                            tree_idx += 1
                    if int(x) % 36 == 0:
                        self._create_streetlight(f"{root}/Lamp_{lamp_idx}", (x, road.y - 2.4, 0.0))
                        lamp_idx += 1
            else:
                for y in self._frange(road.y + 10.0, road.y + road.height - 8.0, 18.0):
                    for x in (road.x - 4.0, road.x + road.width + 4.0):
                        if self._in_bounds(scene, x, y) and not self._inside_rects((x, y), scene.buildings):
                            self._create_tree(f"{root}/Tree_{tree_idx}", (x, y, 0.0))
                            tree_idx += 1
                    if int(y) % 36 == 0:
                        self._create_streetlight(f"{root}/Lamp_{lamp_idx}", (road.x - 2.4, y, 0.0))
                        lamp_idx += 1

    def _create_events(self, events: EventManager) -> None:
        for event in events.events:
            root_path = f"{self.config.events_path}/{event.id}"
            root = self._UsdGeom.Xform.Define(self.stage, root_path).GetPrim()
            self._UsdGeom.XformCommonAPI(root).SetTranslate(self._Gf.Vec3d(event.position[0], event.position[1], 0.0))
            self._event_prims[event.id] = root
            if event.kind == "accident":
                self._create_accident_event(root_path, event.radius)
            elif event.kind in {"blocked", "blockage", "barrier"}:
                self._create_blockage_event(root_path, event.radius)
            else:
                self._create_fire_event(root_path, event.radius)

    def _create_fire_event(self, root_path: str, radius: float) -> None:
        self._create_influence_disk(root_path, radius, "hazard_area")
        self._attach_reference(root_path, self._asset_refs.get("fire_usd"))
        for idx, (x, y, z, r, mat) in enumerate(
            [
                (0.0, 0.0, 0.9, 1.0, "event_fire"),
                (-0.55, 0.25, 1.25, 0.72, "event_flame_yellow"),
                (0.48, -0.18, 1.45, 0.62, "event_fire"),
                (0.08, 0.42, 1.85, 0.45, "event_flame_yellow"),
            ]
        ):
            self._create_sphere(f"{root_path}/Flames/Flame_{idx}", center=(x, y, z), radius=r, material_name=mat)
        for idx, (x, y, z, r) in enumerate([(-0.3, 0.2, 2.7, 0.8), (0.35, -0.1, 3.5, 1.05), (0.0, 0.3, 4.4, 1.25)]):
            self._create_sphere(f"{root_path}/Smoke/Smoke_{idx}", center=(x, y, z), radius=r, material_name="event_smoke")
        light = self._UsdLux.SphereLight.Define(self.stage, f"{root_path}/FireLight")
        light.CreateIntensityAttr(4200.0)
        light.CreateRadiusAttr(5.5)
        light.CreateColorAttr(self._Gf.Vec3f(1.0, 0.36, 0.06))
        self._UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(self._Gf.Vec3d(0.0, 0.0, 2.2))

    def _create_accident_event(self, root_path: str, radius: float) -> None:
        self._create_influence_disk(root_path, radius, "event_accident")
        self._create_vehicle(f"{root_path}/Vehicle_A", center=(-2.2, -0.6, 0.0), material_name="vehicle_red")
        self._create_vehicle(f"{root_path}/Vehicle_B", center=(2.3, 0.8, 0.0), material_name="vehicle_blue")
        for idx, x in enumerate([-4.0, -2.2, 0.0, 2.2, 4.0]):
            self._create_cone(f"{root_path}/TrafficCones/Cone_{idx}", center=(x, -3.2, 0.45), radius=0.24, height=0.9, material_name="event_accident")
        light = self._UsdLux.SphereLight.Define(self.stage, f"{root_path}/WarningLight")
        light.CreateIntensityAttr(1600.0)
        light.CreateRadiusAttr(2.2)
        light.CreateColorAttr(self._Gf.Vec3f(1.0, 0.45, 0.02))
        self._UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(self._Gf.Vec3d(0.0, 0.0, 2.2))

    def _create_blockage_event(self, root_path: str, radius: float) -> None:
        self._create_influence_disk(root_path, radius, "event_blockage")
        self._create_barrier_row(f"{root_path}/BarrierRow", center=(0.0, 0.0, 0.0), width=7.0)
        self._create_debris_pile(f"{root_path}/Debris", center=(1.6, 1.1, 0.0))

    def _create_influence_disk(self, root_path: str, radius: float, material_name: str) -> None:
        disk = self._UsdGeom.Cylinder.Define(self.stage, f"{root_path}/InfluenceRadius")
        disk.CreateRadiusAttr(radius)
        disk.CreateHeightAttr(0.06)
        disk_prim = disk.GetPrim()
        self._UsdGeom.XformCommonAPI(disk_prim).SetTranslate(self._Gf.Vec3d(0.0, 0.0, 0.05))
        self._bind(disk_prim, material_name)

    def _create_humans(self, humans: HumanManager) -> None:
        for human in humans.humans:
            path = f"{self.config.agents_path}/{human.id}"
            root = self._UsdGeom.Xform.Define(self.stage, path).GetPrim()
            self._human_prims[human.id] = root
            human_asset = self._asset_refs.get("human_usd") or self._asset_refs.get("people_usd")
            if self._attach_reference(path, human_asset):
                self._human_state_prims[human.id] = []
                continue
            state_prims = self._create_procedural_human(path, human.personality)
            self._human_state_prims[human.id] = state_prims

    def _create_procedural_human(self, path: str, personality: str) -> list[Any]:
        shirt_material = "human_walking"
        if personality == "cautious":
            shirt_material = "human_blocked"
        elif personality == "curious":
            shirt_material = "human_running"
        torso = self._create_capsule(f"{path}/Torso", center=(0.0, 0.0, 0.98), height=0.72, radius=0.22, material_name=shirt_material)
        self._create_sphere(f"{path}/Head", center=(0.0, 0.0, 1.48), radius=0.18, material_name="skin")
        self._create_box(f"{path}/Leg_L", center=(-0.09, 0.0, 0.42), scale=(0.10, 0.12, 0.72), material_name="pants")
        self._create_box(f"{path}/Leg_R", center=(0.09, 0.0, 0.42), scale=(0.10, 0.12, 0.72), material_name="pants")
        self._create_box(f"{path}/Arm_L", center=(-0.30, 0.0, 0.98), scale=(0.09, 0.10, 0.62), material_name="skin")
        self._create_box(f"{path}/Arm_R", center=(0.30, 0.0, 0.98), scale=(0.09, 0.10, 0.62), material_name="skin")
        return [torso]

    def _create_uav(self, uav: UAVAgent) -> None:
        self._attach_reference(self.config.uav_path, self._asset_refs.get("uav_usd"))
        self._create_box(f"{self.config.uav_path}/Body", center=(0.0, 0.0, 0.0), scale=(1.0, 0.65, 0.22), material_name="uav")
        self._create_box(f"{self.config.uav_path}/Arm_X", center=(0.0, 0.0, 0.0), scale=(2.4, 0.08, 0.08), material_name="metal")
        self._create_box(f"{self.config.uav_path}/Arm_Y", center=(0.0, 0.0, 0.0), scale=(0.08, 2.4, 0.08), material_name="metal")
        for idx, (x, y) in enumerate([(-1.25, -1.25), (-1.25, 1.25), (1.25, -1.25), (1.25, 1.25)]):
            self._create_cylinder(f"{self.config.uav_path}/Rotor_{idx}", center=(x, y, 0.04), radius=0.42, height=0.035, material_name="metal")
        camera = self._UsdGeom.Camera.Define(self.stage, f"{self.config.uav_path}/Camera")
        params = uav.camera_params or {}
        fov_degrees = float(params.get("fov_degrees", 70.0))
        horizontal_aperture = 20.955
        focal_length = horizontal_aperture / (2.0 * tan(radians(fov_degrees) / 2.0))
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        camera_prim = camera.GetPrim()
        self._UsdGeom.XformCommonAPI(camera_prim).SetTranslate(self._Gf.Vec3d(0.0, -0.55, -0.18))
        self._UsdGeom.XformCommonAPI(camera_prim).SetRotate(
            self._Gf.Vec3f(65.0, 0.0, 0.0),
            self._UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        self._bind(camera_prim, "camera")

    def _sync_humans(self, humans: HumanManager) -> None:
        for human in humans.humans:
            prim = self._human_prims.get(human.id)
            if prim is None:
                continue
            xform = self._UsdGeom.XformCommonAPI(prim)
            xform.SetTranslate(self._Gf.Vec3d(human.position[0], human.position[1], 0.0))
            if hypot(human.velocity[0], human.velocity[1]) > 0.02:
                xform.SetRotate(
                    self._Gf.Vec3f(0.0, 0.0, degrees(atan2(human.velocity[1], human.velocity[0]))),
                    self._UsdGeom.XformCommonAPI.RotationOrderXYZ,
                )
            material = "human_running" if human.state == "running" else "human_blocked" if human.state == "blocked" else "human_walking"
            for body_prim in self._human_state_prims.get(human.id, []):
                self._bind(body_prim, material)

    def _sync_events(self, events: EventManager) -> None:
        for event in events.events:
            prim = self._event_prims.get(event.id)
            if prim is None:
                continue
            prim.SetActive(event.active)

    def _sync_uav(self, uav: UAVAgent) -> None:
        prim = self.stage.GetPrimAtPath(self.config.uav_path)
        if not prim:
            return
        xform = self._UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(self._Gf.Vec3d(*uav.position))
        xform.SetRotate(self._Gf.Vec3f(0.0, 0.0, degrees(uav.yaw)), self._UsdGeom.XformCommonAPI.RotationOrderXYZ)

    def _create_vehicle(self, path: str, center: tuple[float, float, float], material_name: str) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        if self._attach_reference(path, self._asset_refs.get("vehicle_usd")):
            self._UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(path)).SetTranslate(self._Gf.Vec3d(*center))
            return
        self._create_box(f"{path}/Body", center=(center[0], center[1], center[2] + 0.45), scale=(3.2, 1.55, 0.65), material_name=material_name)
        self._create_box(f"{path}/Cabin", center=(center[0] + 0.25, center[1], center[2] + 0.92), scale=(1.35, 1.20, 0.58), material_name="window_dark")
        for idx, (dx, dy) in enumerate([(-1.15, -0.82), (-1.15, 0.82), (1.15, -0.82), (1.15, 0.82)]):
            self._create_sphere(f"{path}/Wheel_{idx}", center=(center[0] + dx, center[1] + dy, center[2] + 0.20), radius=0.25, material_name="vehicle_black")
        self._create_box(f"{path}/Headlights", center=(center[0] + 1.68, center[1], center[2] + 0.52), scale=(0.05, 0.95, 0.16), material_name="lamp")

    def _create_barrier_row(self, path: str, center: tuple[float, float, float], width: float) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        count = max(3, int(width // 1.4))
        for idx in range(count):
            x = center[0] + (idx - (count - 1) / 2.0) * 1.25
            if self._asset_refs.get("barrier_usd"):
                child_path = f"{path}/BarrierAsset_{idx}"
                self._attach_reference(child_path, self._asset_refs.get("barrier_usd"))
                self._UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(child_path)).SetTranslate(
                    self._Gf.Vec3d(x, center[1], center[2])
                )
                continue
            self._create_box(f"{path}/Barrier_{idx}", center=(x, center[1], center[2] + 0.45), scale=(1.0, 0.26, 0.9), material_name="event_blockage")
            self._create_box(f"{path}/Stripe_{idx}", center=(x, center[1] - 0.14, center[2] + 0.48), scale=(0.72, 0.03, 0.18), material_name="lane_white")

    def _create_debris_pile(self, path: str, center: tuple[float, float, float]) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        offsets = [(-0.5, -0.2, 0.18), (0.2, 0.1, 0.28), (0.7, -0.3, 0.20), (-0.1, 0.6, 0.24)]
        for idx, (dx, dy, dz) in enumerate(offsets):
            self._create_box(f"{path}/Chunk_{idx}", center=(center[0] + dx, center[1] + dy, center[2] + dz), scale=(0.7, 0.45, 0.35), material_name="obstacle")

    def _create_tree(self, path: str, center: tuple[float, float, float]) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        if self._attach_reference(path, self._asset_refs.get("tree_usd")):
            self._UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(path)).SetTranslate(self._Gf.Vec3d(*center))
            return
        self._create_cylinder(f"{path}/Trunk", center=(center[0], center[1], center[2] + 0.9), radius=0.16, height=1.8, material_name="tree_trunk")
        self._create_sphere(f"{path}/Crown", center=(center[0], center[1], center[2] + 2.25), radius=0.9, material_name="tree_leaf")
        self._create_sphere(f"{path}/Crown_Offset", center=(center[0] + 0.35, center[1] - 0.15, center[2] + 2.55), radius=0.62, material_name="tree_leaf")

    def _create_streetlight(self, path: str, center: tuple[float, float, float]) -> None:
        self._UsdGeom.Xform.Define(self.stage, path)
        if self._attach_reference(path, self._asset_refs.get("streetlight_usd")):
            self._UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(path)).SetTranslate(self._Gf.Vec3d(*center))
            return
        self._create_cylinder(f"{path}/Pole", center=(center[0], center[1], center[2] + 1.7), radius=0.055, height=3.4, material_name="metal")
        self._create_box(f"{path}/Arm", center=(center[0] + 0.55, center[1], center[2] + 3.25), scale=(1.1, 0.08, 0.08), material_name="metal")
        self._create_sphere(f"{path}/Lamp", center=(center[0] + 1.12, center[1], center[2] + 3.18), radius=0.18, material_name="lamp")

    def _create_rect_box(self, path: str, rect: Rect, z_height: float, material_name: str) -> Any:
        return self._create_box(
            path,
            center=(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0, z_height / 2.0),
            scale=(rect.width, rect.height, z_height),
            material_name=material_name,
        )

    def _create_box(self, path: str, center: tuple[float, float, float], scale: tuple[float, float, float], material_name: str) -> Any:
        self._ensure_parent(path)
        cube = self._UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        xform = self._UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(self._Gf.Vec3d(*center))
        xform.SetScale(self._Gf.Vec3f(*scale))
        self._bind(prim, material_name)
        return prim

    def _create_sphere(self, path: str, center: tuple[float, float, float], radius: float, material_name: str) -> Any:
        self._ensure_parent(path)
        sphere = self._UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(radius)
        prim = sphere.GetPrim()
        self._UsdGeom.XformCommonAPI(prim).SetTranslate(self._Gf.Vec3d(*center))
        self._bind(prim, material_name)
        return prim

    def _create_cylinder(self, path: str, center: tuple[float, float, float], radius: float, height: float, material_name: str) -> Any:
        self._ensure_parent(path)
        cylinder = self._UsdGeom.Cylinder.Define(self.stage, path)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        prim = cylinder.GetPrim()
        self._UsdGeom.XformCommonAPI(prim).SetTranslate(self._Gf.Vec3d(*center))
        self._bind(prim, material_name)
        return prim

    def _create_capsule(self, path: str, center: tuple[float, float, float], radius: float, height: float, material_name: str) -> Any:
        self._ensure_parent(path)
        capsule = self._UsdGeom.Capsule.Define(self.stage, path)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(height)
        prim = capsule.GetPrim()
        self._UsdGeom.XformCommonAPI(prim).SetTranslate(self._Gf.Vec3d(*center))
        self._bind(prim, material_name)
        return prim

    def _create_cone(self, path: str, center: tuple[float, float, float], radius: float, height: float, material_name: str) -> Any:
        self._ensure_parent(path)
        cone = self._UsdGeom.Cone.Define(self.stage, path)
        cone.CreateRadiusAttr(radius)
        cone.CreateHeightAttr(height)
        prim = cone.GetPrim()
        self._UsdGeom.XformCommonAPI(prim).SetTranslate(self._Gf.Vec3d(*center))
        self._bind(prim, material_name)
        return prim

    def _attach_reference(self, path: str, reference: Any) -> bool:
        if not reference:
            return False
        if isinstance(reference, list):
            reference = reference[0] if reference else None
        if not reference:
            return False
        resolved = self._resolve_asset(reference)
        if not resolved:
            return False
        self._ensure_parent(path)
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            prim = self._UsdGeom.Xform.Define(self.stage, path).GetPrim()
        prim.GetReferences().AddReference(resolved)
        return True

    def _resolve_asset(self, reference: Any) -> str | None:
        text = str(reference).strip()
        if not text:
            return None
        if "://" in text:
            return text
        path = Path(text)
        if not path.is_absolute() and self._config_base_dir is not None:
            path = self._config_base_dir / path
        return str(path)

    def _building_height(self, rect: Rect) -> float:
        return self.config.default_building_height + float(self._stable_index(rect.id or "building", 7)) * 2.2

    def _stable_index(self, text: str, modulo: int) -> int:
        return sum(ord(char) for char in text) % max(1, modulo)

    def _inside_rects(self, position: tuple[float, float], rects: list[Rect]) -> bool:
        return any(rect.contains(position) for rect in rects)

    def _in_bounds(self, scene: SceneManager, x: float, y: float) -> bool:
        return 0.0 <= x <= scene.width and 0.0 <= y <= scene.height

    def _frange(self, start: float, stop: float, step: float) -> list[float]:
        values: list[float] = []
        value = start
        while value <= stop:
            values.append(value)
            value += step
        return values

    def _ensure_parent(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        current = ""
        for part in parts[:-1]:
            current = f"{current}/{part}"
            if not self.stage.GetPrimAtPath(current):
                self._UsdGeom.Xform.Define(self.stage, current)
