"""Isaac Sim runtime adapter for the city anomaly simulation.

This module keeps all Isaac Sim imports behind ``IsaacCityRuntime.launch()`` so
the rest of the project remains importable in a normal Python environment. Run
it with Isaac Sim's Python interpreter or Script Editor, not stock CPython.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import degrees, tan, radians
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
    default_building_height: float = 12.0
    default_obstacle_height: float = 1.5
    road_thickness: float = 0.05
    human_height: float = 1.7
    human_radius: float = 0.28


@dataclass
class IsaacCityRuntime:
    """Create and update a real 3D Isaac Sim city scene from project state."""

    config: IsaacRuntimeConfig = field(default_factory=IsaacRuntimeConfig)
    simulation_app: Any | None = None
    stage: Any | None = None
    _omni_usd: Any | None = None
    _timeline: Any | None = None
    _UsdGeom: Any | None = None
    _UsdShade: Any | None = None
    _Sdf: Any | None = None
    _Gf: Any | None = None
    _kit_app: Any | None = None
    _human_prims: dict[str, Any] = field(default_factory=dict)
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
            import omni.timeline  # type: ignore[import-not-found]
            import omni.usd  # type: ignore[import-not-found]
            import omni.kit.app  # type: ignore[import-not-found]
            from pxr import Gf, Sdf, UsdGeom, UsdShade  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires Isaac Sim
            raise IsaacUnavailableError(
                "Isaac Sim modules were not available after launch. Run from an "
                "open Isaac Sim Script Editor or Isaac Sim's bundled python.sh."
            ) from exc

        self._kit_app = omni.kit.app.get_app()
        self._omni_usd = omni.usd
        self._timeline = omni.timeline.get_timeline_interface()
        self._UsdGeom = UsdGeom
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
        self._clear_paths([self.config.city_path, self.config.agents_path, self.config.events_path, self.config.uav_path])
        self._ensure_materials()

        self._UsdGeom.Xform.Define(self.stage, self.config.city_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.agents_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.events_path)
        self._UsdGeom.Xform.Define(self.stage, self.config.uav_path)

        use_procedural = bool(scene.raw_config.get("use_procedural_geometry", True))
        self._attach_city_reference(scene)
        if use_procedural:
            self._create_ground(scene)
            for road in scene.roads:
                self._create_rect_box(f"{self.config.city_path}/Roads/{road.id}", road, self.config.road_thickness, "road")
            for building in scene.buildings:
                self._create_rect_box(
                    f"{self.config.city_path}/Buildings/{building.id}",
                    building,
                    self.config.default_building_height,
                    "building",
                )
            for obstacle in scene.obstacles:
                if obstacle.kind == "building":
                    continue
                self._create_rect_box(
                    f"{self.config.city_path}/Obstacles/{obstacle.id}",
                    obstacle,
                    self.config.default_obstacle_height,
                    "obstacle",
                )

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
        elif self._kit_app is not None:
            self._kit_app.update()

    def _require_stage(self) -> None:
        if self.stage is None:
            raise RuntimeError("IsaacCityRuntime.launch() must be called before using the runtime.")

    def _clear_paths(self, paths: list[str]) -> None:
        for path in paths:
            if self.stage.GetPrimAtPath(path):
                self.stage.RemovePrim(path)

    def _ensure_materials(self) -> None:
        colors = {
            "ground": (0.18, 0.22, 0.18),
            "road": (0.04, 0.045, 0.05),
            "building": (0.45, 0.48, 0.50),
            "obstacle": (0.55, 0.42, 0.28),
            "human_walking": (0.12, 0.42, 0.85),
            "human_running": (0.95, 0.22, 0.12),
            "human_blocked": (0.85, 0.70, 0.10),
            "event_fire": (1.0, 0.22, 0.03),
            "event_accident": (1.0, 0.68, 0.05),
            "event_blockage": (0.70, 0.10, 0.10),
            "uav": (0.05, 0.78, 0.72),
            "camera": (0.02, 0.02, 0.02),
        }
        for name, color in colors.items():
            self._materials[name] = self._create_material(name, color)

    def _create_material(self, name: str, color: tuple[float, float, float]) -> Any:
        path = f"{self.config.world_path}/Materials/{name}"
        material = self._UsdShade.Material.Define(self.stage, path)
        shader = self._UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind(self, prim: Any, material_name: str) -> None:
        material = self._materials.get(material_name)
        if material:
            self._UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _create_ground(self, scene: SceneManager) -> None:
        self._create_box(
            f"{self.config.city_path}/Ground",
            center=(scene.width / 2.0, scene.height / 2.0, -0.03),
            scale=(scene.width, scene.height, 0.06),
            material_name="ground",
        )

    def _attach_city_reference(self, scene: SceneManager) -> None:
        assets = scene.raw_config.get("assets", {})
        reference_path = scene.raw_config.get("usd_reference") or assets.get("city_usd")
        if not reference_path:
            return
        reference = Path(reference_path)
        if scene.config_path and not reference.is_absolute():
            reference = scene.config_path.parent / reference
        root = self.stage.GetPrimAtPath(self.config.city_path)
        root.GetReferences().AddReference(str(reference))

    def _create_rect_box(self, path: str, rect: Rect, z_height: float, material_name: str) -> None:
        self._create_box(
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

    def _create_events(self, events: EventManager) -> None:
        for event in events.events:
            root_path = f"{self.config.events_path}/{event.id}"
            self._ensure_parent(root_path)
            self._UsdGeom.Xform.Define(self.stage, root_path)
            disk = self._UsdGeom.Cylinder.Define(self.stage, f"{root_path}/InfluenceRadius")
            disk.CreateRadiusAttr(event.radius)
            disk.CreateHeightAttr(0.08)
            disk_prim = disk.GetPrim()
            self._UsdGeom.XformCommonAPI(disk_prim).SetTranslate(self._Gf.Vec3d(event.position[0], event.position[1], 0.04))
            self._bind(disk_prim, self._event_material_name(event.kind))

            marker = self._UsdGeom.Sphere.Define(self.stage, f"{root_path}/Marker")
            marker.CreateRadiusAttr(1.4)
            marker_prim = marker.GetPrim()
            self._UsdGeom.XformCommonAPI(marker_prim).SetTranslate(self._Gf.Vec3d(event.position[0], event.position[1], 1.8))
            self._bind(marker_prim, self._event_material_name(event.kind))
            self._event_prims[event.id] = marker_prim

    def _create_humans(self, humans: HumanManager) -> None:
        for human in humans.humans:
            path = f"{self.config.agents_path}/{human.id}"
            self._ensure_parent(path)
            capsule = self._UsdGeom.Capsule.Define(self.stage, path)
            capsule.CreateHeightAttr(self.config.human_height)
            capsule.CreateRadiusAttr(self.config.human_radius)
            prim = capsule.GetPrim()
            self._human_prims[human.id] = prim
            self._bind(prim, "human_walking")

    def _create_uav(self, uav: UAVAgent) -> None:
        body = self._create_box(f"{self.config.uav_path}/Body", center=uav.position, scale=(1.2, 1.2, 0.25), material_name="uav")
        self._bind(body, "uav")
        camera = self._UsdGeom.Camera.Define(self.stage, f"{self.config.uav_path}/Camera")
        params = uav.camera_params or {}
        fov_degrees = float(params.get("fov_degrees", 70.0))
        horizontal_aperture = 20.955
        focal_length = horizontal_aperture / (2.0 * tan(radians(fov_degrees) / 2.0))
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        self._bind(camera.GetPrim(), "camera")

    def _sync_humans(self, humans: HumanManager) -> None:
        for human in humans.humans:
            prim = self._human_prims.get(human.id)
            if prim is None:
                continue
            self._UsdGeom.XformCommonAPI(prim).SetTranslate(
                self._Gf.Vec3d(human.position[0], human.position[1], self.config.human_height / 2.0)
            )
            material = "human_running" if human.state == "running" else "human_blocked" if human.state == "blocked" else "human_walking"
            self._bind(prim, material)

    def _sync_events(self, events: EventManager) -> None:
        for event in events.events:
            prim = self._event_prims.get(event.id)
            if prim is None:
                continue
            prim.SetActive(event.active)

    def _sync_uav(self, uav: UAVAgent) -> None:
        yaw = degrees(uav.yaw)
        for child_name in ("Body", "Camera"):
            prim = self.stage.GetPrimAtPath(f"{self.config.uav_path}/{child_name}")
            if not prim:
                continue
            xform = self._UsdGeom.XformCommonAPI(prim)
            xform.SetTranslate(self._Gf.Vec3d(*uav.position))
            if child_name == "Camera":
                xform.SetRotate(self._Gf.Vec3f(0.0, 65.0, yaw), self._UsdGeom.XformCommonAPI.RotationOrderXYZ)
            else:
                xform.SetRotate(self._Gf.Vec3f(0.0, 0.0, yaw), self._UsdGeom.XformCommonAPI.RotationOrderXYZ)

    def _event_material_name(self, kind: str) -> str:
        if kind == "accident":
            return "event_accident"
        if kind in {"blocked", "blockage", "barrier"}:
            return "event_blockage"
        return "event_fire"

    def _ensure_parent(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        current = ""
        for part in parts[:-1]:
            current = f"{current}/{part}"
            if not self.stage.GetPrimAtPath(current):
                self._UsdGeom.Xform.Define(self.stage, current)
