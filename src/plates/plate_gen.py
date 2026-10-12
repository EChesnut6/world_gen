from enum import Enum
import numpy as np
from src.core.noise import SimplexNoise3D
from src.core.sphere_mesh import SphereMesh

DEFAULT_PLANET_RADIUS_KM = 6371.0
CM_PER_YEAR_TO_KM_PER_MYR = 10.0
DEFAULT_PLATE_SPEED_RANGE_CM_PER_YEAR = (0.0, 10.0)


class BoundaryType(Enum):
    CONVERGENT = "Convergent"   # Red: Compression / Collision / Subduction
    DIVERGENT = "Divergent"     # Blue: Extension / Rifting / Ridge
    TRANSFORM = "Transform"     # Yellow: Shear / Strike-Slip


class TectonicPlate:
    """Represents an individual tectonic plate with crustal properties and velocity vector."""

    def __init__(self, plate_id: int, center_vertex: int, is_oceanic: bool = True, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.id = plate_id
        self.center_vertex = center_vertex
        self.color = (128, 128, 128)  # Assigned from crust type and density
        self.vertices: set[int] = set()
        self.faces: set[int] = set()

        # 3D position of center & growth properties
        self.center_pos = np.zeros(3)
        self.drift_vector = np.zeros(3)
        # Rigid-plate Euler vector: its axis is the rotation pole and its
        # magnitude is angular speed. Local velocity is omega cross position.
        self.angular_velocity = np.zeros(3)
        self.surface_speed_cm_per_year = 0.0
        self.motion_initialized = False
        self.growth_weight = float(rng.uniform(0.75, 1.30))

        # Crustal properties
        self.is_oceanic = is_oceanic
        self.is_microplate = False

        if is_oceanic:
            self.crust_thickness = float(rng.uniform(6.0, 10.0))
            self.density = float(rng.uniform(2.9, 3.1))
            self.base_elevation = -0.05
        else:
            self.crust_thickness = float(rng.uniform(35.0, 65.0))
            self.density = float(rng.uniform(2.6, 2.8))
            self.base_elevation = 0.08

    def velocity_at(self, position: np.ndarray) -> np.ndarray:
        """Return this plate's instantaneous surface velocity at a world point."""
        return np.cross(self.angular_velocity, position)

    def set_motion_from_velocity(self, center: np.ndarray, velocity: np.ndarray):
        """Set an Euler rotation that produces the requested tangent velocity at center."""
        radius_sq = float(np.dot(center, center))
        if radius_sq <= 1e-12:
            self.angular_velocity = np.zeros(3)
            self.drift_vector = np.zeros(3)
            self.motion_initialized = True
            return
        tangent_velocity = velocity - np.dot(velocity, center / np.sqrt(radius_sq)) * center / np.sqrt(radius_sq)
        self.angular_velocity = np.cross(center, tangent_velocity) / radius_sq
        self.drift_vector = self.velocity_at(center)
        self.surface_speed_cm_per_year = (
            float(np.linalg.norm(self.angular_velocity)) * DEFAULT_PLANET_RADIUS_KM
            / CM_PER_YEAR_TO_KM_PER_MYR
        )
        self.motion_initialized = True

    def set_motion_from_speed(
        self, center: np.ndarray, direction: np.ndarray, speed_cm_per_year: float,
        planet_radius_km: float = DEFAULT_PLANET_RADIUS_KM,
    ):
        """Set motion using physical speed; Euler rates are radians per Myr."""
        if planet_radius_km <= 0 or speed_cm_per_year < 0:
            raise ValueError("planet radius must be positive and speed nonnegative")
        center_norm = float(np.linalg.norm(center))
        if center_norm <= 1e-12:
            self.angular_velocity = np.zeros(3)
            self.drift_vector = np.zeros(3)
            self.surface_speed_cm_per_year = 0.0
            self.motion_initialized = True
            return
        center_unit = center / center_norm
        tangent = np.asarray(direction, dtype=np.float64)
        tangent -= np.dot(tangent, center_unit) * center_unit
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            raise ValueError("direction must have a nonzero tangent component")
        self.surface_speed_cm_per_year = float(speed_cm_per_year)
        angular_speed = speed_cm_per_year * CM_PER_YEAR_TO_KM_PER_MYR / planet_radius_km
        self.angular_velocity = np.cross(center_unit, tangent / tangent_norm) * angular_speed
        # Keep the rendered arrow in a useful scale; preserve physical speed separately.
        self.drift_vector = np.cross(self.angular_velocity, center_unit)
        self.motion_initialized = True


class BoundarySegment:
    """Represents a boundary edge between two different tectonic plates."""

    def __init__(
        self,
        edge: tuple[int, int],
        plate_a: int,
        plate_b: int,
        position: np.ndarray,
        boundary_type: BoundaryType,
        stress: float,
        normal_velocity: float = 0.0,
        shear_velocity: float = 0.0,
        weakness: float = 0.0,
    ):
        self.edge = edge
        self.plate_a = plate_a
        self.plate_b = plate_b
        self.position = position
        self.boundary_type = boundary_type
        self.stress = stress
        self.normal_velocity = normal_velocity
        self.shear_velocity = shear_velocity
        self.weakness = weakness


def _generate_tangent_vector(center_unit: np.ndarray, rng=None) -> np.ndarray:
    """Generates a guaranteed unit vector tangent to the sphere at center_unit."""
    rand_vec = (rng if rng is not None else np.random.default_rng()).standard_normal(3)
    tangent = rand_vec - np.dot(rand_vec, center_unit) * center_unit
    norm = np.linalg.norm(tangent)
    if norm < 1e-4:
        aux = np.array([0.0, 1.0, 0.0]) if abs(center_unit[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        tangent = np.cross(center_unit, aux)
        norm = np.linalg.norm(tangent)
    return tangent / norm


OCEANIC_DENSITY_RANGE = (2.9, 3.1)
CONTINENTAL_DENSITY_RANGE = (2.6, 2.8)
OCEANIC_LOW_DENSITY_COLOR = np.array([35.0, 175.0, 160.0])
OCEANIC_HIGH_DENSITY_COLOR = np.array([112.0, 55.0, 190.0])
CONTINENTAL_LOW_DENSITY_COLOR = np.array([255.0, 174.0, 76.0])
CONTINENTAL_HIGH_DENSITY_COLOR = np.array([132.0, 24.0, 38.0])


class PlateGenerator:
    """
    Generates tectonic plates over a spherical mesh.
    Uses multi-octave 3D domain warping & weighted metric growth for natural organic plate boundaries.
    Colors plates by crust type and density; lithosphere height modulates brightness.
    Analyzes boundary stress, collision types, and realistic microplate fracturing.
    """

    def __init__(
        self,
        mesh: SphereMesh,
        base_num_plates: int = 16,
        oceanic_ratio: float = 0.70,
        warp_strength: float = 0.16,
        warp_scale: float = 2.2,
        max_microplates: int = 6,
        noise_seed: int = 0,
        planet_radius_km: float = DEFAULT_PLANET_RADIUS_KM,
        plate_speed_range_cm_per_year: tuple[float, float] = DEFAULT_PLATE_SPEED_RANGE_CM_PER_YEAR,
    ):
        self.mesh = mesh
        self.base_num_plates = base_num_plates
        self.num_plates = base_num_plates
        self.oceanic_ratio = oceanic_ratio
        self.warp_strength = warp_strength
        self.warp_scale = warp_scale
        self.max_microplates = max_microplates
        # Fraction of each continental plate's relief field above sea level.
        self.continental_land_percent = 50
        self.oceanic_land_height_cap = 0.10  # Fraction of planetary radius above sea level.
        self.noise_seed = noise_seed
        self.planet_radius_km = float(planet_radius_km)
        self.plate_speed_range_cm_per_year = tuple(map(float, plate_speed_range_cm_per_year))
        if self.planet_radius_km <= 0:
            raise ValueError("planet_radius_km must be positive")
        low_speed, high_speed = self.plate_speed_range_cm_per_year
        if low_speed < 0 or high_speed < low_speed:
            raise ValueError("plate speed range must satisfy 0 <= minimum <= maximum")
        self.rng = np.random.default_rng(noise_seed)
        self.noise_generator = SimplexNoise3D(seed=noise_seed)
        self.lithosphere_height = np.zeros(len(mesh.vertices), dtype=np.float64)
        self.weakness_map = np.zeros(len(mesh.vertices), dtype=np.float64)
        self._generate_lithosphere()
        self.terrain_face_heights = np.zeros(len(mesh.faces), dtype=np.float64)
        self.terrain_height_vertices = np.zeros(len(mesh.vertices), dtype=np.float64)
        self.land_face_mask = np.zeros(len(mesh.faces), dtype=bool)
        self.continent_height_map = np.zeros(len(mesh.faces), dtype=np.float64)
        self.hotspot_tracks: list[dict] = []
        self.hotspots_initialized = False

        self.plates: list[TectonicPlate] = []
        self._base_plate_state: list[TectonicPlate] = []
        self.vertex_plate_map = np.full(len(mesh.vertices), -1, dtype=int)
        self.face_plate_map = np.full(len(mesh.faces), -1, dtype=int)
        self.boundaries: list[BoundarySegment] = []
        self.vertex_stress = np.zeros(len(mesh.vertices), dtype=float)

        # Vectorized boundary arrays for ultra-fast rendering
        self.boundary_edges_arr = np.zeros((0, 2), dtype=int)
        self.boundary_positions_arr = np.zeros((0, 3), dtype=np.float64)
        self.boundary_types_arr = np.zeros(0, dtype=int)  # 0: Convergent, 1: Divergent, 2: Transform
        self.plate_adj_graph: dict[int, set[int]] = {}

    def _generate_lithosphere(self):
        """Generate reusable normalized height and weakness fields on mesh vertices."""
        unit_positions = self.mesh.vertices / self.mesh.radius
        raw_height = self.noise_generator.fractal(
            unit_positions,
            scale=2.4,
            octaves=5,
            persistence=0.52,
            lacunarity=2.0,
        )
        height_min = float(raw_height.min())
        height_range = float(raw_height.max() - height_min)
        if height_range > 1e-12:
            self.lithosphere_height = (raw_height - height_min) / height_range
        else:
            self.lithosphere_height = np.full(len(raw_height), 0.5, dtype=np.float64)
        # Low lithosphere regions are treated as relatively weak in this first model.
        self.weakness_map = 1.0 - self.lithosphere_height

    def regenerate_noise(self, seed: int | None = None):
        """Regenerate shared procedural fields, optionally changing their seed."""
        if seed is not None:
            self.noise_seed = int(seed)
            self.noise_generator = SimplexNoise3D(seed=self.noise_seed)
        self._generate_lithosphere()
        if self.plates:
            self._generate_surface_terrain()

    def _generate_surface_terrain(self):
        """Build continental land and oceanic hotspot island relief on the face mesh."""
        face_count = len(self.mesh.faces)
        radius = self.mesh.radius
        self.terrain_face_heights = np.zeros(face_count, dtype=np.float64)
        self.land_face_mask = np.zeros(face_count, dtype=bool)
        self.continent_height_map = np.zeros(face_count, dtype=np.float64)
        previous_hotspot_tracks = self.hotspot_tracks
        reuse_hotspots = self.hotspots_initialized
        self.hotspot_tracks = list(previous_hotspot_tracks) if reuse_hotspots else []
        if not self.plates or len(self.face_plate_map) != face_count:
            self.terrain_height_vertices = np.zeros(len(self.mesh.vertices), dtype=np.float64)
            return

        centers = self.mesh.face_centers
        center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
        face_units = centers / np.maximum(center_norms, 1e-12)
        rng = np.random.default_rng(self.noise_seed + 101)
        previous_hotspot_owners = []
        if reuse_hotspots:
            for chain in previous_hotspot_tracks:
                points = chain["points"]
                owners = np.argmax(face_units @ points.T, axis=0)
                previous_hotspot_owners.append((chain, self.face_plate_map[owners]))

        for plate in self.plates:
            face_ids = np.flatnonzero(self.face_plate_map == plate.id)
            if not len(face_ids):
                continue

            if not plate.is_oceanic:
                plate_noise = SimplexNoise3D(self.noise_seed + 104729 * (plate.id + 1))
                # Give each continental plate one broad landmass first, then
                # perturb its coastline with low-frequency noise. The previous
                # high-frequency fBm field made isolated speckles regardless
                # of the selected land percentage.
                continent_center = plate.center_pos / max(np.linalg.norm(plate.center_pos), 1e-12)
                if np.linalg.norm(plate.center_pos) < 1e-12:
                    continent_center = face_units[face_ids].mean(axis=0)
                    continent_center /= max(np.linalg.norm(continent_center), 1e-12)
                center_alignment = np.clip(face_units[face_ids] @ continent_center, -1.0, 1.0)
                angular_distance = np.arccos(center_alignment)
                coastline_noise = plate_noise.fractal(
                    face_units[face_ids], scale=2.2, octaves=3, persistence=0.38, lacunarity=2.0
                )
                # Noise moves the coast by roughly a few degrees while the
                # angular core keeps the land connected around the plate center.
                raw = -angular_distance + 0.24 * coastline_noise
                value_range = float(raw.max() - raw.min())
                normalized = (raw - raw.min()) / value_range if value_range > 1e-12 else np.full(len(raw), 0.5)
                land_fraction = self.continental_land_percent / 100.0
                if land_fraction <= 0.0:
                    threshold = float(normalized.max()) + 1e-9
                elif land_fraction >= 1.0:
                    threshold = float(normalized.min()) - 1e-9
                else:
                    threshold = float(np.quantile(normalized, 1.0 - land_fraction))
                self.continent_height_map[face_ids] = normalized
                self.land_face_mask[face_ids] = normalized > threshold
                # Sea level is zero; the selected upper fraction of this field rises above it.
                self.terrain_face_heights[face_ids] = (normalized - threshold) * (0.16 * radius)
                continue

            # Oceanic crust sits below sea level except for volcanic edifices.
            self.terrain_face_heights[face_ids] = -0.025 * radius
            if reuse_hotspots:
                plate_face_units = face_units[face_ids]
                for chain, owners in previous_hotspot_owners:
                    keep = owners == plate.id
                    if not np.any(keep):
                        continue
                    points = chain["points"][keep]
                    heights = chain["heights"][keep]
                    widths = chain["widths"][keep]
                    distances = np.arccos(np.clip(plate_face_units @ points.T, -1.0, 1.0))
                    relief = np.zeros(len(face_ids), dtype=np.float64)
                    for idx, (height, width) in enumerate(zip(heights, widths)):
                        mound = height * np.maximum(0.0, 1.0 - (distances[:, idx] / width) ** 2) ** 1.5
                        # Overlapping volcanic slopes build a shared island shelf.
                        relief += mound
                    self.terrain_face_heights[face_ids] += relief
                self.land_face_mask[face_ids] = self.terrain_face_heights[face_ids] > 0.0
                continue
            if len(face_ids) < 8 or rng.random() > 0.38:
                continue

            axis_norm = np.linalg.norm(plate.angular_velocity)
            if axis_norm < 1e-10:
                continue
            axis = plate.angular_velocity / axis_norm
            plate_center = plate.center_pos / max(np.linalg.norm(plate.center_pos), 1e-12)
            plate_scale = np.sqrt(4.0 * np.pi * len(face_ids) / face_count)
            island_count = int(rng.choice([2, 3, 3, 4]))
            desired_span = min(0.62, max(0.30, 1.05 * plate_scale))
            selected_track = None

            # Choose a hotspot whose whole island chain remains on this oceanic plate.
            candidate_faces = rng.permutation(face_ids)[:min(80, len(face_ids))]
            # Try the desired chain length first, then shorten it for smaller or
            # awkwardly shaped plates. Uneven intervals avoid the evenly spaced,
            # manufactured look of a regular row of identical volcanoes.
            for span_scale in (1.0, 0.8, 0.6):
                gaps = rng.uniform(0.72, 1.28, island_count - 1)
                gaps *= desired_span * span_scale / gaps.sum()
                angles = np.concatenate(([0.0], np.cumsum(gaps)))
                for anchor_face in candidate_faces:
                    anchor = face_units[anchor_face]
                    # Prefer interior locations so the generated chain does not cross a boundary.
                    if np.dot(anchor, plate_center) < 0.55:
                        continue
                    track = []
                    stays_on_plate = True
                    for angle in angles:
                        point = (
                            anchor * np.cos(angle)
                            + np.cross(axis, anchor) * np.sin(angle)
                            + axis * np.dot(axis, anchor) * (1.0 - np.cos(angle))
                        )
                        point /= max(np.linalg.norm(point), 1e-12)
                        nearest_face = int(np.argmax(face_units @ point))
                        if self.face_plate_map[nearest_face] != plate.id:
                            stays_on_plate = False
                            break
                        track.append(point)
                    if stays_on_plate:
                        selected_track = np.asarray(track)
                        break
                if selected_track is not None:
                    break

            if selected_track is None:
                continue

            island_heights = []
            island_widths = []
            for island_index in range(island_count):
                age = island_index / max(island_count - 1, 1)
                erosion = np.exp(-1.35 * age)
                height = 0.025 + 0.105 * erosion * rng.uniform(0.62, 1.28)
                if island_index > 0 and rng.random() < 0.28:
                    height = rng.uniform(0.012, 0.023)  # drowned seamount
                island_heights.append(height)
                island_widths.append(rng.uniform(0.038, 0.066) * (0.88 + 0.24 * erosion))
            self.hotspot_tracks.append({
                "plate_id": plate.id,
                "points": selected_track.copy(),
                "heights": np.asarray(island_heights),
                "widths": np.asarray(island_widths),
            })
            plate_face_units = face_units[face_ids]
            angular_distances = np.arccos(np.clip(plate_face_units @ selected_track.T, -1.0, 1.0))
            # A plume builds a large, young shield volcano near the hotspot;
            # older volcanoes are variably eroded and some persist only as
            # submerged seamounts. Independent widths and heights keep islands
            # from reading as copies of one another.
            island_relief = np.zeros(len(face_ids), dtype=np.float64)
            for island_index, (height, width) in enumerate(zip(island_heights, island_widths)):
                distance = angular_distances[:, island_index]
                mound = height * np.maximum(0.0, 1.0 - (distance / width) ** 2) ** 1.5
                # Let neighboring volcanoes coalesce when their broad bases overlap.
                island_relief += mound
            self.terrain_face_heights[face_ids] += island_relief
            self.land_face_mask[face_ids] = self.terrain_face_heights[face_ids] > 0.0

        self.hotspots_initialized = True

        # Ocean-ocean convergence can create volcanic island arcs. These sit on
        # the overriding plate, offset inland from the trench, and only become
        # islands when the local oceanic crust rises above sea level.
        oceanic_arc_groups: dict[tuple[int, int], list[BoundarySegment]] = {}
        for boundary in self.boundaries:
            if boundary.boundary_type != BoundaryType.CONVERGENT:
                continue
            a, b = self.plates[boundary.plate_a], self.plates[boundary.plate_b]
            if a.is_oceanic and b.is_oceanic:
                oceanic_arc_groups.setdefault(tuple(sorted((a.id, b.id))), []).append(boundary)
        for (plate_a_id, plate_b_id), segments in oceanic_arc_groups.items():
            a, b = self.plates[plate_a_id], self.plates[plate_b_id]
            override_votes = {plate_a_id: 0, plate_b_id: 0}
            for segment in segments:
                subducting_id = segment.plate_a if segment.normal_velocity > 0.0 else segment.plate_b
                override_votes[plate_b_id if subducting_id == plate_a_id else plate_a_id] += 1
            overriding = self.plates[max(override_votes, key=override_votes.get)]
            positions = np.asarray([segment.position for segment in segments])
            units = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12)
            center = units.mean(axis=0)
            center /= max(np.linalg.norm(center), 1e-12)
            ref = np.array([0.0, 1.0, 0.0]) if abs(center[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            tangent_x = np.cross(center, ref)
            tangent_x /= max(np.linalg.norm(tangent_x), 1e-12)
            tangent_y = np.cross(center, tangent_x)
            order = np.argsort(np.arctan2(units @ tangent_y, units @ tangent_x))
            units = units[order]
            count = min(9, max(3, len(units) // 12))
            sample_ids = np.linspace(0, len(units) - 1, count, dtype=int)
            arc_points = []
            for unit in units[sample_ids]:
                toward = overriding.center_pos / max(np.linalg.norm(overriding.center_pos), 1e-12)
                toward -= np.dot(toward, unit) * unit
                toward_norm = np.linalg.norm(toward)
                if toward_norm < 1e-8:
                    continue
                toward /= toward_norm
                point = unit * np.cos(0.13) + toward * np.sin(0.13)
                point /= max(np.linalg.norm(point), 1e-12)
                if self.face_plate_map[int(np.argmax(face_units @ point))] == overriding.id:
                    arc_points.append(point)
            if len(arc_points) < 2:
                continue
            arc_points = np.asarray(arc_points)
            arc_units = face_units[np.flatnonzero(self.face_plate_map == overriding.id)]
            arc_face_ids = np.flatnonzero(self.face_plate_map == overriding.id)
            distances = np.arccos(np.clip(arc_units @ arc_points.T, -1.0, 1.0))
            arc_relief = np.zeros(len(arc_face_ids), dtype=np.float64)
            for idx in range(len(arc_points)):
                height = 0.07 * rng.uniform(0.65, 1.35)
                width = rng.uniform(0.035, 0.055)
                mound = height * np.maximum(0.0, 1.0 - (distances[:, idx] / width) ** 2) ** 1.5
                arc_relief += mound
            self.terrain_face_heights[arc_face_ids] = np.maximum(
                self.terrain_face_heights[arc_face_ids], -0.025 * radius + arc_relief
            )
            self.land_face_mask[arc_face_ids] = self.terrain_face_heights[arc_face_ids] > 0.0

        # Keep stacked hotspot and island-arc relief from creating excessively
        # tall oceanic peaks while leaving the underwater bathymetry unchanged.
        for plate in self.plates:
            if plate.is_oceanic:
                face_ids = np.flatnonzero(self.face_plate_map == plate.id)
                self.terrain_face_heights[face_ids] = np.minimum(
                    self.terrain_face_heights[face_ids], self.oceanic_land_height_cap * radius
                )
                self.land_face_mask[face_ids] = self.terrain_face_heights[face_ids] > 0.0

        # Average face relief onto vertices for a continuous displaced render mesh.
        vertex_totals = np.zeros(len(self.mesh.vertices), dtype=np.float64)
        vertex_counts = np.zeros(len(self.mesh.vertices), dtype=np.int32)
        flat_vertices = self.mesh.faces.reshape(-1)
        np.add.at(vertex_totals, flat_vertices, np.repeat(self.terrain_face_heights, 3))
        np.add.at(vertex_counts, flat_vertices, 1)
        self.terrain_height_vertices = vertex_totals / np.maximum(vertex_counts, 1)

    def set_continental_land_percent(self, percent: int):
        """Set the share of continental terrain above sea level and rebuild terrain."""
        self.continental_land_percent = max(0, min(100, int(percent)))
        if self.plates:
            self._generate_surface_terrain()

    def _apply_domain_warping(self, vertices: np.ndarray, warp_scale: float | None = None, warp_strength: float | None = None) -> np.ndarray:
        """
        Applies multi-frequency 3D sinusoidal domain warping to vertex positions.
        Replaces rigid flat Voronoi edges with natural organic coastlines & wavy fault lines.
        """
        scale = self.warp_scale if warp_scale is None else warp_scale
        strength = self.warp_strength if warp_strength is None else warp_strength

        norm_verts = vertices / self.mesh.radius
        x, y, z = norm_verts[:, 0] * scale, norm_verts[:, 1] * scale, norm_verts[:, 2] * scale

        # Octave 1
        wx = np.sin(y * 1.7 + 0.5) + np.cos(z * 2.3 + 1.2)
        wy = np.sin(z * 1.9 + 2.1) + np.cos(x * 1.5 + 0.7)
        wz = np.sin(x * 2.1 + 1.4) + np.cos(y * 1.8 + 2.5)

        # Octave 2
        wx += 0.5 * (np.sin(z * 4.3 + 3.1) + np.cos(x * 3.7 + 0.4))
        wy += 0.5 * (np.sin(x * 4.1 + 1.8) + np.cos(y * 4.5 + 2.2))
        wz += 0.5 * (np.sin(y * 3.9 + 2.7) + np.cos(z * 4.1 + 1.1))

        warp_vecs = np.column_stack([wx, wy, wz])
        warped = norm_verts + strength * warp_vecs
        norms = np.linalg.norm(warped, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (warped / norms) * self.mesh.radius

    def _assign_plate_colors(self):
        """Map plate density to its crust-type color gradient."""
        if not self.plates:
            return

        adj = {p.id: set() for p in self.plates}
        for (f1, f2), (p1, p2) in self._get_boundary_face_pairs():
            if p1 != p2 and p1 in adj and p2 in adj:
                adj[p1].add(p2)
                adj[p2].add(p1)

        self.plate_adj_graph = adj

        for p in self.plates:
            low, high = OCEANIC_DENSITY_RANGE if p.is_oceanic else CONTINENTAL_DENSITY_RANGE
            t = float(np.clip((p.density - low) / (high - low), 0.0, 1.0))
            if p.is_oceanic:
                rgb = OCEANIC_LOW_DENSITY_COLOR * (1.0 - t) + OCEANIC_HIGH_DENSITY_COLOR * t
            else:
                rgb = CONTINENTAL_LOW_DENSITY_COLOR * (1.0 - t) + CONTINENTAL_HIGH_DENSITY_COLOR * t
            p.color = tuple(np.rint(rgb).astype(np.uint8).tolist())

    def _get_boundary_face_pairs(self):
        """Helper to yield face index pairs and their corresponding plate IDs."""
        pairs = []
        for edge, f_indices in self.mesh.edge_to_faces.items():
            if len(f_indices) == 2:
                f1, f2 = f_indices
                p1, p2 = self.face_plate_map[f1], self.face_plate_map[f2]
                if p1 != p2 and p1 != -1 and p2 != -1:
                    pairs.append(((f1, f2), (p1, p2)))
        return pairs

    def set_mesh(self, new_mesh: SphereMesh):
        """Updates mesh reference and recreates buffers for mesh resolution changes."""
        self.mesh = new_mesh
        self.vertex_plate_map = np.full(len(new_mesh.vertices), -1, dtype=int)
        self.face_plate_map = np.full(len(new_mesh.faces), -1, dtype=int)
        self.vertex_stress = np.zeros(len(new_mesh.vertices), dtype=float)
        self._generate_lithosphere()
        self.terrain_face_heights = np.zeros(len(new_mesh.faces), dtype=np.float64)
        self.terrain_height_vertices = np.zeros(len(new_mesh.vertices), dtype=np.float64)
        self.land_face_mask = np.zeros(len(new_mesh.faces), dtype=bool)
        self.continent_height_map = np.zeros(len(new_mesh.faces), dtype=np.float64)
        self.hotspot_tracks = []
        self.hotspots_initialized = False
        self.generate()

    def update_warp(self, warp_strength: float):
        """Live updates domain warping strength and repartitions boundaries."""
        self.warp_strength = warp_strength
        if self.plates:
            self._repartition_mesh()

    def generate(self, base_num_plates: int | None = None, oceanic_ratio: float | None = None):
        """Generates new clean base tectonic plates, resetting all microplate fractures."""
        self.hotspot_tracks = []
        self.hotspots_initialized = False
        if base_num_plates is not None:
            self.base_num_plates = base_num_plates
        if oceanic_ratio is not None:
            self.oceanic_ratio = oceanic_ratio

        self.num_plates = self.base_num_plates
        num_verts = len(self.mesh.vertices)

        # 1. Seed centers for base plates
        self.rng = np.random.default_rng(self.noise_seed)
        seed_indices = self.rng.choice(num_verts, size=self.num_plates, replace=False)
        self.plates = []
        oceanic_flags = self.rng.random(self.num_plates) < self.oceanic_ratio

        for idx, seed_idx in enumerate(seed_indices):
            plate = TectonicPlate(
                plate_id=idx,
                center_vertex=seed_idx,
                is_oceanic=bool(oceanic_flags[idx]),
                rng=self.rng,
            )
            plate.vertices.add(seed_idx)
            self.plates.append(plate)
            self.vertex_plate_map[seed_idx] = idx

        self._repartition_mesh()
        import copy
        self._base_plate_state = copy.deepcopy(self.plates)

    def reset_fractures(self):
        """Restores the generated base plates after splits or rare fusions."""
        if self._base_plate_state and (
            len(self.plates) != len(self._base_plate_state)
            or any(p.is_microplate for p in self.plates)
        ):
            import copy
            self.plates = copy.deepcopy(self._base_plate_state)
            self.num_plates = len(self.plates)
            self._repartition_mesh()

    def snapshot_fracture_state(self):
        """Capture mutable simulation state needed to undo one fracture."""
        import copy

        fields = (
            "plates", "num_plates", "vertex_plate_map", "face_plate_map", "vertex_stress",
            "boundaries", "boundary_edges_arr", "boundary_positions_arr", "boundary_types_arr",
            "plate_adj_graph", "terrain_face_heights", "terrain_height_vertices", "land_face_mask",
            "continent_height_map", "hotspot_tracks", "hotspots_initialized",
        )
        state = {}
        for field in fields:
            value = getattr(self, field)
            if field in ("plates", "boundaries", "plate_adj_graph", "hotspot_tracks"):
                state[field] = copy.deepcopy(value)
            elif isinstance(value, np.ndarray):
                state[field] = value.copy()
            else:
                state[field] = value
        return state

    def restore_fracture_state(self, state):
        """Restore a snapshot created immediately before a fracture."""
        for field, value in state.items():
            setattr(self, field, value)

    def _repartition_mesh(self):
        """Repartition mesh vertices and faces, then assign density-based plate colors."""
        num_plates = len(self.plates)
        all_seeds = [p.center_vertex for p in self.plates]
        seed_coords = self.mesh.vertices[all_seeds] / self.mesh.radius
        weights = np.array([p.growth_weight for p in self.plates], dtype=np.float64)

        # 1. Domain-warped vertex positions
        warped_verts = self._apply_domain_warping(self.mesh.vertices) / self.mesh.radius

        # 2. Vectorized dot products: (V, 3) @ (3, P) -> (V, P)
        dots = warped_verts @ seed_coords.T
        scores = dots * weights[np.newaxis, :]

        nearest_plates = np.argmax(scores, axis=1)
        self.vertex_plate_map = nearest_plates

        # 3. Vectorized face plate mapping (majority vote per face)
        face_v_plates = self.vertex_plate_map[self.mesh.faces]  # Shape: (F, 3)
        p0, p1, p2 = face_v_plates[:, 0], face_v_plates[:, 1], face_v_plates[:, 2]
        majority_plates = np.where((p0 == p1) | (p0 == p2), p0, p1)
        self.face_plate_map = majority_plates

        # 4. Update plate center positions & drift vectors
        for p_idx, plate in enumerate(self.plates):
            mask = (nearest_plates == p_idx)
            if not np.any(mask):
                continue
            verts = self.mesh.vertices[mask]
            avg_center = np.mean(verts, axis=0)
            norm = np.linalg.norm(avg_center)
            center_unit = avg_center / norm if norm > 0 else np.array([0.0, 1.0, 0.0])
            plate.center_pos = center_unit * self.mesh.radius

            if not plate.motion_initialized:
                direction = _generate_tangent_vector(center_unit, self.rng)
                speed = self.rng.uniform(*self.plate_speed_range_cm_per_year)
                plate.set_motion_from_speed(
                    plate.center_pos, direction, speed, self.planet_radius_km
                )
            else:
                plate.drift_vector = plate.velocity_at(plate.center_pos)

        # 5. Analyze boundaries & assign plate colors
        self._analyze_boundaries()
        self._assign_plate_colors()
        self._generate_surface_terrain()

    def _analyze_boundaries(self):
        """Analyze boundary interactions (Convergent, Divergent, Transform) & calculate stress vectorially."""
        self.boundaries = []
        self.vertex_stress.fill(0.0)

        boundary_edges = []
        edge_centers = []
        edge_dirs = []
        p1_list = []
        p2_list = []

        for edge, f_indices in self.mesh.edge_to_faces.items():
            if len(f_indices) != 2:
                continue

            f1, f2 = f_indices
            p1 = self.face_plate_map[f1]
            p2 = self.face_plate_map[f2]

            if p1 != p2 and p1 != -1 and p2 != -1:
                boundary_edges.append(edge)
                v1_pos = self.mesh.vertices[edge[0]]
                v2_pos = self.mesh.vertices[edge[1]]
                edge_center = (v1_pos + v2_pos) * 0.5
                edge_dir = v2_pos - v1_pos
                edge_centers.append(edge_center)
                edge_dirs.append(edge_dir)
                p1_list.append(p1)
                p2_list.append(p2)

        if not boundary_edges:
            self.boundary_edges_arr = np.zeros((0, 2), dtype=int)
            self.boundary_positions_arr = np.zeros((0, 3), dtype=np.float64)
            self.boundary_types_arr = np.zeros(0, dtype=int)
            return

        centers_arr = np.array(edge_centers)
        dirs_arr = np.array(edge_dirs)
        p1_arr = np.array(p1_list)
        p2_arr = np.array(p2_list)

        dir_norms = np.linalg.norm(dirs_arr, axis=1, keepdims=True)
        dir_norms[dir_norms == 0] = 1.0
        dirs_normalized = dirs_arr / dir_norms

        center_norms = np.linalg.norm(centers_arr, axis=1, keepdims=True)
        center_norms[center_norms == 0] = 1.0
        r_units = centers_arr / center_norms

        normals = np.cross(dirs_normalized, r_units)
        normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normal_norms[normal_norms == 0] = 1.0
        normals /= normal_norms

        c1 = np.array([self.plates[p].center_pos for p in p1_arr])
        c2 = np.array([self.plates[p].center_pos for p in p2_arr])
        center_diffs = c2 - c1

        dot_orient = np.einsum('ij,ij->i', normals, center_diffs)
        normals[dot_orient < 0] *= -1.0

        v1_drift = np.array([
            self.plates[p].velocity_at(center) for p, center in zip(p1_arr, centers_arr)
        ])
        v2_drift = np.array([
            self.plates[p].velocity_at(center) for p, center in zip(p2_arr, centers_arr)
        ])
        v_rel = v1_drift - v2_drift

        normal_stresses = np.einsum('ij,ij->i', v_rel, normals)
        shear_velocities = np.linalg.norm(v_rel - normal_stresses[:, None] * normals, axis=1)

        boundary_types_arr = np.zeros(len(boundary_edges), dtype=int)

        for i, edge in enumerate(boundary_edges):
            ns = normal_stresses[i]
            st = float(np.hypot(max(normal_stresses[i], 0.0), 0.35 * shear_velocities[i]))

            # Normal velocity is radians/Myr on the unit sphere. This deadband
            # corresponds to roughly 0.32 cm/year at Earth's radius.
            if ns > 0.0005:
                b_type = BoundaryType.CONVERGENT
                b_type_id = 0
            elif ns < -0.0005:
                b_type = BoundaryType.DIVERGENT
                b_type_id = 1
            else:
                b_type = BoundaryType.TRANSFORM
                b_type_id = 2

            boundary_types_arr[i] = b_type_id
            boundary_weakness = float(np.mean(self.weakness_map[list(edge)]))

            segment = BoundarySegment(
                edge=edge,
                plate_a=int(p1_arr[i]),
                plate_b=int(p2_arr[i]),
                position=centers_arr[i],
                boundary_type=b_type,
                stress=st,
                normal_velocity=float(normal_stresses[i]),
                shear_velocity=float(shear_velocities[i]),
                weakness=boundary_weakness,
            )
            self.boundaries.append(segment)
            self.vertex_stress[edge[0]] += st
            self.vertex_stress[edge[1]] += st

        self.boundary_edges_arr = np.array(boundary_edges, dtype=int)
        self.boundary_positions_arr = centers_arr
        self.boundary_types_arr = boundary_types_arr

    def shatter_high_stress_zones(self, spawn_count: int = 1):
        """Detach a connected plate patch at a high-stress convergent boundary."""
        current_micro_count = sum(1 for p in self.plates if p.is_microplate)
        if current_micro_count >= self.max_microplates:
            print(f"Microplate limit reached ({self.max_microplates} max). Reset or generate new plates.")
            return

        convergent = sorted(
            (b for b in self.boundaries if b.boundary_type == BoundaryType.CONVERGENT),
            key=lambda boundary: boundary.stress * (0.5 + boundary.weakness),
            reverse=True,
        )
        if not convergent:
            return

        # Rare continent-continent convergence can amalgamate two plate identities.
        continental_collisions = [
            b for b in convergent
            if not self.plates[b.plate_a].is_oceanic and not self.plates[b.plate_b].is_oceanic
        ]
        if continental_collisions and np.random.random() < 0.05:
            boundary = continental_collisions[0]
            self._fuse_plates(boundary.plate_a, boundary.plate_b)
            return

        face_counts = np.bincount(self.face_plate_map, minlength=len(self.plates))
        for boundary in convergent:
            adjacent_faces = self.mesh.edge_to_faces.get(boundary.edge, ())
            if len(adjacent_faces) != 2:
                continue

            parent_ids = sorted(
                (boundary.plate_a, boundary.plate_b),
                key=lambda plate_id: face_counts[plate_id],
                reverse=True,
            )
            for parent_id in parent_ids:
                parent_faces = int(face_counts[parent_id])
                if parent_faces < 4:
                    continue
                seed_face = next(
                    (face for face in adjacent_faces if self.face_plate_map[face] == parent_id),
                    None,
                )
                if seed_face is None:
                    continue

                # Grow a connected patch inward from the stressed boundary. Keep
                # it local and proportional to the source plate's area.
                target_size = max(3, min(round(parent_faces * 0.12), max(4, round(len(self.mesh.faces) * 0.04))))
                patch = {seed_face}
                frontier = [seed_face]
                while frontier and len(patch) < target_size:
                    face = frontier.pop(0)
                    for neighbor, _ in self.mesh.face_neighbors[face]:
                        if neighbor not in patch and self.face_plate_map[neighbor] == parent_id:
                            patch.add(neighbor)
                            frontier.append(neighbor)
                            if len(patch) >= target_size:
                                break

                if len(patch) < 3:
                    continue

                patch_indices = np.fromiter(patch, dtype=int)
                patch_vertices = np.unique(self.mesh.faces[patch_indices])
                patch_center = np.mean(self.mesh.vertices[patch_vertices], axis=0)
                center_vertex = int(patch_vertices[np.argmin(np.linalg.norm(self.mesh.vertices[patch_vertices] - patch_center, axis=1))])

                source = self.plates[parent_id]
                microplate = TectonicPlate(len(self.plates), center_vertex, is_oceanic=source.is_oceanic)
                microplate.is_microplate = True
                microplate.growth_weight = source.growth_weight
                microplate.crust_thickness = source.crust_thickness
                microplate.density = source.density
                microplate.base_elevation = source.base_elevation
                neighbor_id = boundary.plate_b if parent_id == boundary.plate_a else boundary.plate_a
                center_norm = np.linalg.norm(patch_center)
                center_unit = patch_center / center_norm if center_norm > 0 else np.array([0.0, 1.0, 0.0])

                # A fragment mostly keeps its parent's motion, but collision
                # shear and convergence with the neighboring plate can redirect it.
                parent_motion = source.velocity_at(patch_center)
                neighbor_motion = self.plates[neighbor_id].velocity_at(patch_center)
                parent_motion -= np.dot(parent_motion, center_unit) * center_unit
                neighbor_motion -= np.dot(neighbor_motion, center_unit) * center_unit
                relative_boundary_motion = neighbor_motion - parent_motion
                fragment_motion = parent_motion + 0.35 * relative_boundary_motion

                parent_speed = np.linalg.norm(parent_motion)
                fragment_speed = np.linalg.norm(fragment_motion)
                if fragment_speed < 1e-8:
                    fragment_motion = _generate_tangent_vector(center_unit)
                    fragment_speed = 1.0
                target_speed = parent_speed if parent_speed > 1e-8 else fragment_speed
                fragment_motion = fragment_motion / fragment_speed * target_speed
                microplate.set_motion_from_velocity(patch_center, fragment_motion)
                self.plates.append(microplate)
                self.face_plate_map[patch_indices] = microplate.id
                self.num_plates = len(self.plates)
                self._rebuild_plate_membership()
                self._refresh_boundaries_and_colors()
                return

    def _rebuild_plate_membership(self):
        """Rebuild vertex labels and plate membership from authoritative face labels."""
        num_plates = len(self.plates)
        votes = np.zeros((len(self.mesh.vertices), num_plates), dtype=np.uint8)
        for corner in range(3):
            np.add.at(votes, (self.mesh.faces[:, corner], self.face_plate_map), 1)
        self.vertex_plate_map = np.argmax(votes, axis=1)

        for plate in self.plates:
            plate.faces.clear()
            plate.vertices.clear()
        for plate_id, plate in enumerate(self.plates):
            face_indices = np.flatnonzero(self.face_plate_map == plate_id)
            plate.faces.update(face_indices.tolist())
            if len(face_indices):
                vertices = np.unique(self.mesh.faces[face_indices])
                plate.vertices.update(vertices.tolist())
                center = np.mean(self.mesh.vertices[vertices], axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    plate.center_pos = center / norm * self.mesh.radius
                plate.center_vertex = int(vertices[np.argmin(np.linalg.norm(self.mesh.vertices[vertices] - center, axis=1))])
                plate.drift_vector = plate.velocity_at(plate.center_pos)

    def _refresh_boundaries_and_colors(self):
        self._analyze_boundaries()
        self._assign_plate_colors()
        self._generate_surface_terrain()

    def _fuse_plates(self, plate_a_id: int, plate_b_id: int):
        """Rarely merge two continental identities at a convergent boundary."""
        if plate_a_id == plate_b_id:
            return
        counts = np.bincount(self.face_plate_map, minlength=len(self.plates))
        survivor_id, removed_id = sorted((plate_a_id, plate_b_id), key=lambda pid: counts[pid], reverse=True)
        survivor, removed = self.plates[survivor_id], self.plates[removed_id]
        total_area = max(1, counts[survivor_id] + counts[removed_id])
        merged_drift = (
            survivor.drift_vector * counts[survivor_id] + removed.drift_vector * counts[removed_id]
        ) / total_area
        survivor.density = (
            survivor.density * counts[survivor_id] + removed.density * counts[removed_id]
        ) / total_area
        merged_center = survivor.center_pos
        survivor.set_motion_from_velocity(merged_center, merged_drift)
        self.face_plate_map[self.face_plate_map == removed_id] = survivor_id
        self.plates.pop(removed_id)
        for new_id, plate in enumerate(self.plates):
            plate.id = new_id
        self.face_plate_map[self.face_plate_map > removed_id] -= 1
        self.num_plates = len(self.plates)
        self._rebuild_plate_membership()
        self._refresh_boundaries_and_colors()

    def get_face_colors(self) -> np.ndarray:
        """Returns (num_faces, 3) array of RGB colors according to tectonic plate mapping."""
        num_faces = len(self.mesh.faces)
        face_colors = np.zeros((num_faces, 3), dtype=np.uint8)

        max_p_id = max(p.id for p in self.plates) if self.plates else 0
        color_lut = np.zeros((max_p_id + 1, 3), dtype=np.uint8)
        for p in self.plates:
            color_lut[p.id] = p.color

        valid_mask = (self.face_plate_map >= 0) & (self.face_plate_map <= max_p_id)
        face_colors[valid_mask] = color_lut[self.face_plate_map[valid_mask]]
        face_colors[~valid_mask] = [100, 100, 100]

        return face_colors
