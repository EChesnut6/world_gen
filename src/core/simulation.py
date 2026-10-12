"""Deterministic fixed-step kinematics for an already generated planet."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.plates.plate_gen import PlateGenerator


def _rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Return a Rodrigues rotation matrix for an axis-angle vector in radians."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class PlateSimulation:
    """Fixed-step rigid plate motion with inverse remapping on spherical faces.

    The mesh is the fixed destination grid. For each destination face, the
    previous state is sampled at the point obtained by inverse-rotating its
    center through each candidate plate's step rotation. A nearest source face
    lookup transfers ownership and per-face crust/terrain fields. This keeps a
    complete shared grid, at the cost of nearest-neighbor remapping error.
    """

    def __init__(
        self,
        generator: PlateGenerator,
        timestep_myr: float = 1.0,
        playback_steps_per_second: float = 1.0,
    ):
        if timestep_myr <= 0:
            raise ValueError("timestep_myr must be positive")
        if playback_steps_per_second <= 0:
            raise ValueError("playback_steps_per_second must be positive")
        if len(generator.plates) == 0:
            raise ValueError("generate plates before creating a simulation")
        if np.any(generator.face_plate_map < 0):
            raise ValueError("every face must have a plate owner")

        self.generator = generator
        self.timestep_myr = float(timestep_myr)
        self.playback_steps_per_second = float(playback_steps_per_second)
        self.seed = int(generator.noise_seed)
        self.planet_radius_km = float(generator.planet_radius_km)
        self.elapsed_myr = 0.0
        self.step_count = 0
        self.is_playing = False
        self._playback_accumulator = 0.0
        self._face_centers = generator.mesh.face_centers.copy()
        self._face_tree = cKDTree(self._face_centers)
        self._face_areas = self._compute_face_areas()
        self._face_units = self._face_centers / np.maximum(
            np.linalg.norm(self._face_centers, axis=1, keepdims=True), 1e-12
        )
        self._build_face_neighbors()
        self._initial_plate_centers = np.asarray(
            [plate.center_pos.copy() for plate in generator.plates], dtype=np.float64
        )
        self._initial_state = self._capture_generator_state()
        self.hotspot_sources = self._capture_hotspot_sources()
        self._initial_hotspot_sources = [
            {**source, "position": source["position"].copy()}
            for source in self.hotspot_sources
        ]
        self._restore_state(self._initial_state)
        self.last_remap_diagnostics = self._empty_diagnostics()
        self.last_remap_diagnostics["face_count"] = len(self._face_centers)

    def _compute_face_areas(self) -> np.ndarray:
        triangles = self.generator.mesh.vertices[self.generator.mesh.faces]
        return np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ) * 0.5

    def _build_face_neighbors(self) -> None:
        neighbor_lists = self.generator.mesh.face_neighbors
        max_neighbors = max(len(neighbors) for neighbors in neighbor_lists)
        self._neighbor_ids = np.full((len(neighbor_lists), max_neighbors), -1, dtype=np.int32)
        for face_id, neighbors in enumerate(neighbor_lists):
            self._neighbor_ids[face_id, :len(neighbors)] = [item[0] for item in neighbors]
        valid = self._neighbor_ids >= 0
        self._neighbor_rows = np.repeat(np.arange(len(neighbor_lists)), max_neighbors)[valid.ravel()]
        self._neighbor_cols = self._neighbor_ids[valid]

    def _capture_hotspot_sources(self) -> list[dict]:
        """Treat each generated chain's youngest volcano as a mantle-fixed source."""
        sources = []
        for chain in self.generator.hotspot_tracks:
            points = np.asarray(chain.get("points", ()), dtype=np.float64)
            if not len(points):
                continue
            plate_id = int(chain.get("plate_id", self.generator.face_plate_map[
                self._face_tree.query(points[0] * self.generator.mesh.radius)[1]
            ]))
            sources.append({
                "position": points[0].copy(),
                "plate_id": plate_id,
                "age_myr": 0.0,
            })
        return sources

    def _capture_generator_state(self) -> dict[str, np.ndarray]:
        plate_map = self.generator.face_plate_map.copy()
        thickness_by_plate = np.asarray(
            [plate.crust_thickness for plate in self.generator.plates], dtype=np.float64
        )
        density_by_plate = np.asarray(
            [plate.density for plate in self.generator.plates], dtype=np.float64
        )
        oceanic_by_plate = np.asarray(
            [plate.is_oceanic for plate in self.generator.plates], dtype=bool
        )
        return {
            "face_plate_map": plate_map,
            "terrain_face_heights": self.generator.terrain_face_heights.copy(),
            "land_face_mask": self.generator.land_face_mask.copy(),
            "continent_height_map": self.generator.continent_height_map.copy(),
            "crust_type_oceanic": oceanic_by_plate[plate_map],
            "crust_thickness": thickness_by_plate[plate_map],
            "crust_density": density_by_plate[plate_map],
            "crust_age_myr": np.zeros(len(plate_map), dtype=np.float64),
        }

    def _restore_state(self, state: dict[str, np.ndarray]) -> None:
        for key, value in state.items():
            setattr(self, key, value.copy())
        self.plate_rotations = np.repeat(
            np.eye(3)[None, :, :], len(self.generator.plates), axis=0
        )
        for plate, center in zip(self.generator.plates, self._initial_plate_centers):
            plate.center_pos = center.copy()
            plate.drift_vector = plate.velocity_at(center)
        if hasattr(self, "_initial_hotspot_sources"):
            self.hotspot_sources = [
                {**source, "position": source["position"].copy()}
                for source in self._initial_hotspot_sources
            ]

    def _sync_fields_to_generator(self) -> None:
        """Publish remapped face and vertex fields for the viewer and generator."""
        gen = self.generator
        gen.face_plate_map = self.face_plate_map.copy()
        gen.terrain_face_heights = self.terrain_face_heights.copy()
        gen.land_face_mask = self.land_face_mask.copy()
        gen.continent_height_map = self.continent_height_map.copy()

        plate_count = len(gen.plates)
        vertex_count = len(gen.mesh.vertices)
        votes = np.zeros((vertex_count, plate_count), dtype=np.uint16)
        heights_sum = np.zeros(vertex_count, dtype=np.float64)
        heights_count = np.zeros(vertex_count, dtype=np.uint8)
        for corner in range(3):
            vertices = gen.mesh.faces[:, corner]
            np.add.at(votes, (vertices, self.face_plate_map), 1)
            np.add.at(heights_sum, vertices, self.terrain_face_heights)
            np.add.at(heights_count, vertices, 1)
        gen.vertex_plate_map = np.argmax(votes, axis=1)
        gen.terrain_height_vertices = heights_sum / np.maximum(heights_count, 1)

    def _sync_to_generator(self) -> None:
        """Publish state and rebuild boundary locations/classes for the viewer."""
        self._sync_fields_to_generator()
        gen = self.generator

        for plate in gen.plates:
            plate.center_pos = self.plate_rotations[plate.id] @ self._initial_plate_centers[plate.id]
            plate.drift_vector = plate.velocity_at(plate.center_pos)
        # Rebuild line locations and motion labels from the evolved ownership map.
        gen._analyze_boundaries()

    @staticmethod
    def _empty_diagnostics() -> dict[str, float | int]:
        return {
            "face_count": 0,
            "unassigned_faces": 0,
            "fallback_faces": 0,
            "overlap_faces": 0,
            "simplified_faces": 0,
            "terrain_integral_change": 0.0,
            "crust_volume_change": 0.0,
            "crust_mass_proxy_change": 0.0,
            "crust_age_integral_change": 0.0,
        }

    def _remap_state(self, rotations: list[np.ndarray]) -> None:
        if all(np.array_equal(rotation, np.eye(3)) for rotation in rotations):
            self.last_remap_diagnostics = self._empty_diagnostics()
            self.last_remap_diagnostics["face_count"] = len(self._face_centers)
            return
        old = {key: getattr(self, key) for key in (
            "face_plate_map", "terrain_face_heights", "land_face_mask",
            "continent_height_map", "crust_type_oceanic", "crust_thickness",
            "crust_density", "crust_age_myr",
        )}
        count = len(self._face_centers)
        best_distance = np.full(count, np.inf)
        fallback_distance = np.full(count, np.inf)
        fallback_source = np.full(count, -1, dtype=np.int64)
        fallback_owner = np.full(count, -1, dtype=np.int64)
        best_source = np.full(count, -1, dtype=np.int64)
        best_owner = np.full(count, -1, dtype=np.int64)
        valid_candidate_count = np.zeros(count, dtype=np.uint8)
        continuous_keys = (
            "terrain_face_heights", "continent_height_map", "crust_thickness",
            "crust_density", "crust_age_myr",
        )
        remapped = {
            key: np.zeros(count, dtype=np.float64) for key in continuous_keys
        }

        for plate in self.generator.plates:
            # Row vectors are inverse-rotated by multiplying with R.
            inverse_points = self._face_centers @ rotations[plate.id]
            distances, source_ids = self._face_tree.query(inverse_points, k=4)
            source_ids = source_ids.astype(np.int64, copy=False)
            neighbor_owners = old["face_plate_map"][source_ids]
            neighbor_valid = neighbor_owners == plate.id
            valid = np.any(neighbor_valid, axis=1)
            valid_distances = np.where(neighbor_valid, distances, np.inf)
            nearest_valid_slot = np.argmin(valid_distances, axis=1)
            nearest_valid_distance = valid_distances[np.arange(count), nearest_valid_slot]
            nearest_valid_source = source_ids[np.arange(count), nearest_valid_slot]

            nearest_slot = np.argmin(distances, axis=1)
            nearest_source = source_ids[np.arange(count), nearest_slot]
            fallback_better = distances[np.arange(count), nearest_slot] < fallback_distance
            fallback_distance[fallback_better] = distances[np.arange(count), nearest_slot][fallback_better]
            fallback_source[fallback_better] = nearest_source[fallback_better]
            fallback_owner[fallback_better] = old["face_plate_map"][nearest_source[fallback_better]]

            valid_candidate_count += valid.astype(np.uint8)
            better = valid & (nearest_valid_distance < best_distance)
            best_distance[better] = nearest_valid_distance[better]
            best_source[better] = nearest_valid_source[better]
            best_owner[better] = plate.id

            # Interpolate continuous fields from up to four nearby source faces
            # belonging to this plate. Exact center hits receive all the weight.
            weights = np.where(neighbor_valid, 1.0 / np.maximum(distances, 1e-12), 0.0)
            weight_sums = weights.sum(axis=1)
            weights = weights / np.maximum(weight_sums[:, None], 1e-30)
            for key in continuous_keys:
                samples = old[key][source_ids]
                interpolated = np.sum(weights * samples, axis=1)
                remapped[key][better] = interpolated[better]

        covered = np.isfinite(best_distance)
        fallback_faces = int((~covered).sum())
        no_valid = ~covered
        if fallback_faces:
            # The nearest candidate is deterministic and guarantees a complete grid.
            best_source[no_valid] = fallback_source[no_valid]
            best_owner[no_valid] = fallback_owner[no_valid]
        result_owner = best_owner
        unassigned = int(np.count_nonzero(result_owner < 0))
        if unassigned:
            raise RuntimeError(f"inverse remapping left {unassigned} faces unassigned")

        for key in continuous_keys:
            values = remapped[key]
            values[no_valid] = old[key][best_source[no_valid]]
            setattr(self, key, values)
        for key in ("face_plate_map", "crust_type_oceanic"):
            setattr(self, key, old[key][best_source].copy())
        self.face_plate_map = result_owner.astype(np.int32, copy=False)
        self.land_face_mask = self.terrain_face_heights > 0.0
        simplified_faces = self._simplify_boundary_map()

        mesh_radius = float(self.generator.mesh.radius)
        area = self._face_areas * (self.planet_radius_km / mesh_radius) ** 2
        elevation_scale = self.planet_radius_km / mesh_radius
        terrain_before = float(np.dot(old["terrain_face_heights"] * elevation_scale, area))
        terrain_after = float(np.dot(self.terrain_face_heights * elevation_scale, area))
        volume_before = float(np.dot(old["crust_thickness"], area))
        volume_after = float(np.dot(self.crust_thickness, area))
        mass_before = float(np.dot(old["crust_thickness"] * old["crust_density"], area))
        mass_after = float(np.dot(self.crust_thickness * self.crust_density, area))
        age_before = float(np.dot(old["crust_age_myr"], area))
        age_after = float(np.dot(self.crust_age_myr, area))
        self.last_remap_diagnostics = {
            "face_count": count,
            "unassigned_faces": unassigned,
            "fallback_faces": fallback_faces,
            "overlap_faces": int(np.count_nonzero(valid_candidate_count > 1)),
            "simplified_faces": simplified_faces,
            "terrain_integral_change": terrain_after - terrain_before,
            "crust_volume_change": volume_after - volume_before,
            "crust_mass_proxy_change": mass_after - mass_before,
            "crust_age_integral_change": age_after - age_before,
        }

    def _simplify_boundary_map(self) -> int:
        """Remove one-face zigzags and tight notches with a conservative vote."""
        owner_count = len(self.generator.plates)
        neighbor_valid = self._neighbor_ids >= 0
        safe_neighbors = np.maximum(self._neighbor_ids, 0)
        neighbor_labels = self.face_plate_map[safe_neighbors]
        face_count = len(self.face_plate_map)
        flat_bins = self._neighbor_rows * owner_count + neighbor_labels[neighbor_valid]
        votes = np.bincount(flat_bins, minlength=face_count * owner_count).reshape(
            face_count, owner_count
        )
        degrees = neighbor_valid.sum(axis=1)
        rows = np.arange(face_count)
        current_votes = votes[rows, self.face_plate_map]
        replacement = np.argmax(votes, axis=1)
        replacement_votes = votes[rows, replacement]
        required_votes = np.ceil(degrees * 0.55).astype(int)
        change = (
            (replacement != self.face_plate_map)
            & (replacement_votes >= required_votes)
            & (replacement_votes > current_votes)
        )
        changed_faces = np.flatnonzero(change)
        if not len(changed_faces):
            return 0

        source_faces = np.full(face_count, -1, dtype=np.int64)
        for face_id in changed_faces:
            adjacent = self._neighbor_ids[face_id]
            matches = adjacent[(adjacent >= 0) & (self.face_plate_map[np.maximum(adjacent, 0)] == replacement[face_id])]
            if len(matches):
                source_faces[face_id] = int(matches.min())
        changed_faces = changed_faces[source_faces[changed_faces] >= 0]
        if not len(changed_faces):
            return 0

        fields = (
            "terrain_face_heights", "land_face_mask", "continent_height_map",
            "crust_type_oceanic", "crust_thickness", "crust_density", "crust_age_myr",
        )
        for key in fields:
            values = getattr(self, key)
            updated = values.copy()
            updated[changed_faces] = values[source_faces[changed_faces]]
            setattr(self, key, updated)
        self.face_plate_map[changed_faces] = replacement[changed_faces]
        return int(len(changed_faces))

    def _apply_convergent_uplift(self) -> None:
        """Accumulate a bounded, visible uplift belt along convergent boundaries."""
        gen = self.generator
        mesh = gen.mesh
        edge_uplift = np.zeros(len(mesh.faces), dtype=np.float64)
        for boundary in gen.boundaries:
            if boundary.boundary_type.value != "Convergent":
                continue
            plate_a = gen.plates[boundary.plate_a]
            plate_b = gen.plates[boundary.plate_b]
            if not plate_a.is_oceanic and not plate_b.is_oceanic:
                base_rate = 0.0040
            elif plate_a.is_oceanic != plate_b.is_oceanic:
                base_rate = 0.0018
            else:
                base_rate = 0.0008
            speed_factor = np.clip(abs(boundary.normal_velocity) / 0.004, 0.2, 1.5)
            face_ids = mesh.edge_to_faces.get(boundary.edge, ())
            if len(face_ids) == 2:
                np.maximum.at(edge_uplift, np.asarray(face_ids), base_rate * speed_factor)

        if not np.any(edge_uplift):
            return
        uplift = edge_uplift.copy()
        # Spread the boundary signal into a narrow belt; lower rates farther inland.
        for _ in range(4):
            propagated = np.zeros_like(uplift)
            np.maximum.at(propagated, self._neighbor_rows, uplift[self._neighbor_cols] * 0.58)
            uplift = np.maximum(edge_uplift, propagated)
        uplift *= self.timestep_myr * gen.mesh.radius
        max_height = 0.45 * gen.mesh.radius
        self.terrain_face_heights = np.minimum(self.terrain_face_heights + uplift, max_height)
        self.land_face_mask = self.terrain_face_heights > 0.0

    def _grow_hotspot_volcanoes(self) -> None:
        """Erupt at mantle-fixed sources under their original moving plate."""
        if not self.hotspot_sources:
            return
        mesh_radius = self.generator.mesh.radius
        face_radius = self._face_centers
        for source in self.hotspot_sources:
            position = source["position"]
            face_id = int(self._face_tree.query(position * mesh_radius, k=1)[1])
            if self.face_plate_map[face_id] != source["plate_id"]:
                continue
            source["age_myr"] += self.timestep_myr
            distances = np.arccos(np.clip(self._face_units @ position, -1.0, 1.0))
            radius = 0.055
            profile = np.maximum(0.0, 1.0 - (distances / radius) ** 2) ** 1.5
            peak_height = min(0.18, 0.012 * source["age_myr"]) * mesh_radius
            plate_mask = self.face_plate_map == source["plate_id"]
            new_relief = -0.025 * mesh_radius + peak_height * profile
            self.terrain_face_heights[plate_mask] = np.maximum(
                self.terrain_face_heights[plate_mask], new_relief[plate_mask]
            )
        self.land_face_mask = self.terrain_face_heights > 0.0

    def _apply_step(self, direction: float) -> None:
        """Apply one signed fixed step; direction is +1 or -1."""
        rotations = []
        for plate in self.generator.plates:
            omega = np.asarray(plate.angular_velocity, dtype=np.float64)
            rotation = _rotation_matrix(omega * (self.timestep_myr * direction))
            self.plate_rotations[plate.id] = rotation @ self.plate_rotations[plate.id]
            rotations.append(rotation)
        self._remap_state(rotations)
        self._sync_to_generator()
        self._apply_convergent_uplift()
        self._grow_hotspot_volcanoes()
        self._sync_fields_to_generator()

    def step(self) -> float:
        """Advance exactly one fixed timestep and return the new elapsed Myr."""
        self._apply_step(1.0)
        self.step_count += 1
        self.elapsed_myr = self.step_count * self.timestep_myr
        return self.elapsed_myr

    def transform_material_point(self, plate_id: int, position: np.ndarray) -> np.ndarray:
        """Return a point's world position after its plate's accumulated rotation."""
        if plate_id < 0 or plate_id >= len(self.plate_rotations):
            raise IndexError("plate_id is outside the simulation plate set")
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("position must be a 3D vector")
        return self.plate_rotations[plate_id] @ point

    def step_back(self) -> float:
        """Restore the previous fixed step by deterministic replay from reset."""
        if self.step_count > 0:
            target_step = self.step_count - 1
            self._restore_state(self._initial_state)
            self.last_remap_diagnostics = self._empty_diagnostics()
            self.last_remap_diagnostics["face_count"] = len(self._face_centers)
            self.step_count = 0
            self.elapsed_myr = 0.0
            for _ in range(target_step):
                self._apply_step(1.0)
                self.step_count += 1
                self.elapsed_myr = self.step_count * self.timestep_myr
        self._playback_accumulator = 0.0
        self._sync_to_generator()
        return self.elapsed_myr

    def reset(self) -> None:
        """Restore initial material positions and simulation clock."""
        self._restore_state(self._initial_state)
        self.last_remap_diagnostics = self._empty_diagnostics()
        self.last_remap_diagnostics["face_count"] = len(self._face_centers)
        self.elapsed_myr = 0.0
        self.step_count = 0
        self._playback_accumulator = 0.0
        self.is_playing = False
        self._sync_to_generator()

    def pause(self) -> None:
        self.is_playing = False

    def resume(self) -> None:
        self.is_playing = True

    def set_playback_speed(self, steps_per_second: float) -> None:
        if steps_per_second <= 0:
            raise ValueError("playback speed must be positive")
        self.playback_steps_per_second = float(steps_per_second)

    def advance(self, real_seconds: float) -> int:
        """Advance playback from frame duration; return fixed steps performed."""
        if real_seconds < 0:
            raise ValueError("real_seconds cannot be negative")
        if not self.is_playing or real_seconds == 0:
            return 0
        self._playback_accumulator += real_seconds * self.playback_steps_per_second
        step_count = int(self._playback_accumulator)
        for _ in range(step_count):
            self.step()
        self._playback_accumulator -= step_count
        return step_count
