import sys
import numpy as np
import pygame

from src.core.sphere_mesh import SphereMesh
from src.core.simulation import PlateSimulation
from src.plates.plate_gen import BoundaryType, PlateGenerator
from src.viewer.camera import Camera3D
from src.viewer.ui import UIManager


class GlobeViewer:
    """
    High-performance 3D Pygame window for viewing and rotating the spherical world.
    Supports boundary classification, controlled microplate fracturing, adaptive LOD rendering,
    graph-colored tectonic plates, RGB orientation axes, live interactive controls, and real-time frame rates.
    """

    def __init__(self, mesh: SphereMesh | None = None, width: int = 1000, height: int = 800):
        pygame.init()
        pygame.display.set_caption("World Gen - 3D Planet & Tectonic Viewer")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.camera = Camera3D(width=width, height=height)
        self.mesh = mesh if mesh is not None else SphereMesh(radius=1.0, subdivisions=3)
        self.plate_generator: PlateGenerator | None = None
        self.simulation: PlateSimulation | None = None
        self._fracture_undo_state = None

        # Adaptive LOD configuration for ultra-fast 60+ FPS rendering
        self.use_adaptive_lod = True
        self.max_render_subdiv = 4  # Cap rendering at 5,120 faces for software rasterizer
        self._update_render_mesh()

        # Render and simulation options
        self.show_wireframe = True
        self.show_shading = True
        self.show_lithosphere = False
        self.show_terrain_colors = False
        self.show_orientation_axes = True
        self.show_drift_arrows = True
        self.show_boundaries = True
        self.is_rotating_auto = False
        self.rotation_speed = 1.0  # Multiplier for auto-spin speed
        self.planet_spin_angle = 0.0

        # Interaction state
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.font = pygame.font.SysFont("Helvetica", 16)
        self.title_font = pygame.font.SysFont("Helvetica", 20, bold=True)

        # Initialize modern interactive HUD UI manager
        self.ui = UIManager(self)

    def _update_render_mesh(self):
        """Precomputes active render geometry and face normal metadata based on adaptive LOD setting."""
        if self.use_adaptive_lod and self.mesh.subdivisions > self.max_render_subdiv:
            render_level = self.max_render_subdiv
            verts, faces = self.mesh.lod_meshes[render_level]
        else:
            render_level = self.mesh.subdivisions
            verts, faces = self.mesh.vertices, self.mesh.faces

        self.render_subdiv = render_level
        self.base_render_verts = verts
        self.render_verts = verts
        self.render_faces = faces
        self.num_render_faces = len(faces)
        self._sync_terrain_geometry()
        self.render_face_colors = np.zeros((self.num_render_faces, 3), dtype=np.uint8)
        self._sync_face_colors()
        self._sync_lithosphere_map()

    def set_plate_generator(self, generator: PlateGenerator):
        """Assign plate generator reference and synchronize face colors."""
        self.plate_generator = generator
        self._rebuild_simulation()
        self._sync_face_colors()
        self._sync_lithosphere_map()

    def _rebuild_simulation(self):
        """Start a fresh fixed-step history from the current generated world."""
        self.simulation = PlateSimulation(self.plate_generator) if self.plate_generator else None

    def simulation_step_forward(self):
        if self.simulation:
            self.simulation.step()
            self._sync_face_colors()

    def simulation_step_back(self):
        if self.simulation:
            self.simulation.step_back()
            self._sync_face_colors()

    def simulation_reset(self):
        if self.simulation:
            self.simulation.reset()
            self._sync_face_colors()

    def simulation_play(self):
        if self.simulation:
            self.simulation.resume()

    def simulation_pause(self):
        if self.simulation:
            self.simulation.pause()

    def adjust_simulation_speed(self, multiplier: float):
        if not self.simulation:
            return
        speed = self.simulation.playback_steps_per_second * multiplier
        self.simulation.set_playback_speed(max(0.125, min(64.0, speed)))

    def _get_lod_descendant_offsets(self, r_level: int, t_level: int) -> np.ndarray:
        """
        Computes exact hierarchical child face index offsets from render LOD level to simulation target level.
        In the icosphere subdivision tree, each face generates 4 sub-faces with offsets k * N_level.
        """
        offsets = [0]
        for lvl in range(r_level, t_level):
            n_lvl = len(self.mesh.lod_meshes[lvl][1])
            new_offsets = []
            for base in offsets:
                for k in range(4):
                    new_offsets.append(base + k * n_lvl)
            offsets = new_offsets
        return np.array(offsets, dtype=int)

    def _sync_face_colors(self):
        """
        Synchronizes color mapping between high-res plate simulation and render LOD mesh.
        Uses exact hierarchical child face mapping and sub-millisecond majority voting
        so that simplified LOD rendering precisely preserves tectonic plate boundaries and shapes.
        """
        if self.plate_generator is None:
            self.render_face_colors = np.zeros((self.num_render_faces, 3), dtype=np.uint8)
            self._sync_terrain_geometry()
            return

        target_subdiv = self.mesh.subdivisions
        if self.render_subdiv < target_subdiv:
            # 1. Retrieve cached hierarchical child offsets
            cache_key = (self.render_subdiv, target_subdiv)
            if not hasattr(self, "_lod_offset_cache"):
                self._lod_offset_cache = {}
            if cache_key not in self._lod_offset_cache:
                self._lod_offset_cache[cache_key] = self._get_lod_descendant_offsets(self.render_subdiv, target_subdiv)
            offsets = self._lod_offset_cache[cache_key]

            # 2. Gather plate IDs for all descendant faces: shape (num_render_faces, num_descendants)
            descendant_indices = np.arange(self.num_render_faces, dtype=int)[:, None] + offsets[None, :]
            descendant_plates = self.plate_generator.face_plate_map[descendant_indices]

            # 3. Vectorized majority vote per LOD face using 2D bincount
            max_p_id = max(p.id for p in self.plate_generator.plates) if self.plate_generator.plates else 0
            num_bins = max_p_id + 1

            valid_mask = (descendant_plates >= 0) & (descendant_plates <= max_p_id)
            clean_plates = np.where(valid_mask, descendant_plates, 0)
            flat = clean_plates + np.arange(self.num_render_faces)[:, None] * num_bins
            counts = np.bincount(
                flat.ravel(),
                weights=valid_mask.ravel().astype(int),
                minlength=self.num_render_faces * num_bins
            ).reshape(self.num_render_faces, num_bins)

            lod_plate_ids = np.argmax(counts, axis=1)
            # At a coarse LOD, boundaries often split a face evenly. Resolve
            # those ties with the descendant nearest the coarse face center so
            # the visible boundary follows the actual high resolution plate map.
            tied = (counts == counts.max(axis=1, keepdims=True))
            ambiguous = tied.sum(axis=1) > 1
            if np.any(ambiguous):
                centers = self.render_face_centers[ambiguous]
                centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
                child_centers = self.mesh.face_centers[descendant_indices[ambiguous]]
                child_centers /= np.maximum(np.linalg.norm(child_centers, axis=2, keepdims=True), 1e-12)
                tied_rows = tied[ambiguous]
                tied_plate_ids = descendant_plates[ambiguous]
                child_alignment = (child_centers @ centers[:, :, None]).squeeze(axis=2)
                safe_ids = np.clip(tied_plate_ids, 0, tied_rows.shape[1] - 1)
                candidate_children = valid_mask[ambiguous] & tied_rows[np.arange(len(tied_rows))[:, None], safe_ids]
                child_alignment[~candidate_children] = -np.inf
                nearest = np.argmax(child_alignment, axis=1)
                lod_plate_ids[ambiguous] = tied_plate_ids[np.arange(len(nearest)), nearest]
            has_valid = counts.sum(axis=1) > 0

            # 4. Map dominant plate ID to color
            color_lut = np.zeros((num_bins, 3), dtype=np.uint8)
            for p in self.plate_generator.plates:
                color_lut[p.id] = p.color

            self.render_face_colors = color_lut[lod_plate_ids]
            self.render_face_colors[~has_valid] = [100, 100, 100]
            land_fraction = self.plate_generator.land_face_mask[descendant_indices].mean(axis=1)
            self.render_face_colors[land_fraction >= 0.25] = [65, 175, 85]
        else:
            self.render_face_colors = self.plate_generator.get_face_colors()
            self.render_face_colors[self.plate_generator.land_face_mask] = [65, 175, 85]
            render_heights = self.plate_generator.terrain_face_heights
        if self.show_terrain_colors:
            radius = self.mesh.radius
            face_heights = self.plate_generator.terrain_face_heights
            land = face_heights > 0.0
            terrain_colors = np.empty((len(face_heights), 3), dtype=np.float64)
            if np.any(land):
                max_elevation = max(float(face_heights[land].max()), 1e-12)
                altitude = np.clip(face_heights[land] / max_elevation, 0.0, 1.0) ** 0.7
                lowland = np.array([48.0, 145.0, 72.0])
                snow = np.array([255.0, 255.0, 248.0])
                terrain_colors[land] = lowland[None, :] * (1.0 - altitude[:, None]) + snow[None, :] * altitude[:, None]
            water = ~land
            if np.any(water):
                depth = np.clip(-face_heights[water] / max(0.08 * radius, 1e-12), 0.0, 1.0)
                depth = depth * depth * (3.0 - 2.0 * depth)
                plate_is_oceanic = np.zeros(len(self.plate_generator.plates), dtype=bool)
                for plate in self.plate_generator.plates:
                    plate_is_oceanic[plate.id] = plate.is_oceanic
                if len(plate_is_oceanic):
                    water_plate_ids = np.clip(self.plate_generator.face_plate_map[water], 0, len(plate_is_oceanic) - 1)
                    ocean_water = plate_is_oceanic[water_plate_ids]
                else:
                    ocean_water = np.ones(np.count_nonzero(water), dtype=bool)
                ocean_blue = np.array([28.0, 78.0, 142.0])
                continental_water = np.array([78.0, 132.0, 178.0])
                dark_blue = np.array([12.0, 42.0, 96.0])
                water_base = np.where(ocean_water[:, None], ocean_blue[None, :], continental_water[None, :])
                terrain_colors[water] = water_base * (1.0 - 0.20 * depth[:, None]) + dark_blue[None, :] * (0.20 * depth[:, None])

                # Propagate distance from land through water faces so the
                # shallow blue coastal band follows every continent and island.
                from collections import deque
                shore_radius = 20
                shore_distance = np.full(len(face_heights), shore_radius + 1, dtype=np.int8)
                queue = deque()
                for face_id in np.flatnonzero(land):
                    shore_distance[face_id] = 0
                    queue.append(face_id)
                while queue:
                    face_id = queue.popleft()
                    next_distance = shore_distance[face_id] + 1
                    if next_distance > shore_radius:
                        continue
                    for neighbor, _ in self.mesh.face_neighbors[face_id]:
                        if water[neighbor] and next_distance < shore_distance[neighbor]:
                            shore_distance[neighbor] = next_distance
                            queue.append(neighbor)
                coast = np.array([104.0, 166.0, 198.0])
                water_shore_distance = shore_distance[water].astype(np.float64)
                # Every land cell gets a visible shallow-water buffer extending
                # five neighboring face steps into the surrounding ocean.
                coast_factor = 0.78 * np.clip(
                    (shore_radius + 1.0 - water_shore_distance) / shore_radius, 0.0, 1.0
                )
                terrain_colors[water] = (
                    terrain_colors[water] * (1.0 - coast_factor[:, None])
                    + coast[None, :] * coast_factor[:, None]
                )

                # Blend neighboring water faces slightly to soften abrupt
                # color seams between ocean basins while keeping coastlines crisp.
                cache_key = id(self.mesh)
                if getattr(self, "_water_neighbor_cache_key", None) != cache_key:
                    self._water_neighbor_pairs = np.asarray(
                        [(face, neighbor) for face, neighbors in enumerate(self.mesh.face_neighbors)
                         for neighbor, _ in neighbors if face < neighbor],
                        dtype=np.int32,
                    ).reshape(-1, 2)
                    self._water_neighbor_cache_key = cache_key
                pairs = self._water_neighbor_pairs
                wet_pairs = pairs[water[pairs].all(axis=1)]
                if len(wet_pairs):
                    blendable = np.zeros(len(face_heights), dtype=bool)
                    blendable[wet_pairs.ravel()] = True
                    neighbor_counts = np.zeros(len(face_heights), dtype=np.float64)
                    np.add.at(neighbor_counts, wet_pairs[:, 0], 1.0)
                    np.add.at(neighbor_counts, wet_pairs[:, 1], 1.0)
                    # Diffuse water colors over several face steps so the
                    # oceanic/continental plate boundary reads as a gradient.
                    for _ in range(8):
                        neighbor_sums = np.zeros_like(terrain_colors)
                        np.add.at(neighbor_sums, wet_pairs[:, 0], terrain_colors[wet_pairs[:, 1]])
                        np.add.at(neighbor_sums, wet_pairs[:, 1], terrain_colors[wet_pairs[:, 0]])
                        neighbor_average = neighbor_sums[blendable] / neighbor_counts[blendable, None]
                        terrain_colors[blendable] = 0.55 * terrain_colors[blendable] + 0.45 * neighbor_average

            terrain_colors = terrain_colors.astype(np.uint8)
            if self.render_subdiv < self.mesh.subdivisions:
                self.render_face_colors = terrain_colors[descendant_indices].mean(axis=1).astype(np.uint8)
            else:
                self.render_face_colors = terrain_colors
        self._sync_terrain_geometry()

    def _displace_vertices(self, base_vertices: np.ndarray) -> np.ndarray:
        """Apply generated terrain relief radially to a mesh vertex array."""
        if self.plate_generator is None:
            return base_vertices
        heights = self.plate_generator.terrain_height_vertices[:len(base_vertices)]
        lengths = np.linalg.norm(base_vertices, axis=1, keepdims=True)
        directions = base_vertices / np.maximum(lengths, 1e-12)
        return base_vertices + directions * heights[:, None]

    def _sync_terrain_geometry(self):
        """Refresh displaced vertices and face geometry after terrain changes."""
        self.render_verts = self._displace_vertices(self.base_render_verts)
        face_verts = self.render_verts[self.render_faces]
        self.render_face_centers = np.mean(face_verts, axis=1)
        e1 = face_verts[:, 1, :] - face_verts[:, 0, :]
        e2 = face_verts[:, 2, :] - face_verts[:, 0, :]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.render_face_normals = normals / norms

    def _sync_lithosphere_map(self):
        """Map vertex noise values onto the active render faces, including adaptive LOD."""
        if self.plate_generator is None:
            self.render_lithosphere_values = np.zeros(self.num_render_faces, dtype=np.float64)
            return

        heights = self.plate_generator.lithosphere_height
        if self.render_subdiv == self.mesh.subdivisions:
            self.render_lithosphere_values = np.mean(heights[self.render_faces], axis=1)
            return

        cache_key = (self.render_subdiv, self.mesh.subdivisions)
        if not hasattr(self, "_lod_offset_cache"):
            self._lod_offset_cache = {}
        if cache_key not in self._lod_offset_cache:
            self._lod_offset_cache[cache_key] = self._get_lod_descendant_offsets(*cache_key)
        offsets = self._lod_offset_cache[cache_key]
        descendants = np.arange(self.num_render_faces, dtype=int)[:, None] + offsets[None, :]
        descendant_vertices = self.mesh.faces[descendants]
        self.render_lithosphere_values = np.mean(heights[descendant_vertices], axis=(1, 2))

    # --- Live Key Stats Modifiers ---
    def adjust_base_plates(self, delta: int):
        if not self.plate_generator:
            return
        new_val = max(4, min(32, self.plate_generator.base_num_plates + delta))
        if new_val != self.plate_generator.base_num_plates:
            print(f"Live Updating Base Plates: {new_val}")
            self.plate_generator.generate(base_num_plates=new_val)
            self._rebuild_simulation()
            self._sync_face_colors()

    def adjust_oceanic_ratio(self, delta: float):
        if not self.plate_generator:
            return
        new_val = round(max(0.0, min(1.0, self.plate_generator.oceanic_ratio + delta)), 2)
        if new_val != self.plate_generator.oceanic_ratio:
            print(f"Live Updating Oceanic Ratio: {int(new_val * 100)}%")
            self.plate_generator.generate(oceanic_ratio=new_val)
            self._rebuild_simulation()
            self._sync_face_colors()

    def adjust_subdivisions(self, delta: int):
        new_subdiv = max(2, min(6, self.mesh.subdivisions + delta))
        if new_subdiv != self.mesh.subdivisions:
            print(f"Live Updating Mesh Subdivision Level: {new_subdiv} ({20 * 4**(new_subdiv-1):,} faces)")
            self.mesh = SphereMesh(radius=self.mesh.radius, subdivisions=new_subdiv)
            if self.plate_generator:
                self.plate_generator.set_mesh(self.mesh)
            self._update_render_mesh()
            self._rebuild_simulation()
            self._sync_face_colors()

    def adjust_warp_strength(self, delta: float):
        if not self.plate_generator:
            return
        new_val = round(max(0.0, min(0.40, self.plate_generator.warp_strength + delta)), 2)
        if new_val != self.plate_generator.warp_strength:
            print(f"Live Updating Warp Jaggedness: {new_val:.2f}")
            self.plate_generator.update_warp(new_val)
            self._rebuild_simulation()
            self._sync_face_colors()

    def adjust_max_microplates(self, delta: int):
        if not self.plate_generator:
            return
        new_val = max(0, min(16, self.plate_generator.max_microplates + delta))
        self.plate_generator.max_microplates = new_val
        print(f"Live Updating Max Microplates: {new_val}")

    def adjust_continental_land_percent(self, delta: int):
        if not self.plate_generator:
            return
        new_val = max(0, min(100, self.plate_generator.continental_land_percent + delta))
        if new_val != self.plate_generator.continental_land_percent:
            self.plate_generator.set_continental_land_percent(new_val)
            self._rebuild_simulation()
            self._sync_face_colors()
            print(f"Continental land coverage: {new_val}%")

    def adjust_rotation_speed(self, delta: float):
        new_val = round(max(0.2, min(3.0, self.rotation_speed + delta)), 1)
        self.rotation_speed = new_val

    # --- Toggles ---
    def toggle_boundaries(self):
        self.show_boundaries = not self.show_boundaries

    def toggle_drift_arrows(self):
        self.show_drift_arrows = not self.show_drift_arrows

    def toggle_wireframe(self):
        self.show_wireframe = not self.show_wireframe

    def toggle_shading(self):
        self.show_shading = not self.show_shading

    def toggle_lithosphere(self):
        self.show_lithosphere = not self.show_lithosphere

    def toggle_terrain_colors(self):
        self.show_terrain_colors = not self.show_terrain_colors
        self._sync_face_colors()

    def toggle_orientation_axes(self):
        self.show_orientation_axes = not self.show_orientation_axes

    def toggle_adaptive_lod(self):
        self.use_adaptive_lod = not self.use_adaptive_lod
        self._update_render_mesh()
        print(f"Adaptive LOD: {'ENABLED' if self.use_adaptive_lod else 'DISABLED'} (Subdiv Level: {self.render_subdiv})")

    def toggle_auto_rotation(self):
        self.is_rotating_auto = not self.is_rotating_auto

    def toggle_invert_controls(self):
        self.camera.invert_x = not self.camera.invert_x
        self.camera.invert_y = not self.camera.invert_y

    # --- Actions ---
    def action_new_plates(self):
        if self.plate_generator:
            print("Generating New Base Tectonic Plates...")
            self._fracture_undo_state = None
            next_seed = self.plate_generator.noise_seed + 1
            self.plate_generator.regenerate_noise(seed=next_seed)
            self.plate_generator.generate()
            self._rebuild_simulation()
            self._sync_face_colors()

    def action_fracture_microplate(self):
        if self.plate_generator:
            print("Applying collision-zone fracture or fusion event...")
            previous_count = len(self.plate_generator.plates)
            snapshot = self.plate_generator.snapshot_fracture_state()
            self.plate_generator.shatter_high_stress_zones(spawn_count=1)
            if len(self.plate_generator.plates) > previous_count:
                self._fracture_undo_state = snapshot
            self._rebuild_simulation()
            self._sync_face_colors()

    def action_undo_fracture(self):
        if self.plate_generator and self._fracture_undo_state is not None:
            self.plate_generator.restore_fracture_state(self._fracture_undo_state)
            self._fracture_undo_state = None
            self._rebuild_simulation()
            self._sync_face_colors()

    def action_reset_fractures(self):
        if self.plate_generator:
            print("Resetting microplate fractures...")
            self._fracture_undo_state = None
            self.plate_generator.reset_fractures()
            self._rebuild_simulation()
            self._sync_face_colors()

    def action_reset_camera(self):
        self.camera.yaw = 0.0
        self.camera.pitch = 0.0
        self.camera.distance = 3.5

    def handle_events(self) -> bool:
        """Processes user input events cleanly with UI priority routing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.camera.width = self.width
                self.camera.height = self.height
                self.ui.update_screen_size(self.width, self.height)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Let UI handle the click first; only drag camera if click was outside UI
                    if not self.ui.handle_click(event.pos):
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                elif event.button == 4:
                    self.camera.zoom(1.0)
                elif event.button == 5:
                    self.camera.zoom(-1.0)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                self.ui.handle_mouse_motion(event.pos)
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera.rotate(dx, dy)
                    self.last_mouse_pos = event.pos

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.action_new_plates()
                elif event.key == pygame.K_z:
                    self.action_reset_fractures()
                elif event.key == pygame.K_f:
                    self.action_fracture_microplate()
                elif event.key == pygame.K_u:
                    self.action_undo_fracture()
                elif event.key == pygame.K_l:
                    self.toggle_adaptive_lod()
                elif event.key == pygame.K_b:
                    self.toggle_boundaries()
                elif event.key == pygame.K_d:
                    self.toggle_drift_arrows()
                elif event.key == pygame.K_o:
                    self.toggle_orientation_axes()
                elif event.key == pygame.K_i:
                    self.toggle_invert_controls()
                elif event.key == pygame.K_w:
                    self.toggle_wireframe()
                elif event.key == pygame.K_s:
                    self.toggle_shading()
                elif event.key == pygame.K_n:
                    self.toggle_lithosphere()
                elif event.key == pygame.K_m:
                    self.toggle_terrain_colors()
                elif event.key == pygame.K_SPACE:
                    self.toggle_auto_rotation()
                elif event.key == pygame.K_r:
                    self.action_reset_camera()
                elif event.key == pygame.K_h:
                    self.ui.toggle_collapse()

        return True

    def render(self):
        """High performance vectorized 3D planet scene rendering."""
        self.screen.fill((15, 18, 26))

        if self.is_rotating_auto:
            self.planet_spin_angle += 1.5 * self.rotation_speed * 0.005

        spin_cos, spin_sin = np.cos(self.planet_spin_angle), np.sin(self.planet_spin_angle)
        planet_rotation = np.array([
            [spin_cos, 0.0, spin_sin],
            [0.0, 1.0, 0.0],
            [-spin_sin, 0.0, spin_cos],
        ])
        camera_rotation = self.camera.get_rotation_matrix()
        scene_rotation = camera_rotation @ planet_rotation

        # Single vectorized project_points call for render mesh
        world_verts = self.render_verts @ planet_rotation.T
        screen_pts, z_depths = self.camera.project_points(world_verts)

        cam_normals = self.render_face_normals @ scene_rotation.T
        cam_centers = self.render_face_centers @ scene_rotation.T
        cam_centers[:, 2] += self.camera.distance

        view_dirs = cam_centers / np.linalg.norm(cam_centers, axis=1, keepdims=True)
        dots = np.einsum('ij,ij->i', cam_normals, view_dirs)

        visible_mask = dots < 0.05
        vis_indices = np.where(visible_mask)[0]

        if len(vis_indices) > 0:
            face_z = np.mean(z_depths[self.render_faces[vis_indices]], axis=1)
            sorted_order = np.argsort(-face_z)
            sorted_vis_indices = vis_indices[sorted_order]

            light_dir = np.array([0.5, 0.7, -1.0])
            light_dir /= np.linalg.norm(light_dir)
            light_dir_camera = light_dir @ camera_rotation.T
            if self.show_shading:
                intensities = np.clip(cam_normals[sorted_vis_indices] @ light_dir_camera, 0.2, 1.0)
            else:
                intensities = np.ones(len(sorted_vis_indices))

            if self.show_lithosphere:
                grayscale = np.clip(self.render_lithosphere_values[sorted_vis_indices], 0.0, 1.0)
                colors = np.repeat((grayscale[:, None] * 255).astype(np.uint8), 3, axis=1)
            else:
                colors = self.render_face_colors[sorted_vis_indices]
                height_brightness = 0.68 + 0.32 * np.clip(
                    self.render_lithosphere_values[sorted_vis_indices], 0.0, 1.0
                )
                colors = np.clip(colors.astype(np.float64) * height_brightness[:, None], 0, 255).astype(np.uint8)
            wire_col = (45, 65, 90)

            face_screen_pts = screen_pts[self.render_faces[sorted_vis_indices]]

            for idx in range(len(sorted_vis_indices)):
                poly = face_screen_pts[idx]

                intens = intensities[idx]
                base_col = colors[idx]
                lit_col = (
                    int(base_col[0] * intens),
                    int(base_col[1] * intens),
                    int(base_col[2] * intens)
                )
                pygame.draw.polygon(self.screen, lit_col, poly)

                if self.show_wireframe:
                    pygame.draw.polygon(self.screen, wire_col, poly, 1)

        if self.show_boundaries and self.plate_generator is not None:
            if self.render_subdiv < self.mesh.subdivisions:
                highres_verts = self._displace_vertices(self.mesh.vertices) @ planet_rotation.T
                highres_screen_pts, _ = self.camera.project_points(highres_verts)
            else:
                highres_screen_pts = screen_pts

            self._render_boundaries(highres_screen_pts, scene_rotation)

        if self.show_drift_arrows and self.plate_generator is not None:
            self._render_drift_arrows(planet_rotation)

        if self.show_orientation_axes:
            self._render_orientation_axes(planet_rotation)

        # Render modern restructured UI with live steppers and clean toggles
        self.ui.draw(self.screen)
        pygame.display.flip()

    def _render_boundaries(self, highres_screen_pts: np.ndarray, R: np.ndarray):
        """Draws color-coded plate boundary lines using vectorized matrix operations."""
        gen = self.plate_generator
        if gen is None or len(gen.boundary_positions_arr) == 0:
            return

        edges = gen.boundary_edges_arr
        type_ids = gen.boundary_types_arr
        edge_heights = np.mean(gen.terrain_height_vertices[edges], axis=1)
        radial = gen.boundary_positions_arr / np.maximum(
            np.linalg.norm(gen.boundary_positions_arr, axis=1, keepdims=True), 1e-12
        )
        positions = gen.boundary_positions_arr + radial * edge_heights[:, None]

        cam_pos = positions @ R.T
        cam_pos[:, 2] += self.camera.distance

        norms = np.linalg.norm(cam_pos, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        view_dirs = cam_pos / norms

        norm_units = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        cam_normals = norm_units @ R.T

        dots = np.einsum('ij,ij->i', cam_normals, view_dirs)
        vis_mask = dots < 0.15
        vis_indices = np.where(vis_mask)[0]

        if len(vis_indices) == 0:
            return

        type_colors = np.array([
            [255, 60, 60],   # 0: Convergent (Red)
            [60, 180, 255],  # 1: Divergent (Blue)
            [255, 220, 50],  # 2: Transform (Yellow)
        ], dtype=np.uint8)

        vis_edges = edges[vis_indices]
        vis_types = type_ids[vis_indices]

        p1_pts = highres_screen_pts[vis_edges[:, 0]]
        p2_pts = highres_screen_pts[vis_edges[:, 1]]

        for i in range(len(vis_indices)):
            col = type_colors[vis_types[i]]
            pygame.draw.line(self.screen, col, p1_pts[i], p2_pts[i], 3)

    def _render_drift_arrows(self, planet_rotation: np.ndarray):
        """Draws 3D direction vectors showing movement direction for each tectonic plate."""
        arrow_starts = []
        arrow_ends = []

        for plate in self.plate_generator.plates:
            start_pos = plate.center_pos * 1.05
            direction = plate.drift_vector
            speed = np.linalg.norm(direction)
            arrow_vector = direction / speed * 0.35 if speed > 1e-12 else direction
            end_pos = start_pos + arrow_vector
            arrow_starts.append(start_pos)
            arrow_ends.append(end_pos)

        if not arrow_starts:
            return

        all_pts = np.vstack([arrow_starts, arrow_ends]) @ planet_rotation.T
        screen_pts, z_depths = self.camera.project_points(all_pts)
        num_plates = len(arrow_starts)

        for i in range(num_plates):
            start_2d = screen_pts[i]
            end_2d = screen_pts[i + num_plates]
            z_val = z_depths[i]

            if z_val < self.camera.distance * 1.05:
                pygame.draw.line(self.screen, (20, 20, 20), start_2d, end_2d, 5)
                pygame.draw.line(self.screen, (255, 235, 60), start_2d, end_2d, 3)

                vec = end_2d - start_2d
                length = np.linalg.norm(vec)
                if length > 5:
                    u = vec / length
                    perp = np.array([-u[1], u[0]])
                    wing1 = end_2d - u * 10 + perp * 5
                    wing2 = end_2d - u * 10 - perp * 5

                    pygame.draw.polygon(self.screen, (255, 235, 60), [end_2d, wing1, wing2])
                    pygame.draw.polygon(self.screen, (20, 20, 20), [end_2d, wing1, wing2], 1)

    def _render_orientation_axes(self, planet_rotation: np.ndarray):
        """Show short N/S pole markers attached to the rotating planet."""
        radius = self.mesh.radius
        pole_points = np.array([
            [0.0, radius, 0.0],
            [0.0, radius * 1.14, 0.0],
            [0.0, -radius, 0.0],
            [0.0, -radius * 1.14, 0.0],
        ]) @ planet_rotation.T
        screen_points, depths = self.camera.project_points(pole_points)

        for base_idx, tip_idx, label, color in (
            (0, 1, "N", (115, 220, 255)),
            (2, 3, "S", (255, 190, 125)),
        ):
            # Do not draw a far-side pole through the globe.
            if depths[base_idx] > self.camera.distance + radius * 0.02:
                continue
            base = screen_points[base_idx]
            tip = screen_points[tip_idx]
            pygame.draw.line(self.screen, (18, 22, 30), base, tip, 5)
            pygame.draw.line(self.screen, color, base, tip, 2)
            label_surface = self.title_font.render(label, True, color)
            label_pos = (int(tip[0] + 4), int(tip[1] - label_surface.get_height() // 2))
            self.screen.blit(label_surface, (label_pos[0] + 1, label_pos[1] + 1))
            self.screen.blit(label_surface, label_pos)



    def run(self):
        """Main rendering loop targeted for maximum frame rate."""
        running = True
        while running:
            running = self.handle_events()
            elapsed_seconds = self.clock.tick(144) / 1000.0
            if self.simulation and self.simulation.advance(elapsed_seconds):
                self._sync_face_colors()
            self.render()
        pygame.quit()
