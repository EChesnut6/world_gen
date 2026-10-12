import unittest

import numpy as np

from src.core.simulation import PlateSimulation
from src.core.sphere_mesh import SphereMesh
from src.plates.plate_gen import BoundarySegment, BoundaryType, PlateGenerator


class TestPlateSimulation(unittest.TestCase):
    def make_generator(self, seed=31, plate_count=5):
        mesh = SphereMesh(radius=1.0, subdivisions=3)
        generator = PlateGenerator(
            mesh=mesh,
            base_num_plates=plate_count,
            noise_seed=seed,
        )
        generator.generate()
        return generator

    def test_zero_velocity_preserves_surface_fields_and_advances_clock(self):
        generator = self.make_generator()
        for plate in generator.plates:
            plate.angular_velocity[:] = 0.0
        simulation = PlateSimulation(generator)
        initial_owners = simulation.face_plate_map.copy()
        initial_terrain = simulation.terrain_face_heights.copy()
        initial_ages = simulation.crust_age_myr.copy()

        self.assertEqual(simulation.step(), 1.0)
        self.assertEqual(simulation.step_count, 1)
        np.testing.assert_array_equal(simulation.face_plate_map, initial_owners)
        np.testing.assert_array_equal(simulation.terrain_face_heights, initial_terrain)
        np.testing.assert_array_equal(simulation.crust_age_myr, initial_ages)
        self.assertEqual(simulation.last_remap_diagnostics["unassigned_faces"], 0)
        self.assertEqual(simulation.last_remap_diagnostics["fallback_faces"], 0)

    def test_rigid_rotation_matches_expected_angular_displacement(self):
        generator = self.make_generator(plate_count=1)
        plate = generator.plates[0]
        plate.angular_velocity = np.array([0.0, 0.0, 0.4])
        simulation = PlateSimulation(generator)
        start = np.array([1.0, 0.0, 0.0])

        simulation.step()
        moved = simulation.transform_material_point(plate.id, start)
        expected = np.array([np.cos(0.4), np.sin(0.4), 0.0])
        np.testing.assert_allclose(moved, expected, atol=1e-12)
        self.assertAlmostEqual(np.arccos(np.clip(np.dot(start, moved), -1.0, 1.0)), 0.4)

    def test_terrain_feature_moves_with_plate_while_hotspot_source_stays_fixed(self):
        generator = self.make_generator(seed=8, plate_count=1)
        plate = generator.plates[0]
        plate.angular_velocity = np.array([0.0, 0.0, 0.6])

        face_units = generator.mesh.face_centers
        face_units = face_units / np.linalg.norm(face_units, axis=1, keepdims=True)
        source_position = face_units[
            int(np.argmax(face_units @ np.array([1.0, 0.0, 0.0])))
        ].copy()
        source_face = int(np.argmax(face_units @ source_position))
        generator.terrain_face_heights[:] = 0.0
        generator.terrain_face_heights[source_face] = 1.0
        generator.land_face_mask[:] = False
        generator.land_face_mask[source_face] = True
        generator.hotspot_tracks = []

        simulation = PlateSimulation(generator)
        simulation.step()

        expected_position = simulation.transform_material_point(plate.id, source_position)
        weighted_center = np.average(
            face_units, axis=0, weights=simulation.terrain_face_heights
        )
        weighted_center /= np.linalg.norm(weighted_center)
        feature_error = np.arccos(np.clip(np.dot(expected_position, weighted_center), -1.0, 1.0))
        self.assertGreater(simulation.terrain_face_heights.max(), 0.0)
        self.assertLess(feature_error, 0.05)

    def test_hotspot_source_is_fixed_and_builds_a_new_island_under_the_plate(self):
        generator = self.make_generator(seed=29, plate_count=1)
        generator.plates[0].is_oceanic = True
        generator.plates[0].angular_velocity = np.array([0.0, 0.0, 0.04])
        centers = generator.mesh.face_centers
        units = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        hotspot_source = units[100].copy()
        generator.terrain_face_heights[:] = -0.025 * generator.mesh.radius
        generator.hotspot_tracks = [{"plate_id": 0, "points": np.array([
            hotspot_source, [0.0, 0.9, 0.2]
        ])}]
        simulation = PlateSimulation(generator)

        for _ in range(3):
            simulation.step()

        np.testing.assert_array_equal(simulation.hotspot_sources[0]["position"], hotspot_source)
        self.assertAlmostEqual(simulation.hotspot_sources[0]["age_myr"], 3.0)
        source_face = int(simulation._face_tree.query(
            hotspot_source * generator.mesh.radius, k=1
        )[1])
        self.assertGreater(simulation.terrain_face_heights[source_face], 0.0)

    def test_boundary_cleanup_removes_an_isolated_one_face_notch(self):
        generator = self.make_generator(seed=30, plate_count=2)
        simulation = PlateSimulation(generator)
        owners = np.zeros(len(generator.mesh.faces), dtype=np.int32)
        notch_face = 100
        owners[notch_face] = 1
        simulation.face_plate_map = owners

        simplified = simulation._simplify_boundary_map()

        self.assertEqual(simplified, 1)
        self.assertEqual(simulation.face_plate_map[notch_face], 0)

    def test_continental_convergence_accumulates_a_broad_uplift_belt(self):
        generator = self.make_generator(seed=33, plate_count=2)
        generator.plates[0].is_oceanic = False
        generator.plates[1].is_oceanic = False
        simulation = PlateSimulation(generator)
        edge, adjacent_faces = next(iter(generator.mesh.edge_to_faces.items()))
        position = generator.mesh.vertices[list(edge)].mean(axis=0)
        generator.boundaries = [BoundarySegment(
            edge=edge,
            plate_a=0,
            plate_b=1,
            position=position,
            boundary_type=BoundaryType.CONVERGENT,
            stress=1.0,
            normal_velocity=0.004,
        )]
        before = simulation.terrain_face_heights.copy()

        simulation._apply_convergent_uplift()

        uplift = simulation.terrain_face_heights - before
        self.assertTrue(np.all(uplift[list(adjacent_faces)] > 0.0))
        self.assertGreater(float(uplift.max()), 0.003)
        self.assertGreater(np.count_nonzero(uplift), len(adjacent_faces))

    def test_coverage_diagnostics_and_reset_replay_are_deterministic(self):
        first_generator = self.make_generator(seed=17)
        second_generator = self.make_generator(seed=17)
        np.testing.assert_array_equal(first_generator.face_plate_map, second_generator.face_plate_map)
        np.testing.assert_array_equal(first_generator.terrain_face_heights, second_generator.terrain_face_heights)
        for a, b in zip(first_generator.plates, second_generator.plates):
            np.testing.assert_array_equal(a.angular_velocity, b.angular_velocity)

        first = PlateSimulation(first_generator)
        second = PlateSimulation(second_generator)
        for _ in range(4):
            first.step()
            second.step()
        np.testing.assert_array_equal(first.face_plate_map, second.face_plate_map)
        np.testing.assert_array_equal(first.terrain_face_heights, second.terrain_face_heights)
        self.assertTrue(np.all(first.face_plate_map >= 0))
        self.assertTrue(np.all(first.face_plate_map < len(first_generator.plates)))
        self.assertEqual(first.last_remap_diagnostics["unassigned_faces"], 0)

        expected_owners = first.face_plate_map.copy()
        expected_terrain = first.terrain_face_heights.copy()
        first.reset()
        self.assertEqual(first.elapsed_myr, 0.0)
        self.assertEqual(first.step_count, 0)
        for _ in range(4):
            first.step()
        np.testing.assert_array_equal(first.face_plate_map, expected_owners)
        np.testing.assert_array_equal(first.terrain_face_heights, expected_terrain)

    def test_playback_uses_fixed_steps_and_pause_stops_advancement(self):
        generator = self.make_generator()
        for plate in generator.plates:
            plate.angular_velocity[:] = 0.0
        simulation = PlateSimulation(generator, playback_steps_per_second=2.0)
        self.assertEqual(simulation.advance(1.0), 0)
        simulation.resume()
        self.assertEqual(simulation.advance(0.24), 0)
        self.assertEqual(simulation.advance(0.26), 1)
        self.assertEqual(simulation.step_count, 1)
        simulation.pause()
        self.assertEqual(simulation.advance(3.0), 0)
        self.assertEqual(simulation.step_count, 1)
        self.assertEqual(simulation.step_back(), 0.0)


if __name__ == "__main__":
    unittest.main()
