import os
import unittest
import numpy as np

# Set dummy video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

from src.core.sphere_mesh import SphereMesh
from src.plates.plate_gen import PlateGenerator
from src.viewer.globe_viewer import GlobeViewer


class TestUIAndLiveStats(unittest.TestCase):
    def setUp(self):
        self.mesh = SphereMesh(radius=1.0, subdivisions=3)
        self.generator = PlateGenerator(
            mesh=self.mesh,
            base_num_plates=12,
            oceanic_ratio=0.70,
            warp_strength=0.16,
            max_microplates=6
        )
        self.generator.generate()
        self.viewer = GlobeViewer(mesh=self.mesh, width=1024, height=768)
        self.viewer.set_plate_generator(self.generator)

    def test_live_stat_adjustments(self):
        # Base plates adjustment
        self.viewer.adjust_base_plates(4)
        self.assertEqual(self.generator.base_num_plates, 16)
        self.assertEqual(len(self.generator.plates), 16)

        # Oceanic ratio adjustment
        self.viewer.adjust_oceanic_ratio(-0.10)
        self.assertEqual(round(self.generator.oceanic_ratio, 2), 0.60)

        # Warp distortion adjustment
        self.viewer.adjust_warp_strength(0.04)
        self.assertEqual(round(self.generator.warp_strength, 2), 0.20)

        # Max microplates adjustment
        self.viewer.adjust_max_microplates(3)
        self.assertEqual(self.generator.max_microplates, 9)

        # Auto rotation speed adjustment
        self.viewer.adjust_rotation_speed(0.4)
        self.assertEqual(round(self.viewer.rotation_speed, 1), 1.4)

        # Mesh subdivision live adjustment
        orig_faces = len(self.viewer.mesh.faces)
        self.viewer.adjust_subdivisions(1)
        self.assertEqual(self.viewer.mesh.subdivisions, 4)
        self.assertGreater(len(self.viewer.mesh.faces), orig_faces)

    def test_toggles_and_actions(self):
        # Toggle boundaries
        b_before = self.viewer.show_boundaries
        self.viewer.toggle_boundaries()
        self.assertNotEqual(self.viewer.show_boundaries, b_before)

        # Toggle drift arrows
        d_before = self.viewer.show_drift_arrows
        self.viewer.toggle_drift_arrows()
        self.assertNotEqual(self.viewer.show_drift_arrows, d_before)

        # Toggle wireframe
        w_before = self.viewer.show_wireframe
        self.viewer.toggle_wireframe()
        self.assertNotEqual(self.viewer.show_wireframe, w_before)

        # Toggle shading
        s_before = self.viewer.show_shading
        self.viewer.toggle_shading()
        self.assertNotEqual(self.viewer.show_shading, s_before)

        # Toggle auto rotation
        rot_before = self.viewer.is_rotating_auto
        self.viewer.toggle_auto_rotation()
        self.assertNotEqual(self.viewer.is_rotating_auto, rot_before)

        # Fracture and reset
        self.viewer.action_fracture_microplate()
        self.assertGreaterEqual(len(self.generator.plates), self.generator.base_num_plates)
        self.viewer.action_reset_fractures()
        self.assertEqual(len(self.generator.plates), self.generator.base_num_plates)

    def test_ui_manager_interactions(self):
        ui = self.viewer.ui

        # UI elements present
        self.assertEqual(len(ui.steppers), 8)
        self.assertEqual(len(ui.toggles), 10)
        self.assertEqual(len(ui.action_buttons), 5)

        # Collapse / expand
        self.assertFalse(ui.is_collapsed)
        ui.handle_click(ui.collapse_btn_rect.center)
        self.assertTrue(ui.is_collapsed)
        ui.handle_click(ui.collapse_btn_rect.center)
        self.assertFalse(ui.is_collapsed)

        # Stepper increment click
        base_stepper = ui.steppers[0]
        curr_plates = self.generator.base_num_plates
        handled = base_stepper.handle_click(base_stepper.btn_inc_rect.center)
        self.assertTrue(handled)
        self.assertEqual(self.generator.base_num_plates, curr_plates + 1)

        # Stepper decrement click
        handled_dec = base_stepper.handle_click(base_stepper.btn_dec_rect.center)
        self.assertTrue(handled_dec)
        self.assertEqual(self.generator.base_num_plates, curr_plates)

        # Toggle click
        first_toggle = ui.toggles[0]
        state = first_toggle.get_state()
        handled_t = first_toggle.handle_click(first_toggle.rect.center)
        self.assertTrue(handled_t)
        self.assertNotEqual(first_toggle.get_state(), state)

        # Render pass
        self.viewer.render()

    def test_adaptive_lod_color_accuracy(self):
        # Create a mesh with higher subdivision (e.g. 5, 20480 faces) and compare with LOD (subdiv 4, 5120 faces)
        mesh = SphereMesh(radius=1.0, subdivisions=5)
        gen = PlateGenerator(mesh=mesh, base_num_plates=16)
        gen.generate()

        viewer = GlobeViewer(mesh=mesh, width=800, height=600)
        viewer.set_plate_generator(gen)

        viewer.use_adaptive_lod = True
        viewer._update_render_mesh()
        self.assertEqual(viewer.render_subdiv, 4)

        # Check that unique colors in LOD mesh match the plates generated
        lod_unique_colors = len(np.unique(viewer.render_face_colors, axis=0))
        highres_colors = gen.get_face_colors()
        highres_unique_colors = len(np.unique(highres_colors, axis=0))

        # LOD should preserve all (or almost all) distinct plate colors
        self.assertGreaterEqual(lod_unique_colors, highres_unique_colors - 1)

        # Verify child face alignment dot products
        offsets = viewer._get_lod_descendant_offsets(4, 5)
        self.assertEqual(len(offsets), 4)

        v4, f4 = mesh.lod_meshes[4]
        v5, f5 = mesh.lod_meshes[5]
        sample_idx = 42
        c_lod = np.mean(v4[f4[sample_idx]], axis=0)
        c_lod /= np.linalg.norm(c_lod)

        for off in offsets:
            c_child = np.mean(v5[f5[sample_idx + off]], axis=0)
            c_child /= np.linalg.norm(c_child)
            # Alignment must be > 0.999 (true geometric descendants)
            self.assertGreater(np.dot(c_lod, c_child), 0.999)


if __name__ == "__main__":
    unittest.main()
