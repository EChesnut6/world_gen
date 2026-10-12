from src.core.sphere_mesh import SphereMesh
from src.plates.plate_gen import PlateGenerator
from src.viewer.globe_viewer import GlobeViewer


def main():
    print("Initializing World Gen 3D Globe & Tectonic Plate Simulation...")
    
    # 1. Create spherical mesh (subdivisions=4 gives smooth surface with 5120 faces)
    mesh = SphereMesh(radius=1.0, subdivisions=6)
    print(f"Mesh generated with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    # 2. Generate initial Tectonic Plates & Boundary Analysis
    base_num_plates = 16
    generator = PlateGenerator(mesh=mesh, base_num_plates=base_num_plates)
    generator.generate()
    print(f"Generated {base_num_plates} base tectonic plates with boundary stress classification.")

    # 3. Initialize 3D Viewer & assign plate generator
    viewer = GlobeViewer(mesh=mesh, width=1024, height=768)
    viewer.set_plate_generator(generator)

    # 4. Start Viewer Loop
    print("\nControls & Live Interface:")
    print("  [Mouse Click UI] Adjust Live Key Stats, Toggles, and Actions directly on screen")
    print("  [Mouse Drag] Rotate Globe (outside UI)")
    print("  [Mouse Scroll] Zoom In / Out")
    print("  [Simulation HUD] Play, pause, step forward/back, reset time, and adjust playback speed")
    print("  [H] Toggle / Collapse Controls HUD")
    print("  [G] Generate New Base Plates")
    print("  [F] Fractures High-Stress Collision Zone (+1 Microplate)")
    print("  [Z] Reset Microplate Fractures")
    print("  [B] Toggle Boundary Lines (Red=Convergent, Blue=Divergent, Yellow=Transform)")
    print("  [D] Toggle Plate Drift Arrows")
    print("  [W] Toggle Wireframe")
    print("  [S] Toggle Shading")
    print("  [O] Toggle RGB Orientation Axes")
    print("  [L] Toggle Adaptive LOD Rendering")
    print("  [I] Toggle Inverted Mouse Controls")
    print("  [SPACE] Toggle Auto Rotation")
    print("  [R] Reset Camera")

    viewer.run()


if __name__ == "__main__":
    main()
