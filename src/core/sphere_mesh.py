import numpy as np


class SphereMesh:
    """
    Generates and holds 3D mesh data for spherical planets.
    Supports fast vectorized Icosphere subdivision for uniform vertex density across the globe.
    Precomputes face normals, centers, edge adjacency, and LOD meshes for high-performance rendering.
    """

    def __init__(self, radius: float = 1.0, subdivisions: int = 3):
        self.radius = radius
        self.subdivisions = subdivisions
        
        self.lod_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.vertices, self.faces = self._generate_icosphere(subdivisions)
        
        # Precompute face centers, face normals, and edge adjacency
        self._compute_face_geometry()
        self._compute_adjacency()

    def _compute_face_geometry(self):
        """Precomputes face center positions and unit normal vectors."""
        face_verts = self.vertices[self.faces]  # Shape: (F, 3, 3)
        self.face_centers = np.mean(face_verts, axis=1)  # Shape: (F, 3)
        
        e1 = face_verts[:, 1, :] - face_verts[:, 0, :]  # (F, 3)
        e2 = face_verts[:, 2, :] - face_verts[:, 0, :]  # (F, 3)
        
        normals = np.cross(e1, e2)  # (F, 3)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.face_normals = normals / norms

    def _compute_adjacency(self):
        """Computes edge-to-face adjacency map for boundary analysis using vectorized numpy operations."""
        num_faces = len(self.faces)
        face_indices = np.arange(num_faces, dtype=np.int32)

        # 3 edges per triangle: (v0, v1), (v1, v2), (v2, v0)
        e0 = self.faces[:, [0, 1]]
        e1 = self.faces[:, [1, 2]]
        e2 = self.faces[:, [2, 0]]

        all_edges = np.vstack([e0, e1, e2])
        all_faces = np.tile(face_indices, 3)
        all_edges_sorted = np.sort(all_edges, axis=1)

        # Combine 32-bit vertex indices into a 64-bit edge key for fast hashing
        edge_keys = (all_edges_sorted[:, 0].astype(np.int64) << 32) | all_edges_sorted[:, 1].astype(np.int64)

        sort_order = np.argsort(edge_keys)
        sorted_keys = edge_keys[sort_order]
        sorted_faces = all_faces[sort_order]
        sorted_edges = all_edges_sorted[sort_order]

        self.edge_to_faces = {}
        self.face_neighbors = [[] for _ in range(num_faces)]

        same_mask = np.diff(sorted_keys) == 0
        idx0 = np.where(same_mask)[0]

        for idx in idx0:
            f1 = int(sorted_faces[idx])
            f2 = int(sorted_faces[idx + 1])
            edge = (int(sorted_edges[idx, 0]), int(sorted_edges[idx, 1]))

            self.edge_to_faces[edge] = [f1, f2]
            self.face_neighbors[f1].append((f2, edge))
            self.face_neighbors[f2].append((f1, edge))

    def _generate_icosphere(self, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized Icosphere subdivision with LOD mesh hierarchy cache."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        raw_verts = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
        ], dtype=np.float64)

        verts = raw_verts / np.linalg.norm(raw_verts, axis=1, keepdims=True)
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)

        self.lod_meshes[0] = (verts.copy() * self.radius, faces.copy())

        for level in range(1, subdivisions + 1):
            e0 = faces[:, [0, 1]]
            e1 = faces[:, [1, 2]]
            e2 = faces[:, [2, 0]]
            all_edges = np.vstack([e0, e1, e2])
            sorted_edges = np.sort(all_edges, axis=1)

            unique_edges, inv = np.unique(sorted_edges, axis=0, return_inverse=True)

            p1 = verts[unique_edges[:, 0]]
            p2 = verts[unique_edges[:, 1]]
            mids = (p1 + p2) * 0.5
            mids /= np.linalg.norm(mids, axis=1, keepdims=True)

            num_old_verts = len(verts)
            verts = np.vstack([verts, mids])

            num_faces = len(faces)
            a_idx = num_old_verts + inv[0:num_faces]
            b_idx = num_old_verts + inv[num_faces:2*num_faces]
            c_idx = num_old_verts + inv[2*num_faces:3*num_faces]

            v1, v2, v3 = faces[:, 0], faces[:, 1], faces[:, 2]

            f1 = np.column_stack([v1, a_idx, c_idx])
            f2 = np.column_stack([v2, b_idx, a_idx])
            f3 = np.column_stack([v3, c_idx, b_idx])
            f4 = np.column_stack([a_idx, b_idx, c_idx])

            faces = np.vstack([f1, f2, f3, f4])
            self.lod_meshes[level] = (verts.copy() * self.radius, faces.copy())

        final_verts = verts * self.radius
        return final_verts, faces
