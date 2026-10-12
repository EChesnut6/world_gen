import numpy as np


class Camera3D:
    """
    Orbit camera that handles 3D position, rotation (yaw, pitch), zoom,
    and perspective projection onto a 2D screen.
    """

    def __init__(self, width: int = 1000, height: int = 800, fov: float = 60.0):
        self.width = width
        self.height = height
        self.fov = fov
        
        # Camera orbit angles (radians) and distance
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 3.5
        
        # Invert mouse drag settings (Default True as requested)
        self.invert_x = True
        self.invert_y = True
        
        # Minimum and maximum distance limits
        self.min_distance = 1.2
        self.max_distance = 10.0

    def rotate(self, dx: float, dy: float, sensitivity: float = 0.005):
        """Rotate the camera around the origin with customizable direction inversion."""
        dir_x = -1.0 if self.invert_x else 1.0
        dir_y = -1.0 if self.invert_y else 1.0

        self.yaw += dx * sensitivity * dir_x
        # Clamp pitch to prevent flipping
        self.pitch = np.clip(self.pitch + dy * sensitivity * dir_y, -np.pi / 2.1, np.pi / 2.1)

    def zoom(self, delta: float, speed: float = 0.3):
        """Zoom camera in or out."""
        self.distance = np.clip(self.distance - delta * speed, self.min_distance, self.max_distance)

    def get_rotation_matrix(self) -> np.ndarray:
        """Calculate camera transformation rotation matrix."""
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)

        # Yaw around Y axis, Pitch around X axis
        R_y = np.array([
            [cos_y, 0, sin_y],
            [0,     1, 0],
            [-sin_y, 0, cos_y]
        ])

        R_x = np.array([
            [1, 0,      0],
            [0, cos_p, -sin_p],
            [0, sin_p,  cos_p]
        ])

        return R_x @ R_y

    def project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Projects 3D world points to 2D screen coordinates.
        Returns:
            screen_coords: (N, 2) array of (x, y) pixel coordinates
            z_depths: (N,) array of Z distance values for depth ordering
        """
        R = self.get_rotation_matrix()
        
        # Rotate points
        rotated = points @ R.T
        
        # Translate back along Z axis for camera view distance
        view_pts = rotated.copy()
        view_pts[:, 2] += self.distance
        
        # Avoid division by zero/negative Z for points behind camera
        z = np.maximum(view_pts[:, 2], 0.01)
        
        # Perspective projection
        focal_length = self.height / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        
        x_proj = (view_pts[:, 0] * focal_length) / z + (self.width / 2.0)
        y_proj = (-view_pts[:, 1] * focal_length) / z + (self.height / 2.0)
        
        screen_coords = np.column_stack((x_proj, y_proj))
        return screen_coords, z
