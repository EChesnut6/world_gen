"""Reusable seeded 3D simplex and fractal Brownian motion noise."""

import numpy as np


class SimplexNoise3D:
    """Vectorized 3D simplex noise suitable for sampling spherical surfaces."""

    _GRADIENTS = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ], dtype=np.float64)

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(256)
        self._permutation = np.concatenate((permutation, permutation))

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Sample points with shape (..., 3), returning values approximately in [-1, 1]."""
        original_shape = points.shape[:-1]
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)

        f3 = 1.0 / 3.0
        g3 = 1.0 / 6.0
        skew = np.sum(xyz, axis=1, keepdims=True) * f3
        lattice = np.floor(xyz + skew).astype(np.int64)
        unskew = np.sum(lattice, axis=1, keepdims=True) * g3
        x0 = xyz - (lattice - unskew)

        x_ge_y = x0[:, 0] >= x0[:, 1]
        y_ge_z = x0[:, 1] >= x0[:, 2]
        x_ge_z = x0[:, 0] >= x0[:, 2]
        i1 = np.zeros((len(xyz), 3), dtype=np.int64)
        i2 = np.zeros((len(xyz), 3), dtype=np.int64)

        case = np.select(
            [x_ge_y & y_ge_z, x_ge_y & ~y_ge_z & x_ge_z, x_ge_y & ~y_ge_z & ~x_ge_z,
             ~x_ge_y & ~y_ge_z, ~x_ge_y & y_ge_z & ~x_ge_z, ~x_ge_y & y_ge_z & x_ge_z],
            [0, 1, 2, 3, 4, 5],
            default=0,
        )
        corner_steps = (
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 0, 1), (1, 0, 1)),
            ((0, 0, 1), (0, 1, 1)),
            ((0, 1, 0), (0, 1, 1)),
            ((0, 1, 0), (1, 1, 0)),
        )
        for case_id, (first, second) in enumerate(corner_steps):
            mask = case == case_id
            i1[mask] = first
            i2[mask] = second

        offsets = (
            x0,
            x0 - i1 + g3,
            x0 - i2 + 2.0 * g3,
            x0 - 1.0 + 3.0 * g3,
        )
        lattice_corners = (lattice, lattice + i1, lattice + i2, lattice + 1)
        permutation = self._permutation
        result = np.zeros(len(xyz), dtype=np.float64)

        for offset, corner in zip(offsets, lattice_corners):
            attenuation = 0.6 - np.sum(offset * offset, axis=1)
            active = attenuation > 0
            if not np.any(active):
                continue
            corner = corner[active] & 255
            hashed = permutation[
                corner[:, 0] + permutation[corner[:, 1] + permutation[corner[:, 2]]]
            ] % 12
            dot = np.einsum("ij,ij->i", self._GRADIENTS[hashed], offset[active])
            attenuation = attenuation[active]
            result[active] += attenuation**4 * dot

        return (32.0 * result).reshape(original_shape)

    def fractal(
        self,
        points: np.ndarray,
        *,
        scale: float = 1.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        """Sample normalized fractal Brownian motion over 3D points."""
        if octaves < 1:
            raise ValueError("octaves must be at least 1")
        coordinates = np.asarray(points, dtype=np.float64)
        result = np.zeros(coordinates.shape[:-1], dtype=np.float64)
        amplitude = 1.0
        total_amplitude = 0.0
        frequency = scale
        for _ in range(octaves):
            result += amplitude * self.sample(coordinates * frequency)
            total_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return result / total_amplitude
