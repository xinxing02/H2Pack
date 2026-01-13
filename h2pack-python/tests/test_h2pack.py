"""
Unit tests for h2pack package.

These tests verify the Python API layer functionality.
The C extension tests will be added in Phase 2.
"""

import pytest
import numpy as np
from h2pack import H2Matrix, HSSMatrix
from h2pack.kernels import GaussianKernel, MaternKernel
from h2pack.utils import generate_points


class TestPointGeneration:
    """Test point generation utilities."""

    def test_uniform_points(self):
        points = generate_points(100, dim=3, distribution='uniform', seed=42)
        assert points.shape == (100, 3)
        assert np.all(points >= 0) and np.all(points <= 1)

    def test_gaussian_points(self):
        points = generate_points(100, dim=2, distribution='gaussian', seed=42)
        assert points.shape == (100, 2)

    def test_sphere_points(self):
        points = generate_points(100, dim=3, distribution='sphere', seed=42)
        assert points.shape == (100, 3)
        # Check points are on unit sphere
        norms = np.linalg.norm(points, axis=1)
        assert np.allclose(norms, 1.0)

    def test_ball_points(self):
        points = generate_points(100, dim=3, distribution='ball', seed=42)
        assert points.shape == (100, 3)
        # Check points are in unit ball
        norms = np.linalg.norm(points, axis=1)
        assert np.all(norms <= 1.0)


class TestKernels:
    """Test kernel classes."""

    def test_gaussian_kernel(self):
        kernel = GaussianKernel(lengthscale=2.0)
        assert kernel.name == "Gaussian"
        assert kernel.lengthscale == 2.0
        assert kernel.params['lengthscale'] == 2.0

    def test_matern_kernel(self):
        kernel = MaternKernel(lengthscale=1.5, nu=1.5)
        assert kernel.name == "Matern"
        assert kernel.nu == 1.5
        assert kernel.kernel_type == "Matern32"

    def test_matern_kernel_52(self):
        kernel = MaternKernel(lengthscale=1.0, nu=2.5)
        assert kernel.kernel_type == "Matern52"

    def test_invalid_matern_nu(self):
        with pytest.raises(ValueError):
            MaternKernel(nu=3.5)


class TestH2Matrix:
    """Test H2Matrix class."""

    def test_initialization(self):
        points = np.random.randn(100, 3)
        H = H2Matrix(points, kernel='gaussian', rel_tol=1e-6)
        assert H.n_points == 100
        assert H.dim == 3
        assert H.shape == (100, 100)
        assert not H.is_built

    def test_kernel_from_string(self):
        points = np.random.randn(50, 2)
        H = H2Matrix(points, kernel='matern32')
        assert H.kernel.name == "Matern"

    def test_kernel_from_object(self):
        points = np.random.randn(50, 2)
        kernel = GaussianKernel(lengthscale=3.0)
        H = H2Matrix(points, kernel=kernel)
        assert H.kernel.lengthscale == 3.0

    def test_invalid_points_shape(self):
        with pytest.raises(ValueError):
            points = np.random.randn(100)  # 1D array
            H2Matrix(points)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            points = np.random.randn(100, 5)  # 5D not supported
            H2Matrix(points)

    def test_repr(self):
        points = np.random.randn(100, 3)
        H = H2Matrix(points, kernel='gaussian')
        repr_str = repr(H)
        assert 'H2Matrix' in repr_str
        assert 'n_points=100' in repr_str
        assert 'not built' in repr_str


class TestHSSMatrix:
    """Test HSSMatrix class."""

    def test_initialization(self):
        points = np.random.randn(100, 2)
        H = HSSMatrix(points, kernel='gaussian')
        assert H.n_points == 100
        assert H.dim == 2
        assert not H.is_built
        assert not H.is_factorized

    def test_repr(self):
        points = np.random.randn(50, 2)
        H = HSSMatrix(points)
        repr_str = repr(H)
        assert 'HSSMatrix' in repr_str
        assert 'not built' in repr_str


class TestIntegration:
    """Integration tests (will be more comprehensive when C extension is ready)."""

    def test_create_multiple_matrices(self):
        """Test creating multiple H2Matrix objects."""
        points1 = np.random.randn(100, 2)
        points2 = np.random.randn(200, 3)

        H1 = H2Matrix(points1, kernel='gaussian')
        H2 = H2Matrix(points2, kernel='matern32')

        assert H1.n_points == 100
        assert H2.n_points == 200
        assert H1.dim == 2
        assert H2.dim == 3


# TODO: Add tests for C extension functionality in Phase 2
# class TestMatvec:
#     """Test matrix-vector multiplication."""
#     pass
#
# class TestAccuracy:
#     """Test H² matrix accuracy."""
#     pass
