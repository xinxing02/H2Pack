"""
Utility functions for H2Pack.

This module provides helper functions for generating test data,
benchmarking, and visualization.
"""

import numpy as np
from typing import Tuple, Literal


def generate_points(
    n_points: int,
    dim: int = 3,
    distribution: Literal['uniform', 'gaussian', 'sphere', 'ball'] = 'uniform',
    seed: int = None
) -> np.ndarray:
    """
    Generate test point sets.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    dim : int, default=3
        Dimension of the space (1, 2, or 3).
    distribution : {'uniform', 'gaussian', 'sphere', 'ball'}, default='uniform'
        Distribution of points:
        - 'uniform': Uniform in [0, 1]^d
        - 'gaussian': Standard normal distribution
        - 'sphere': Uniform on unit sphere (3D only)
        - 'ball': Uniform in unit ball
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray, shape (n_points, dim)
        Generated point coordinates.

    Examples
    --------
    >>> points = generate_points(1000, dim=3, distribution='gaussian')
    >>> points.shape
    (1000, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    if distribution == 'uniform':
        points = np.random.rand(n_points, dim)

    elif distribution == 'gaussian':
        points = np.random.randn(n_points, dim)

    elif distribution == 'sphere':
        if dim != 3:
            raise ValueError("Sphere distribution only available for 3D")
        # Generate points on unit sphere
        points = np.random.randn(n_points, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    elif distribution == 'ball':
        # Generate points in unit ball
        points = np.random.randn(n_points, dim)
        r = np.random.rand(n_points) ** (1.0 / dim)
        points *= r[:, np.newaxis] / np.linalg.norm(points, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return points


def direct_matvec(
    points: np.ndarray,
    x: np.ndarray,
    kernel: str = 'gaussian',
    **kernel_params
) -> np.ndarray:
    """
    Direct matrix-vector multiplication (for validation/testing).

    Computes y = K * x where K is the dense kernel matrix.
    WARNING: This is O(N²) in memory and computation.

    Parameters
    ----------
    points : ndarray, shape (n_points, dim)
        Point coordinates.
    x : ndarray, shape (n_points,)
        Input vector.
    kernel : str, default='gaussian'
        Kernel function name.
    **kernel_params
        Additional kernel parameters.

    Returns
    -------
    y : ndarray, shape (n_points,)
        Result of K * x.

    Notes
    -----
    For Coulomb kernel:
    - In 2D: Uses Laplace kernel K(x,y) = -log(r) with r = ||x-y||
    - In 3D: Uses Coulomb kernel K(x,y) = 1/r
    """
    n = len(points)
    dim = points.shape[1]

    if n > 10000:
        raise ValueError(
            f"Direct matvec too expensive for {n} points. "
            "Use only for small test cases."
        )

    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)

    # Compute kernel matrix
    if kernel == 'gaussian':
        l = kernel_params.get('lengthscale', 1.0)
        K = np.exp(-dist_sq / (2 * l ** 2))
    elif kernel == 'matern32':
        l = kernel_params.get('lengthscale', 1.0)
        dist = np.sqrt(dist_sq + 1e-16)
        scaled_dist = np.sqrt(3) * dist / l
        K = (1 + scaled_dist) * np.exp(-scaled_dist)
    elif kernel == 'matern52':
        l = kernel_params.get('lengthscale', 1.0)
        dist = np.sqrt(dist_sq + 1e-16)
        scaled_dist = np.sqrt(5) * dist / l
        K = (1 + scaled_dist + scaled_dist ** 2 / 3) * np.exp(-scaled_dist)
    elif kernel == 'exponential':
        l = kernel_params.get('lengthscale', 1.0)
        dist = np.sqrt(dist_sq + 1e-16)
        K = np.exp(-dist / l)
    elif kernel == 'coulomb':
        epsilon = kernel_params.get('epsilon', 0.01)
        if dim == 2:
            # 2D Laplace kernel: K(x,y) = -0.5 * log(r²) = -log(r)
            # Add small value to avoid log(0)
            K = -0.5 * np.log(dist_sq + 1e-16)
            np.fill_diagonal(K, epsilon)
        else:
            # 3D Coulomb kernel: K(x,y) = 1/r
            dist = np.sqrt(dist_sq + 1e-16)
            K = 1.0 / dist
            np.fill_diagonal(K, epsilon)
    elif kernel == 'quadratic':
        c = kernel_params.get('c', 1.0)
        a = kernel_params.get('a', -0.5)
        K = (1 + c * dist_sq) ** a
    else:
        raise ValueError(f"Direct matvec not implemented for kernel: {kernel}")

    return K @ x


def estimate_accuracy(
    h2matrix,
    n_test_vectors: int = 10,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Estimate H² matrix accuracy via random test vectors.

    Parameters
    ----------
    h2matrix : H2Matrix
        Built H² matrix to test.
    n_test_vectors : int, default=10
        Number of random test vectors.
    seed : int, default=42
        Random seed.

    Returns
    -------
    rel_error : float
        Average relative error.
    max_error : float
        Maximum relative error.
    """
    if not h2matrix.is_built:
        raise ValueError("H2 matrix must be built first")

    if h2matrix.n_points > 5000:
        raise ValueError(
            "Accuracy estimation too expensive for large matrices. "
            f"Got {h2matrix.n_points} points, limit is 5000."
        )

    np.random.seed(seed)
    errors = []

    for _ in range(n_test_vectors):
        x = np.random.randn(h2matrix.n_points)

        # H² matvec
        y_h2 = h2matrix.matvec(x)

        # Direct matvec
        kernel_name = h2matrix.kernel.name.lower()
        y_exact = direct_matvec(
            h2matrix.points,
            x,
            kernel=kernel_name,
            **h2matrix.kernel.params
        )

        # Relative error
        rel_err = np.linalg.norm(y_h2 - y_exact) / np.linalg.norm(y_exact)
        errors.append(rel_err)

    return np.mean(errors), np.max(errors)


def benchmark_scaling(
    sizes: list,
    dim: int = 3,
    kernel: str = 'gaussian',
    rel_tol: float = 1e-6,
    n_trials: int = 3
) -> dict:
    """
    Benchmark H² matrix scaling with problem size.

    Parameters
    ----------
    sizes : list of int
        Problem sizes to test.
    dim : int, default=3
        Point dimension.
    kernel : str, default='gaussian'
        Kernel function.
    rel_tol : float, default=1e-6
        H² tolerance.
    n_trials : int, default=3
        Number of trials per size.

    Returns
    -------
    results : dict
        Benchmark results with keys:
        - 'sizes': Problem sizes
        - 'build_times': Construction times
        - 'matvec_times': Matvec times
        - 'memory_mb': Memory usage
    """
    import time
    from .core import H2Matrix

    results = {
        'sizes': sizes,
        'build_times': [],
        'matvec_times': [],
        'memory_mb': []
    }

    for n in sizes:
        print(f"Benchmarking n={n}...")

        build_times = []
        matvec_times = []

        for trial in range(n_trials):
            # Generate points
            points = generate_points(n, dim=dim, seed=trial)

            # Build H² matrix
            H = H2Matrix(points, kernel=kernel, rel_tol=rel_tol)
            t0 = time.time()
            H.build()
            build_time = time.time() - t0
            build_times.append(build_time)

            # Matvec benchmark
            x = np.random.randn(n)
            times = []
            for _ in range(10):
                t0 = time.time()
                y = H.matvec(x)
                times.append(time.time() - t0)
            matvec_times.append(np.median(times))

        results['build_times'].append(np.median(build_times))
        results['matvec_times'].append(np.median(matvec_times))

        # Get memory usage from stats
        if hasattr(H, 'stats'):
            stats = H.stats
            results['memory_mb'].append(stats.get('storage_mb', 0))
        else:
            results['memory_mb'].append(0)

    return results


def print_stats(h2matrix) -> None:
    """
    Print formatted statistics for an H² matrix.

    Parameters
    ----------
    h2matrix : H2Matrix or HSSMatrix
        Matrix to print stats for.
    """
    if not h2matrix.is_built:
        print(f"{h2matrix.__class__.__name__} (not built)")
        return

    stats = h2matrix.stats

    print("=" * 60)
    print(f"{h2matrix.__class__.__name__} Statistics")
    print("=" * 60)
    print(f"Number of points        : {stats.get('n_points', 'N/A')}")
    print(f"Dimension               : {h2matrix.dim}")
    print(f"Kernel function         : {h2matrix.kernel.name}")
    print(f"Relative tolerance      : {h2matrix.rel_tol:.2e}")
    print("-" * 60)
    print(f"Number of tree levels   : {stats.get('n_levels', 'N/A')}")
    print(f"Total tree nodes        : {stats.get('n_nodes', 'N/A')}")
    avg_rank = stats.get('avg_rank', 'N/A')
    if isinstance(avg_rank, (int, float)):
        print(f"Average rank            : {avg_rank:.2f}")
    else:
        print(f"Average rank            : {avg_rank}")
    print(f"Maximum rank            : {stats.get('max_rank', 'N/A')}")
    print("-" * 60)
    storage_mb = stats.get('storage_mb', 'N/A')
    if isinstance(storage_mb, (int, float)):
        print(f"Storage (MB)            : {storage_mb:.2f}")
    else:
        print(f"Storage (MB)            : {storage_mb}")
    compression = stats.get('compression_ratio', 'N/A')
    if isinstance(compression, (int, float)):
        print(f"Compression ratio       : {compression:.1f}x")
    else:
        print(f"Compression ratio       : {compression}")
    print("=" * 60)
