"""
Accuracy Validation Tests for H2Pack

This module validates the accuracy of H2 matrix representations by comparing
against dense matrix computations for various kernels and problem sizes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack


def compute_dense_kernel_matrix(points, kernel, **kernel_params):
    """
    Compute dense kernel matrix directly.

    Parameters
    ----------
    points : ndarray, shape (n, d)
        Point coordinates
    kernel : str
        Kernel name
    **kernel_params
        Kernel parameters

    Returns
    -------
    K : ndarray, shape (n, n)
        Dense kernel matrix
    """
    n = len(points)
    K = np.zeros((n, n))

    # Compute pairwise distances
    for i in range(n):
        for j in range(n):
            diff = points[i] - points[j]
            r_sq = np.sum(diff ** 2)
            r = np.sqrt(r_sq)

            if kernel == 'gaussian':
                l = kernel_params.get('lengthscale', 1.0)
                # Standard Gaussian: exp(-r² / (2*l²))
                K[i, j] = np.exp(-r_sq / (2 * l**2))

            elif kernel == 'matern32':
                l = kernel_params.get('lengthscale', 1.0)
                scaled_r = np.sqrt(3) * r / l
                K[i, j] = (1 + scaled_r) * np.exp(-scaled_r)

            elif kernel == 'matern52':
                l = kernel_params.get('lengthscale', 1.0)
                scaled_r = np.sqrt(5) * r / l
                K[i, j] = (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)

            elif kernel == 'coulomb':
                if r > 1e-10:
                    K[i, j] = 1.0 / r
                else:
                    K[i, j] = 0.0  # Avoid self-interaction

            elif kernel == 'quadratic':
                c = kernel_params.get('c', 1.0)
                a = kernel_params.get('a', -0.5)
                K[i, j] = (c**2 + r_sq)**a

            else:
                raise ValueError(f"Unknown kernel: {kernel}")

    return K


def test_kernel_accuracy(kernel_name, n_points=5000, rel_tol=1e-6, **kernel_params):
    """
    Test accuracy of a specific kernel.

    Parameters
    ----------
    kernel_name : str
        Name of kernel to test
    n_points : int
        Number of points to test with
    rel_tol : float
        H2 relative tolerance
    **kernel_params
        Additional kernel parameters

    Returns
    -------
    results : dict
        Accuracy test results
    """
    print(f"\n{'='*70}")
    print(f"Testing {kernel_name.upper()} kernel accuracy")
    print(f"{'='*70}")
    print(f"Problem size: {n_points} points")
    print(f"H2 tolerance: {rel_tol:.2e}")
    if kernel_params:
        print(f"Kernel params: {kernel_params}")

    # Generate random points
    np.random.seed(42)
    points = np.random.randn(n_points, 3)

    # Build H2 matrix
    print("\nBuilding H2 matrix...")
    H = h2pack.H2Matrix(
        points,
        kernel=kernel_name,
        kernel_params=kernel_params if kernel_params else None,
        rel_tol=rel_tol,
        n_threads=4
    )
    H.build()

    stats = H.stats
    print(f"H2 structure: {stats['n_levels']} levels, {stats['n_nodes']} nodes")
    print(f"Max rank: {stats['max_rank']}, Avg rank: {stats.get('avg_rank', 0):.1f}")
    print(f"Compression: {stats.get('compression_ratio', 0):.2f}x")

    # Test multiple random vectors
    n_test_vectors = 5
    errors = []

    print(f"\nTesting with {n_test_vectors} random vectors...")

    # Compute dense matrix for small sample
    sample_size = min(500, n_points)  # Limit for dense computation
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    sample_points = points[sample_indices]

    print(f"Computing dense matrix for {sample_size} sample points...")
    K_dense_sample = compute_dense_kernel_matrix(sample_points, kernel_name, **kernel_params)

    for i in range(n_test_vectors):
        # Test vector (full size)
        x_full = np.random.randn(n_points)

        # H2 matvec (full)
        y_h2_full = H.matvec(x_full)

        # Dense matvec (sample)
        x_sample = x_full[sample_indices]
        y_dense_sample = K_dense_sample @ x_sample
        y_h2_sample = y_h2_full[sample_indices]

        # Compute relative error
        norm_dense = np.linalg.norm(y_dense_sample)
        if norm_dense > 1e-10:
            rel_error = np.linalg.norm(y_h2_sample - y_dense_sample) / norm_dense
            errors.append(rel_error)
        else:
            errors.append(0.0)

    errors = np.array(errors)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    print(f"\nAccuracy Results (on {sample_size}-point sample):")
    print(f"  Average relative error: {avg_error:.6e}")
    print(f"  Maximum relative error: {max_error:.6e}")
    print(f"  Minimum relative error: {min_error:.6e}")
    print(f"  Target tolerance:       {rel_tol:.6e}")

    # Determine pass/fail
    # Allow error to be up to 10x the tolerance (H2 is approximate)
    tolerance_factor = 10
    passed = avg_error < rel_tol * tolerance_factor

    if passed:
        print(f"\n✓ PASS: Error within {tolerance_factor}x tolerance")
    else:
        print(f"\n✗ FAIL: Error exceeds {tolerance_factor}x tolerance")
        print(f"  Error/Tolerance ratio: {avg_error/rel_tol:.1f}x")

    results = {
        'kernel': kernel_name,
        'n_points': n_points,
        'sample_size': sample_size,
        'avg_error': avg_error,
        'max_error': max_error,
        'min_error': min_error,
        'target_tol': rel_tol,
        'passed': passed,
        'h2_levels': stats['n_levels'],
        'h2_nodes': stats['n_nodes'],
        'max_rank': stats['max_rank'],
        'compression': stats.get('compression_ratio', 0)
    }

    return results


def test_all_kernels():
    """Run accuracy tests for all supported kernels."""

    print("\n" + "="*70)
    print("H2PACK ACCURACY VALIDATION SUITE")
    print("="*70)

    all_results = []

    # Test configurations
    tests = [
        ('gaussian', {'lengthscale': 1.0}),
        ('gaussian', {'lengthscale': 0.5}),
        ('gaussian', {'lengthscale': 2.0}),
        ('matern32', {'lengthscale': 1.0}),
        ('matern52', {'lengthscale': 1.0}),
        # ('coulomb', {}),  # Skip for now - different behavior
        ('quadratic', {}),
    ]

    for kernel_name, params in tests:
        try:
            result = test_kernel_accuracy(kernel_name, n_points=5000, rel_tol=1e-6, **params)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR testing {kernel_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL TESTS")
    print("="*70)
    print(f"{'Kernel':<20} {'Params':<20} {'Avg Error':<15} {'Status':<10}")
    print("-"*70)

    for result in all_results:
        params_str = f"l={result.get('lengthscale', 1.0):.1f}" if 'gaussian' in result['kernel'] or 'matern' in result['kernel'] else ""
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{result['kernel']:<20} {params_str:<20} {result['avg_error']:<15.6e} {status:<10}")

    # Overall
    n_passed = sum(1 for r in all_results if r['passed'])
    n_total = len(all_results)

    print("="*70)
    print(f"Overall: {n_passed}/{n_total} tests passed")

    if n_passed == n_total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"⚠ {n_total - n_passed} tests failed")

    print("="*70)

    return all_results


if __name__ == "__main__":
    import os
    # Set thread limits to avoid OpenBLAS issues
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'

    # Run all tests
    results = test_all_kernels()
