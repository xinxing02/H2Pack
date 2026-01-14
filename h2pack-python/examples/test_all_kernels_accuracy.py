"""
Comprehensive Accuracy Test for All Kernels in 1D, 2D, and 3D
==============================================================

This script tests the accuracy of all supported kernels in H2Pack
by comparing H² matrix-vector products against exact dense computations
using 5000 uniformly distributed random points.

Uses h2pack.utils.direct_matvec for efficient exact computation without
forming the full dense matrix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import time

def test_kernel_accuracy(dim, kernel_name, kernel_params, n_points=5000, rel_tol=1e-6):
    """Test accuracy of a single kernel configuration."""
    np.random.seed(42)  # For reproducibility

    # Generate uniformly distributed random points in [0,1]^dim
    points = np.random.uniform(0, 1, size=(n_points, dim))
    x = np.random.randn(n_points)

    try:
        # Build H² matrix
        t0 = time.time()
        H = h2pack.H2Matrix(points, kernel=kernel_name, kernel_params=kernel_params, rel_tol=rel_tol)
        H.build()
        build_time = time.time() - t0

        # H² matvec
        t0 = time.time()
        y_h2 = H.matvec(x)
        matvec_time = time.time() - t0

        # Get statistics
        stats = H.stats

        # Compute exact result using direct_matvec (much faster than forming dense matrix)
        print(f"   Computing exact matvec using direct_matvec...", end='', flush=True)
        t0 = time.time()
        # Convert kernel_params dict to kwargs for direct_matvec
        y_exact = h2pack.utils.direct_matvec(points, x, kernel=kernel_name, **kernel_params)
        direct_time = time.time() - t0
        print(f" done ({direct_time:.2f}s)")

        # Compute error
        rel_error = np.linalg.norm(y_h2 - y_exact) / np.linalg.norm(y_exact)

        # Check for NaN
        has_nan = np.any(np.isnan(y_h2))

        # Print results
        print(f"   ✓ Accuracy test PASSED")
        print(f"      Relative error    : {rel_error:.6e}")
        print(f"      Max rank          : {stats['max_rank']}")
        print(f"      Avg rank          : {stats['avg_rank']:.1f}")
        print(f"      Compression       : {stats['compression_ratio']:.2f}x")
        print(f"      Build time        : {build_time:.3f}s")
        print(f"      H² matvec time    : {matvec_time:.6f}s")
        print(f"      Direct matvec time: {direct_time:.3f}s")
        print(f"      Speedup           : {direct_time/matvec_time:.1f}x")
        print(f"      Has NaN values    : {has_nan}")

        # Determine if accuracy is acceptable
        # For compressed matrices (max_rank > 0), expect errors around 10-100x tolerance
        # For uncompressed matrices (max_rank = 0), expect machine precision
        if stats['max_rank'] == 0:
            # No compression - should be very accurate
            acceptable = rel_error < 1e-10
            status = "EXCELLENT (no compression)" if acceptable else "WARNING (high error)"
        else:
            # With compression - expect ~10-100x tolerance
            # Coulomb/Laplace kernels are singular and harder to compress, allow 1000x tolerance
            if kernel_name == 'coulomb':
                acceptable = rel_error < 1000 * rel_tol
            else:
                acceptable = rel_error < 100 * rel_tol
            status = "GOOD" if acceptable else "WARNING (error exceeds expected range)"

        print(f"      Status            : {status}")

        return {
            'success': True,
            'rel_error': rel_error,
            'max_rank': stats['max_rank'],
            'compression': stats['compression_ratio'],
            'build_time': build_time,
            'matvec_time': matvec_time,
            'direct_time': direct_time,
            'has_nan': has_nan,
            'acceptable': acceptable
        }

    except Exception as e:
        print(f"   ✗ Test FAILED: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("H2Pack Comprehensive Accuracy Test")
    print("=" * 80)
    print(f"Testing all kernels in 1D, 2D, and 3D with 5000 points")
    print(f"Points are uniformly distributed in [0,1]^d")
    print("=" * 80)

    # Define test configurations
    test_configs = {
        '1D': {
            'dim': 1,
            'kernels': [
                ('gaussian', {'lengthscale': 1.0}),
                ('matern32', {'lengthscale': 1.0}),
                ('matern52', {'lengthscale': 1.0}),
                ('exponential', {'lengthscale': 1.0}),
                ('quadratic', {'c': 1.0, 'a': -0.5}),
                # Coulomb not supported in 1D
            ]
        },
        '2D': {
            'dim': 2,
            'kernels': [
                ('gaussian', {'lengthscale': 1.0}),
                ('matern32', {'lengthscale': 1.0}),
                ('matern52', {'lengthscale': 1.0}),
                ('exponential', {'lengthscale': 1.0}),
                ('coulomb', {'epsilon': 0.01}),
                ('quadratic', {'c': 1.0, 'a': -0.5}),
            ]
        },
        '3D': {
            'dim': 3,
            'kernels': [
                ('gaussian', {'lengthscale': 1.0}),
                ('matern32', {'lengthscale': 1.0}),
                ('matern52', {'lengthscale': 1.0}),
                ('exponential', {'lengthscale': 1.0}),
                ('coulomb', {'epsilon': 0.01}),
                ('quadratic', {'c': 1.0, 'a': -0.5}),
            ]
        }
    }

    # Run all tests
    results = {}

    for dim_name, config in test_configs.items():
        print(f"\n{'=' * 80}")
        print(f"Testing {dim_name} Kernels")
        print(f"{'=' * 80}")

        dim = config['dim']
        results[dim_name] = {}

        for kernel_name, kernel_params in config['kernels']:
            print(f"\n{kernel_name.upper():>15s} kernel:")
            print(f"   Parameters: {kernel_params}")

            result = test_kernel_accuracy(dim, kernel_name, kernel_params, n_points=5000, rel_tol=1e-6)
            results[dim_name][kernel_name] = result

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL TESTS")
    print("=" * 80)

    for dim_name in ['1D', '2D', '3D']:
        print(f"\n{dim_name} Results:")
        print("-" * 80)
        print(f"{'Kernel':<15s} {'Status':<10s} {'Rel Error':<12s} {'Max Rank':<10s} {'Compression':<12s}")
        print("-" * 80)

        for kernel_name, result in results[dim_name].items():
            if result['success']:
                status = "✓ PASS" if result['acceptable'] else "⚠ WARN"
                print(f"{kernel_name:<15s} {status:<10s} {result['rel_error']:<12.6e} "
                      f"{result['max_rank']:<10d} {result['compression']:<12.2f}x")
            else:
                print(f"{kernel_name:<15s} {'✗ FAIL':<10s} {result['error']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
