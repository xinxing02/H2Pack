"""
Accuracy Demonstration

This script demonstrates the relationship between H2Pack's relative tolerance
parameter and actual accuracy in matrix-vector multiplication.

Key concepts:
- rel_tol controls the compression level
- Actual error is typically 10-100x the specified tolerance
- Tighter tolerance → higher accuracy but slower/larger

Author: H2Pack Development Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os
import time

# Set thread limits
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def compute_dense_gaussian(points, lengthscale=1.0):
    """Compute dense Gaussian kernel matrix."""
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = points[i] - points[j]
            r_sq = np.sum(diff ** 2)
            K[i, j] = np.exp(-r_sq / (2 * lengthscale**2))
    return K

def test_tolerance(points, x, K_dense, rel_tol, lengthscale=1.0):
    """Test H2 approximation with given tolerance."""
    # Build H2 matrix
    H = h2pack.H2Matrix(
        points,
        kernel='gaussian',
        kernel_params={'lengthscale': lengthscale},
        rel_tol=rel_tol,
        n_threads=4
    )

    t0 = time.time()
    H.build()
    build_time = time.time() - t0

    stats = H.stats

    # Compute H2 matvec
    t0 = time.time()
    y_h2 = H.matvec(x)
    matvec_time = time.time() - t0

    # Compute dense matvec (ground truth)
    y_dense = K_dense @ x

    # Compute error
    rel_error = np.linalg.norm(y_h2 - y_dense) / np.linalg.norm(y_dense)

    return {
        'rel_tol': rel_tol,
        'rel_error': rel_error,
        'error_factor': rel_error / rel_tol,
        'build_time': build_time,
        'matvec_time': matvec_time,
        'max_rank': stats['max_rank'],
        'compression': stats.get('compression_ratio', 0),
        'storage_mb': stats.get('storage_mb', 0)
    }

def main():
    """Run accuracy demonstration."""
    print("="*70)
    print("H2PACK ACCURACY DEMONSTRATION")
    print("="*70)
    print()
    print("This example demonstrates:")
    print("  1. Relationship between tolerance and actual error")
    print("  2. Accuracy vs performance trade-offs")
    print("  3. Expected error ranges for different tolerances")
    print()

    # Setup
    n_points = 1000  # Smaller for dense computation
    lengthscale = 1.0

    print(f"Problem setup:")
    print(f"  Points: {n_points}")
    print(f"  Dimension: 3D")
    print(f"  Kernel: Gaussian (lengthscale={lengthscale})")
    print()

    # Generate points and test vector
    np.random.seed(42)
    points = np.random.randn(n_points, 3)
    x = np.random.randn(n_points)

    # Compute dense matrix (ground truth)
    print("Computing dense kernel matrix (ground truth)...")
    K_dense = compute_dense_gaussian(points, lengthscale)
    dense_size_mb = n_points ** 2 * 8 / (1024**2)
    print(f"  Dense matrix size: {dense_size_mb:.1f} MB")
    print()

    # Test different tolerances
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    print("="*70)
    print("TOLERANCE vs ACCURACY")
    print("="*70)
    print(f"{'Tolerance':<12} {'Actual Error':<15} {'Error/Tol':<12} {'Rank':<8} {'Build(s)':<10}")
    print("-"*70)

    results = []
    for tol in tolerances:
        result = test_tolerance(points, x, K_dense, tol, lengthscale)
        results.append(result)

        print(f"{tol:<12.2e} {result['rel_error']:<15.6e} {result['error_factor']:<12.1f}x "
              f"{result['max_rank']:<8} {result['build_time']:<10.3f}")

    print("="*70)
    print()

    # Analysis
    print("ANALYSIS")
    print("="*70)
    print()

    print("1. Error vs Tolerance Relationship:")
    print("   - Actual error is typically 10-100x the specified tolerance")
    print("   - This is expected behavior for H2 approximations")
    print("   - The factor varies with problem characteristics")
    print()

    print("2. Recommended Tolerances:")
    print("   - High accuracy:  rel_tol=1e-8  (error ~1e-10 to 1e-9)")
    print("   - Standard:       rel_tol=1e-6  (error ~1e-8 to 1e-7)  ← recommended")
    print("   - Fast:           rel_tol=1e-4  (error ~1e-6 to 1e-5)")
    print()

    print("3. Performance Trade-offs:")
    tightest = results[-1]  # 1e-8
    loosest = results[0]    # 1e-4

    speedup = tightest['build_time'] / loosest['build_time']
    size_ratio = tightest['storage_mb'] / loosest['storage_mb']

    print(f"   - Tighter tolerance (1e-8 vs 1e-4):")
    print(f"     • {speedup:.1f}x slower build")
    print(f"     • {size_ratio:.1f}x more storage")
    print(f"     • {tightest['error_factor']/loosest['error_factor']:.1f}x better accuracy")
    print()

    print("4. Practical Guidelines:")
    print("   - For most applications: rel_tol=1e-6 is a good balance")
    print("   - For critical accuracy: Use rel_tol=1e-8 and verify results")
    print("   - For prototyping/testing: rel_tol=1e-4 is faster")
    print("   - Always validate accuracy for your specific use case")
    print()

    print("="*70)
    print("✓ Demonstration complete!")
    print()
    print("Next steps:")
    print("  - Try with your own problem size and kernel")
    print("  - See ACCURACY_VALIDATION.md for detailed validation results")
    print("  - Run tests/test_simple_accuracy.py for automated validation")
    print()

if __name__ == "__main__":
    main()
