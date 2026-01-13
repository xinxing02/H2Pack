"""
Test to find the problem size where H2 compression breaks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def compute_gaussian_kernel(points, lengthscale=1.0):
    """Compute dense Gaussian kernel matrix."""
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = points[i] - points[j]
            r_sq = np.sum(diff ** 2)
            K[i, j] = np.exp(-r_sq / (2 * lengthscale**2))
    return K

def test_size(n_points, lengthscale=1.0, rel_tol=1e-6):
    """Test a specific problem size."""
    np.random.seed(42)
    points = np.random.randn(n_points, 3)

    # Build H2 matrix
    H = h2pack.H2Matrix(
        points,
        kernel='gaussian',
        kernel_params={'lengthscale': lengthscale},
        rel_tol=rel_tol,
        n_threads=4
    )
    H.build()
    stats = H.stats

    # Test matvec accuracy
    K_dense = compute_gaussian_kernel(points, lengthscale)
    x = np.random.randn(n_points)
    y_dense = K_dense @ x
    y_h2 = H.matvec(x)

    rel_error = np.linalg.norm(y_h2 - y_dense) / np.linalg.norm(y_dense)

    return stats['max_rank'], rel_error

print("="*70)
print("FINDING COMPRESSION THRESHOLD")
print("="*70)

sizes = [10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000, 5000]

print(f"{'Size':<8} {'Max Rank':<12} {'Rel Error':<15} {'Status'}")
print("-"*70)

for n in sizes:
    try:
        max_rank, error = test_size(n)
        status = "✓ PASS" if error < 1e-3 else "✗ FAIL"

        if max_rank == 0:
            rank_str = "0 (no comp)"
        else:
            rank_str = str(max_rank)

        print(f"{n:<8} {rank_str:<12} {error:<15.6e} {status}")

        # Stop if we found where it breaks
        if max_rank > 0 and error > 0.1:
            print()
            print(f"⚠ COMPRESSION BREAKS AT SIZE {n}")
            print(f"  Max rank: {max_rank}")
            print(f"  Error: {error:.6e}")
            break

    except Exception as e:
        print(f"{n:<8} ERROR: {e}")
        break

print("="*70)
