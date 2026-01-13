"""
Accuracy test for H2Pack with actual compression (5000 points)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os

# Set thread limits
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
            # Standard Gaussian: exp(-r² / (2*l²))
            K[i, j] = np.exp(-r_sq / (2 * lengthscale**2))

    return K

# Test parameters
n_points = 5000
lengthscale = 1.0
rel_tol = 1e-6

print("="*70)
print("H2 COMPRESSION ACCURACY TEST: Gaussian Kernel (5000 points)")
print("="*70)
print(f"Points: {n_points}")
print(f"Lengthscale: {lengthscale}")
print(f"Relative tolerance: {rel_tol:.2e}")
print()

# Generate points
np.random.seed(42)
points = np.random.randn(n_points, 3)

# Build H2 matrix
print("Building H2 matrix...")
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    kernel_params={'lengthscale': lengthscale},
    rel_tol=rel_tol,
    n_threads=4
)
H.build()
stats = H.stats
print(f"H2 structure: {stats['n_levels']} levels, {stats['n_nodes']} nodes")
print(f"Max rank: {stats['max_rank']}, Avg rank: {stats.get('avg_rank', 0):.1f}")
print(f"Compression: {stats.get('compression_ratio', 0):.2f}x")
print()

# Test matvec accuracy with random vectors
print("Testing matvec accuracy (full problem)...")
print("-" * 70)

# Use smaller sample for dense computation
sample_size = 500
sample_indices = np.random.choice(n_points, sample_size, replace=False)
sample_indices = np.sort(sample_indices)  # Sort for easier debugging

print(f"Computing dense kernel for {sample_size}-point sample...")
sample_points = points[sample_indices]
K_dense_sample = compute_gaussian_kernel(sample_points, lengthscale)
print(f"Dense sample matrix: {K_dense_sample.shape}, norm = {np.linalg.norm(K_dense_sample):.6f}")
print()

errors = []
n_test_vectors = 10

print(f"Testing with {n_test_vectors} random vectors...")
for i in range(n_test_vectors):
    # Full-size test vector
    x_full = np.random.randn(n_points)

    # H2 matvec (full problem)
    y_h2_full = H.matvec(x_full)

    # Dense matvec (sample only)
    x_sample = x_full[sample_indices]
    y_dense_sample = K_dense_sample @ x_sample
    y_h2_sample = y_h2_full[sample_indices]

    # Compute error
    norm_dense = np.linalg.norm(y_dense_sample)
    rel_error = np.linalg.norm(y_h2_sample - y_dense_sample) / norm_dense
    errors.append(rel_error)

    if i < 3:  # Print details for first 3 vectors
        print(f"  Vector {i+1}:")
        print(f"    Dense norm: {norm_dense:.6f}")
        print(f"    H2 norm:    {np.linalg.norm(y_h2_sample):.6f}")
        print(f"    Rel error:  {rel_error:.6e}")

if n_test_vectors > 3:
    print(f"  ... ({n_test_vectors - 3} more vectors tested)")

errors = np.array(errors)
avg_error = np.mean(errors)
max_error = np.max(errors)
min_error = np.min(errors)

print()
print("="*70)
print("ACCURACY RESULTS")
print("="*70)
print(f"Average relative error: {avg_error:.6e}")
print(f"Maximum relative error: {max_error:.6e}")
print(f"Minimum relative error: {min_error:.6e}")
print(f"Target tolerance:       {rel_tol:.6e}")
print()

# Determine pass/fail (allow 10x tolerance for H2 approximation)
tolerance_factor = 10
if avg_error < rel_tol * tolerance_factor:
    print(f"✓ PASS: Error within {tolerance_factor}x tolerance")
    print(f"  Error/Tolerance ratio: {avg_error/rel_tol:.2f}x")
elif avg_error < rel_tol * 100:
    print(f"⚠ MARGINAL: Error within 100x tolerance but exceeds {tolerance_factor}x")
    print(f"  Error/Tolerance ratio: {avg_error/rel_tol:.1f}x")
else:
    print(f"✗ FAIL: Error exceeds {tolerance_factor}x tolerance!")
    print(f"  Error/Tolerance ratio: {avg_error/rel_tol:.1f}x")
    print()
    print("DEBUGGING INFO:")
    print("-" * 70)

    # Test a single diagnostic vector
    x_test = np.ones(n_points)
    y_h2_test = H.matvec(x_test)
    y_dense_test = K_dense_sample @ np.ones(sample_size)
    y_h2_test_sample = y_h2_test[sample_indices]

    print(f"Diagnostic (x=ones vector):")
    print(f"  Dense result (sample): first 5 = {y_dense_test[:5]}")
    print(f"  H2 result (sample):    first 5 = {y_h2_test_sample[:5]}")
    print(f"  Ratio: {np.mean(y_h2_test_sample / y_dense_test):.6f}")

print("="*70)
