"""
Simple accuracy test for H2Pack Gaussian kernel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack

# Set thread limits
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
            # Standard Gaussian: exp(-r² / (2*l²))
            K[i, j] = np.exp(-r_sq / (2 * lengthscale**2))

    return K

# Test parameters
n_points = 100
lengthscale = 1.0
rel_tol = 1e-6

print("="*70)
print("SIMPLE ACCURACY TEST: Gaussian Kernel")
print("="*70)
print(f"Points: {n_points}")
print(f"Lengthscale: {lengthscale}")
print(f"Relative tolerance: {rel_tol:.2e}")
print()

# Generate points
np.random.seed(42)
points = np.random.randn(n_points, 3)

# Compute dense kernel matrix
print("Computing dense kernel matrix...")
K_dense = compute_gaussian_kernel(points, lengthscale)
print(f"Dense matrix: {K_dense.shape}, norm = {np.linalg.norm(K_dense):.6f}")
print(f"Sample K[0,0] = {K_dense[0,0]:.6f} (should be 1.0)")
print(f"Sample K[0,1] = {K_dense[0,1]:.6f}")
print()

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
print(f"Max rank: {stats['max_rank']}")
print()

# Test 1: Full matvec accuracy with random vectors
print("Test 1: Matvec accuracy with random vectors")
print("-" * 70)

errors = []
for i in range(5):
    x = np.random.randn(n_points)

    # Dense matvec
    y_dense = K_dense @ x

    # H2 matvec
    y_h2 = H.matvec(x)

    # Compute error
    rel_error = np.linalg.norm(y_h2 - y_dense) / np.linalg.norm(y_dense)
    errors.append(rel_error)
    print(f"  Vector {i+1}: rel_error = {rel_error:.6e}")

avg_error = np.mean(errors)
max_error = np.max(errors)

print(f"\nAverage error: {avg_error:.6e}")
print(f"Maximum error: {max_error:.6e}")
print(f"Target tolerance: {rel_tol:.6e}")

# Check if error is within acceptable bounds (10x tolerance)
if avg_error < 10 * rel_tol:
    print(f"\n✓ PASS: Error within 10x tolerance")
else:
    print(f"\n✗ FAIL: Error {avg_error/rel_tol:.1f}x tolerance!")
    print(f"\nDEBUG INFO:")
    print(f"  Dense result norm: {np.linalg.norm(y_dense):.6f}")
    print(f"  H2 result norm: {np.linalg.norm(y_h2):.6f}")
    print(f"  First 5 dense values: {y_dense[:5]}")
    print(f"  First 5 H2 values: {y_h2[:5]}")

# Test 2: Diagonal accuracy
print("\n" + "="*70)
print("Test 2: Diagonal accuracy (should be exactly 1.0)")
print("-" * 70)

diagonal_errors = []
for i in range(min(10, n_points)):
    # Create unit vector
    x = np.zeros(n_points)
    x[i] = 1.0

    # Compute y = K * x (should give i-th column of K)
    y_h2 = H.matvec(x)
    y_dense = K_dense @ x

    # Check diagonal entry
    dense_diag = y_dense[i]
    h2_diag = y_h2[i]
    error = abs(h2_diag - dense_diag)
    diagonal_errors.append(error)

    print(f"  Point {i}: dense={dense_diag:.6f}, h2={h2_diag:.6f}, error={error:.6e}")

avg_diag_error = np.mean(diagonal_errors)
print(f"\nAverage diagonal error: {avg_diag_error:.6e}")

if avg_diag_error < 1e-10:
    print("✓ PASS: Diagonal is exact")
else:
    print(f"✗ WARNING: Diagonal has error {avg_diag_error:.6e}")

print("\n" + "="*70)
