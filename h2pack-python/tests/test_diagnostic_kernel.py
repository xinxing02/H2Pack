"""
Minimal diagnostic test to check kernel evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Test with 2 known points
points = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]], dtype=np.float64)

lengthscale = 1.0

print("="*70)
print("DIAGNOSTIC: Kernel Evaluation with Known Points")
print("="*70)
print(f"Point 0: {points[0]}")
print(f"Point 1: {points[1]}")
print(f"Distance: {np.linalg.norm(points[1] - points[0]):.6f}")
print(f"Lengthscale: {lengthscale}")
print()

# Expected kernel values
r_sq = 1.0  # distance squared
expected_k_01 = np.exp(-r_sq / (2 * lengthscale**2))
expected_k_00 = 1.0

print("EXPECTED VALUES (Python formula):")
print(f"  K[0,0] (diagonal): {expected_k_00:.10f}")
print(f"  K[0,1] (r=1.0):    {expected_k_01:.10f}")
print()

# Build H2 matrix
print("Building H2 matrix...")
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    kernel_params={'lengthscale': lengthscale},
    rel_tol=1e-12,  # Very tight tolerance
    n_threads=1
)
H.build()

stats = H.stats
print(f"H2 structure: {stats['n_levels']} levels, {stats['n_nodes']} nodes")
print(f"Max rank: {stats['max_rank']}")
print()

# Test with unit vectors to extract matrix columns
print("H2 MATRIX VALUES:")

# Extract K[*, 0] (first column)
x0 = np.array([1.0, 0.0])
y0 = H.matvec(x0)
print(f"  K[0,0] = {y0[0]:.10f} (expected {expected_k_00:.10f})")
print(f"  K[1,0] = {y0[1]:.10f} (expected {expected_k_01:.10f})")

# Extract K[*, 1] (second column)
x1 = np.array([0.0, 1.0])
y1 = H.matvec(x1)
print(f"  K[0,1] = {y1[0]:.10f} (expected {expected_k_01:.10f})")
print(f"  K[1,1] = {y1[1]:.10f} (expected {expected_k_00:.10f})")

print()

# Check errors
err_00 = abs(y0[0] - expected_k_00)
err_01 = abs(y0[1] - expected_k_01)
err_10 = abs(y1[0] - expected_k_01)
err_11 = abs(y1[1] - expected_k_00)

print("ERRORS:")
print(f"  K[0,0]: {err_00:.2e}")
print(f"  K[0,1]: {err_01:.2e}")
print(f"  K[1,0]: {err_10:.2e}")
print(f"  K[1,1]: {err_11:.2e}")
print()

max_error = max(err_00, err_01, err_10, err_11)
if max_error < 1e-10:
    print("✓ PASS: Kernel evaluation is correct!")
else:
    print(f"✗ FAIL: Maximum error = {max_error:.2e}")
    print()
    print("RATIOS (H2 / Expected):")
    print(f"  K[0,0]: {y0[0] / expected_k_00:.6e}")
    print(f"  K[0,1]: {y0[1] / expected_k_01:.6e}")
    print(f"  K[1,0]: {y1[0] / expected_k_01:.6e}")
    print(f"  K[1,1]: {y1[1] / expected_k_00:.6e}")

print("="*70)
