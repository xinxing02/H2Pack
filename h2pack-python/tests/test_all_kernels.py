"""
Quick validation of all supported kernels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def compute_kernel_matrix(points, kernel, **params):
    """Compute dense kernel matrix for validation."""
    n = len(points)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff = points[i] - points[j]
            r_sq = np.sum(diff ** 2)
            r = np.sqrt(r_sq)

            if kernel == 'gaussian':
                l = params.get('lengthscale', 1.0)
                K[i, j] = np.exp(-r_sq / (2 * l**2))

            elif kernel == 'matern32':
                l = params.get('lengthscale', 1.0)
                scaled_r = np.sqrt(3) * r / l
                K[i, j] = (1 + scaled_r) * np.exp(-scaled_r)

            elif kernel == 'matern52':
                l = params.get('lengthscale', 1.0)
                scaled_r = np.sqrt(5) * r / l
                K[i, j] = (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)

            elif kernel == 'coulomb':
                if r > 1e-10:
                    K[i, j] = 1.0 / r
                else:
                    K[i, j] = 0.0

            elif kernel == 'quadratic':
                c = params.get('c', 1.0)
                a = params.get('a', -0.5)
                K[i, j] = (c**2 + r_sq)**a

    return K

def test_kernel(kernel_name, n_points=1000, **kernel_params):
    """Test a kernel with given parameters."""
    print(f"\nTesting {kernel_name.upper()} kernel...")
    print(f"  Parameters: {kernel_params if kernel_params else 'default'}")

    # Generate points
    np.random.seed(42)
    points = np.random.randn(n_points, 3)

    # Build H2 matrix
    try:
        H = h2pack.H2Matrix(
            points,
            kernel=kernel_name,
            kernel_params=kernel_params if kernel_params else None,
            rel_tol=1e-6,
            n_threads=4
        )
        H.build()
    except Exception as e:
        print(f"  ✗ FAILED to build: {e}")
        return False

    stats = H.stats

    # Compute dense matrix
    K_dense = compute_kernel_matrix(points, kernel_name, **kernel_params)

    # Test matvec
    x = np.random.randn(n_points)
    y_dense = K_dense @ x
    y_h2 = H.matvec(x)

    rel_error = np.linalg.norm(y_h2 - y_dense) / np.linalg.norm(y_dense)

    # Report results
    print(f"  Structure: {stats['n_levels']} levels, rank={stats['max_rank']}")
    print(f"  Accuracy: {rel_error:.6e}")

    # Pass if error is reasonable
    if stats['max_rank'] == 0:
        passed = rel_error < 1e-10  # Machine precision
    else:
        passed = rel_error < 1e-4  # 100x tolerance

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Status: {status}")

    return passed

print("="*70)
print("KERNEL VALIDATION - ALL KERNELS")
print("="*70)

results = {}

# Gaussian (already validated, quick check)
results['Gaussian'] = test_kernel('gaussian', n_points=1000, lengthscale=1.0)

# Matern 3/2
results['Matern32'] = test_kernel('matern32', n_points=1000, lengthscale=1.0)

# Matern 5/2
results['Matern52'] = test_kernel('matern52', n_points=1000, lengthscale=1.0)

# Coulomb
results['Coulomb'] = test_kernel('coulomb', n_points=1000)

# Quadratic
results['Quadratic'] = test_kernel('quadratic', n_points=1000)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for kernel, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {kernel:<15} {status}")

n_passed = sum(1 for p in results.values() if p)
n_total = len(results)

print("="*70)
print(f"Overall: {n_passed}/{n_total} kernels passed")
if n_passed == n_total:
    print("✅ ALL KERNELS VALIDATED")
else:
    print(f"⚠ {n_total - n_passed} kernels failed")

print("="*70)
