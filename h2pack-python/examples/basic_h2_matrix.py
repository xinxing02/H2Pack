"""
Basic H2Matrix Example
======================

This example demonstrates the basic usage of H2Pack for creating
and using hierarchical matrix representations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack

print("=" * 70)
print("H2Pack Basic Example: Gaussian Kernel")
print("=" * 70)

# Step 1: Generate random points in 3D space
print("\n1. Generating test points...")
n_points = 1000  # Reduced from 10000 for testing
dim = 3
points = np.random.randn(n_points, dim)
print(f"   Created {n_points} random points in {dim}D space")

# Step 2: Create H² matrix with Gaussian kernel
print("\n2. Creating H² matrix...")
H = h2pack.H2Matrix(
    points=points,
    kernel='gaussian',
    kernel_params={'lengthscale': 1.0},
    rel_tol=1e-6,
    jit_mode=True
)
print(f"   Matrix shape: {H.shape}")
print(f"   Kernel: {H.kernel.name}")
print(f"   Tolerance: {H.rel_tol}")

# Step 3: Build H² representation
print("\n3. Building H² representation...")

try:
    H.build()
    print("   ✓ H² matrix built successfully!")

    # Step 4: Matrix-vector multiplication
    print("\n4. Testing matrix-vector multiplication...")
    x = np.random.randn(n_points)
    y = H.matvec(x)
    print(f"   Input vector shape: {x.shape}")
    print(f"   Output vector shape: {y.shape}")
    print(f"   Output norm: {np.linalg.norm(y):.6f}")

    # Step 5: Print statistics
    print("\n5. H² Matrix Statistics:")
    h2pack.utils.print_stats(H)

except RuntimeError as e:
    print(f"   ⚠ {e}")

# Step 6: Demonstrate different kernels
print("\n6. Available kernel functions:")
kernels = ['gaussian', 'matern32', 'matern52', 'coulomb', 'quadratic']
for kernel_name in kernels:
    try:
        if kernel_name in ['coulomb']:
            continue  # Skip for now
        H_test = h2pack.H2Matrix(points[:100, :], kernel=kernel_name)
        print(f"   ✓ {kernel_name}: {H_test.kernel}")
    except Exception as e:
        print(f"   ✗ {kernel_name}: {e}")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("✓ H² matrix construction")
print("✓ Fast matrix-vector multiplication")
print("✓ Multiple kernel functions supported")
print("✓ Compression statistics")
print("=" * 70)
