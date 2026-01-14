"""
Basic H2Matrix Example
======================

This example demonstrates the basic usage of H2Pack for creating
and using hierarchical matrix representations with various kernel functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack

print("=" * 70)
print("H2Pack Basic Example: Multiple Kernels and Dimensions")
print("=" * 70)

# Step 1: Generate random points in 3D space
print("\n1. Generating test points...")
n_points = 5000
dim = 3
points = np.random.rand(n_points, dim)
print(f"   Created {n_points} random points in {dim}D space")

# Step 2: Create H² matrix with Gaussian kernel
print("\n2. Creating H² matrix with Gaussian kernel...")
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

# Step 6: Demonstrate all available kernels in 3D
print("\n" + "=" * 70)
print("6. Testing All Available Kernels (3D)")
print("=" * 70)

kernels_3d = [
    ('gaussian', {'lengthscale': 1.0}),
    ('matern32', {'lengthscale': 1.0}),
    ('matern52', {'lengthscale': 1.0}),
    ('exponential', {'lengthscale': 1.0}),
    ('coulomb', {'epsilon': 0.01}),
    ('quadratic', {'c': 1.0, 'a': -0.5}),
]

# Use smaller dataset for testing multiple kernels
points_small = np.random.rand(1000, 3)
x_small = np.random.randn(1000)

for kernel_name, kernel_params in kernels_3d:
    try:
        H_test = h2pack.H2Matrix(points_small, kernel=kernel_name,
                                  kernel_params=kernel_params, rel_tol=1e-6)
        H_test.build()
        y_test = H_test.matvec(x_small)
        stats = H_test.stats
        print(f"   ✓ {kernel_name:12s}: max_rank={stats['max_rank']:3d}, "
              f"compression={stats['compression_ratio']:.2f}x, "
              f"||y||={np.linalg.norm(y_test):.2e}")
    except Exception as e:
        print(f"   ✗ {kernel_name:12s}: {e}")

# Step 7: Demonstrate 2D kernels
print("\n" + "=" * 70)
print("7. Testing Kernels in 2D")
print("=" * 70)

points_2d = np.random.rand(1000, 2)
x_2d = np.random.randn(1000)

kernels_2d = [
    ('gaussian', {'lengthscale': 1.0}),
    ('matern32', {'lengthscale': 1.0}),
    ('matern52', {'lengthscale': 1.0}),
    ('exponential', {'lengthscale': 1.0}),
    ('coulomb', {'epsilon': 0.01}),  # Uses Laplace_2D internally
    ('quadratic', {'c': 1.0, 'a': -0.5}),
]

for kernel_name, kernel_params in kernels_2d:
    try:
        H_test = h2pack.H2Matrix(points_2d, kernel=kernel_name,
                                  kernel_params=kernel_params, rel_tol=1e-6)
        H_test.build()
        y_test = H_test.matvec(x_2d)
        stats = H_test.stats
        print(f"   ✓ {kernel_name:12s}: max_rank={stats['max_rank']:3d}, "
              f"compression={stats['compression_ratio']:.2f}x, "
              f"||y||={np.linalg.norm(y_test):.2e}")
    except Exception as e:
        print(f"   ✗ {kernel_name:12s}: {e}")

# Step 8: Demonstrate 1D kernels
print("\n" + "=" * 70)
print("8. Testing Kernels in 1D")
print("=" * 70)

points_1d = np.random.rand(1000, 1)
x_1d = np.random.randn(1000)

kernels_1d = [
    ('gaussian', {'lengthscale': 1.0}),
    ('matern32', {'lengthscale': 1.0}),
    ('matern52', {'lengthscale': 1.0}),
    ('exponential', {'lengthscale': 1.0}),
    ('quadratic', {'c': 1.0, 'a': -0.5}),
    # Note: Coulomb not supported in 1D
]

for kernel_name, kernel_params in kernels_1d:
    try:
        H_test = h2pack.H2Matrix(points_1d, kernel=kernel_name,
                                  kernel_params=kernel_params, rel_tol=1e-6)
        H_test.build()
        y_test = H_test.matvec(x_1d)
        stats = H_test.stats
        print(f"   ✓ {kernel_name:12s}: max_rank={stats['max_rank']:3d}, "
              f"compression={stats['compression_ratio']:.2f}x, "
              f"||y||={np.linalg.norm(y_test):.2e}")
    except Exception as e:
        print(f"   ✗ {kernel_name:12s}: {e}")

# Step 9: Demonstrate using kernel objects
print("\n" + "=" * 70)
print("9. Using Kernel Objects Directly")
print("=" * 70)

kernel_objects = [
    h2pack.GaussianKernel(lengthscale=1.5),
    h2pack.MaternKernel(lengthscale=2.0, nu=1.5),
    h2pack.ExponentialKernel(lengthscale=1.0),
    h2pack.CoulombKernel(epsilon=0.01),
    h2pack.QuadraticKernel(c=1.0, a=-0.5),
]

for kernel_obj in kernel_objects:
    try:
        H_test = h2pack.H2Matrix(points_small, kernel=kernel_obj, rel_tol=1e-6)
        H_test.build()
        y_test = H_test.matvec(x_small)
        stats = H_test.stats
        print(f"   ✓ {kernel_obj.name:12s}: max_rank={stats['max_rank']:3d}, "
              f"compression={stats['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"   ✗ {kernel_obj.name:12s}: {e}")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("✓ H² matrix construction")
print("✓ Fast matrix-vector multiplication")
print("✓ All supported kernel functions (Gaussian, Matern, Exponential, Coulomb, Quadratic)")
print("✓ 1D, 2D, and 3D support")
print("✓ Kernel objects and string-based kernel selection")
print("✓ Compression statistics")
print("=" * 70)
