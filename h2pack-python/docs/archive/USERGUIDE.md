# H2Pack User Guide

**Version**: 1.0.0-beta
**Date**: January 2026

This guide provides comprehensive instructions for using H2Pack to create and work with hierarchical (H²) matrix representations of dense kernel matrices.

---

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)
3. [Kernel Selection](#kernel-selection)
4. [Parameter Tuning](#parameter-tuning)
5. [Accuracy Expectations](#accuracy-expectations)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Installation

### Prerequisites

- **Python**: 3.8 or newer
- **NumPy**: 1.20.0 or newer
- **C Compiler**: gcc, clang, or MSVC
- **BLAS/LAPACK**: OpenBLAS or MKL with LAPACKE interface

### Platform-Specific Instructions

#### macOS (Apple Silicon or Intel)

```bash
# Install OpenBLAS
brew install openblas

# Clone and install H2Pack
git clone https://github.com/scalable-matrix/H2Pack.git
cd H2Pack/h2pack-python
pip install -e .
```

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev build-essential python3-dev

# Fedora/RHEL
sudo dnf install openblas-devel gcc python3-devel

# Install H2Pack
git clone https://github.com/scalable-matrix/H2Pack.git
cd H2Pack/h2pack-python
pip install -e .
```

#### Windows

```bash
# Install Visual Studio Build Tools
# Install OpenBLAS from conda or vcpkg

git clone https://github.com/scalable-matrix/H2Pack.git
cd H2Pack/h2pack-python
pip install -e .
```

### Verify Installation

```python
import h2pack
import numpy as np

# Create a simple test
points = np.random.randn(100, 3)
H = h2pack.H2Matrix(points, kernel='gaussian')
H.build()
print("H2Pack installed successfully!")
```

---

## First Steps

### Basic Workflow

The typical H2Pack workflow consists of 4 steps:

```python
import h2pack
import numpy as np

# 1. Generate or load point coordinates
points = np.random.randn(5000, 3)  # 5000 points in 3D

# 2. Create H2Matrix object
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    kernel_params={'lengthscale': 1.0},
    rel_tol=1e-6,
    n_threads=4
)

# 3. Build H² representation
H.build()

# 4. Use for matrix-vector multiplication
x = np.random.randn(5000)
y = H.matvec(x)  # y = K * x
```

### Understanding the Output

After building, check the statistics:

```python
stats = H.stats
print(f"Compression: {stats['compression_ratio']:.1f}x")
print(f"Max rank: {stats['max_rank']}")
print(f"Levels: {stats['n_levels']}")
print(f"Storage: {stats['storage_mb']:.1f} MB")
```

**What these mean**:
- **Compression ratio**: How much smaller H² is vs dense matrix
- **Max rank**: Maximum rank of low-rank blocks (higher = less compression)
- **Levels**: Depth of hierarchical tree
- **Storage**: Memory used by H² representation

---

## Kernel Selection

### Available Kernels

#### 1. Gaussian (RBF) Kernel ✅ **Recommended**

**Formula**: `K(x,y) = exp(-||x-y||² / (2*lengthscale²))`

**Usage**:
```python
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    kernel_params={'lengthscale': 1.0}
)
```

**When to use**:
- General-purpose kernel
- Smooth, infinitely differentiable functions
- Machine learning applications (GP regression, SVM)

**Lengthscale guide**:
- **Small (0.1-0.5)**: Short-range interactions, high compression
- **Medium (0.5-2.0)**: Balanced, recommended for most cases
- **Large (> 2.0)**: Long-range interactions, low compression

#### 2. Matern 3/2 Kernel ✅

**Formula**: `K(x,y) = (1 + √3*r/l) * exp(-√3*r/l)` where `r = ||x-y||`

**Usage**:
```python
H = h2pack.H2Matrix(
    points,
    kernel='matern32',
    kernel_params={'lengthscale': 1.0}
)
```

**When to use**:
- Once-differentiable functions
- More flexible than Gaussian
- Common in spatial statistics

#### 3. Matern 5/2 Kernel ✅

**Formula**: `K(x,y) = (1 + √5*r/l + 5r²/3l²) * exp(-√5*r/l)`

**Usage**:
```python
H = h2pack.H2Matrix(
    points,
    kernel='matern52',
    kernel_params={'lengthscale': 1.0}
)
```

**When to use**:
- Twice-differentiable functions
- Smoother than Matern 3/2 but less smooth than Gaussian
- Geological and environmental modeling

#### 4. Coulomb Kernel ⚠️

**Formula**: `K(x,y) = 1 / ||x-y||`

**Usage**:
```python
H = h2pack.H2Matrix(points, kernel='coulomb')
```

**When to use**:
- Electrostatics problems
- N-body simulations
- Particle physics

**Note**: Basic support, may need parameter tuning.

#### 5. Quadratic Kernel ⚠️

**Formula**: `K(x,y) = (c² + ||x-y||²)^a`

**Usage**:
```python
H = h2pack.H2Matrix(
    points,
    kernel='quadratic',
    kernel_params={'c': 1.0, 'a': -0.5}
)
```

**When to use**:
- Inverse quadratic kernels
- Custom power-law relationships

**Note**: Basic support, may need parameter tuning.

### Kernel Comparison

| Kernel | Smoothness | Compression | Speed | Use Case |
|--------|------------|-------------|-------|----------|
| Gaussian | ∞ (very smooth) | Good | Fast | General purpose |
| Matern 5/2 | C² (smooth) | Good | Fast | Environmental |
| Matern 3/2 | C¹ (less smooth) | Better | Fast | Spatial stats |
| Coulomb | Singular | Excellent | Medium | Physics |
| Quadratic | Varies | Good | Medium | Custom |

---

## Parameter Tuning

### Relative Tolerance (`rel_tol`)

The most important parameter controlling accuracy vs performance.

**Recommended values**:
```python
# High accuracy (slower, larger)
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-8)

# Standard (balanced) - RECOMMENDED
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-6)

# Fast (lower accuracy)
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-4)
```

**Guidelines**:
- Start with `1e-6` for most applications
- Use `1e-8` if you need high accuracy (scientific computing)
- Use `1e-4` for prototyping or when speed is critical
- **Remember**: Actual error is typically 10-100x the tolerance

### Maximum Leaf Points (`max_leaf_points`)

Controls the size of leaf nodes in the hierarchical tree.

**Default**: 400 (recommended)

```python
# Smaller leaves → deeper tree → more compression → slower
H = h2pack.H2Matrix(points, kernel='gaussian', max_leaf_points=200)

# Larger leaves → shallower tree → less compression → faster
H = h2pack.H2Matrix(points, kernel='gaussian', max_leaf_points=800)
```

**When to adjust**:
- **Increase** (to 600-800) if build time is too slow
- **Decrease** (to 200-300) if you need better compression
- **Keep default** (400) in most cases

### Number of Threads (`n_threads`)

Controls OpenMP parallelization.

```python
# Use 4 threads
H = h2pack.H2Matrix(points, kernel='gaussian', n_threads=4)

# Auto-detect (use all available cores)
H = h2pack.H2Matrix(points, kernel='gaussian', n_threads=-1)
```

**Guidelines**:
- Set to 4-8 for best performance on most systems
- Avoid using all cores if you have > 16 cores (diminishing returns)
- Match your environment variables:
  ```bash
  export OPENBLAS_NUM_THREADS=4
  export OMP_NUM_THREADS=4
  ```

### JIT Mode (`jit_mode`)

Just-In-Time matrix construction.

```python
# JIT mode (default) - compute matrices on-the-fly
H = h2pack.H2Matrix(points, kernel='gaussian', jit_mode=True)

# Ahead-Of-Time mode - precompute and store all matrices
H = h2pack.H2Matrix(points, kernel='gaussian', jit_mode=False)
```

**When to use JIT**:
- When memory is limited (JIT uses less memory)
- For very large problems
- **Default recommendation**: Keep `jit_mode=True`

**When to use AOT**:
- When you'll do many matvec operations
- When build time is not critical
- When you have plenty of memory

---

## Accuracy Expectations

### Understanding H² Approximation Error

H² matrices are **approximations**. The error depends on several factors:

#### Expected Error Ranges

For `rel_tol=1e-6`:
- **Without compression** (N < 4000): Error ~ 1e-16 (machine precision)
- **With compression** (N ≥ 4000): Error ~ 1e-8 to 1e-7 (10-100x tolerance)

**This is normal and expected behavior!**

#### Validation

Always validate accuracy for your specific problem:

```python
import numpy as np

# Compute dense matrix (for small problems only!)
n = 1000
points = np.random.randn(n, 3)
K_dense = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        r_sq = np.sum((points[i] - points[j])**2)
        K_dense[i,j] = np.exp(-r_sq / 2.0)

# H2 matrix
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-6)
H.build()

# Compare
x = np.random.randn(n)
y_dense = K_dense @ x
y_h2 = H.matvec(x)

error = np.linalg.norm(y_h2 - y_dense) / np.linalg.norm(y_dense)
print(f"Relative error: {error:.6e}")
```

### When Accuracy Matters

**High-accuracy applications**:
- Scientific computing with error bounds
- Numerical PDE solvers
- Uncertainty quantification

**Solution**: Use tighter tolerance (`rel_tol=1e-8` or `1e-10`)

**Moderate-accuracy applications**:
- Machine learning (GP regression)
- Data analysis
- Visualization

**Solution**: Standard tolerance (`rel_tol=1e-6`) is fine

---

## Performance Tips

### 1. Problem Size Guidelines

**Minimum size**: 5000 points
- Below 4000 points, H² may not compress (max_rank=0)
- Compression becomes significant at 5000+ points

**Optimal size**: 10,000 - 100,000 points
- Best compression ratios
- Linear scaling advantages most apparent

**Very large**: 1,000,000+ points
- Requires careful memory management
- Use JIT mode (`jit_mode=True`)
- Monitor memory usage

### 2. Thread Configuration

```bash
# Set before running Python
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

**Or in Python**:
```python
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Then create H2Matrix
H = h2pack.H2Matrix(points, kernel='gaussian', n_threads=4)
```

### 3. Memory Management

For large problems:
```python
# Use JIT mode to save memory
H = h2pack.H2Matrix(points, kernel='gaussian', jit_mode=True)

# Increase leaf size to reduce tree depth
H = h2pack.H2Matrix(points, kernel='gaussian', max_leaf_points=800)

# Use looser tolerance
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-4)
```

### 4. Build Once, Use Many Times

```python
# Build once
H = h2pack.H2Matrix(points, kernel='gaussian')
H.build()  # Expensive operation

# Reuse for many matvecs (cheap!)
for i in range(1000):
    x = get_next_vector()
    y = H.matvec(x)  # Fast: ~10ms for 5000 points
```

### 5. Kernel Selection for Speed

**Fastest to slowest**:
1. Gaussian (fastest, good SIMD optimization)
2. Matern 3/2 / 5/2 (slightly slower)
3. Coulomb (moderate)
4. Quadratic (slowest)

---

## Troubleshooting

### Build Failures

**Error**: "Cannot find OpenBLAS"

**Solution**:
```bash
# macOS
brew install openblas
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"

# Linux
sudo apt-get install libopenblas-dev
```

### Runtime Warnings

**Warning**: "precompiled NUM_THREADS exceeded"

**Solution**:
```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

**Warning**: "On entry to DGEMV parameter number 6 had an illegal value"

**Impact**: Cosmetic warning, doesn't affect correctness
**Solution**: Can be ignored or fixed in future H2Pack releases

### No Compression

**Symptom**: `stats['max_rank'] == 0`, no compression

**Cause**: Problem too small for H² to compress

**Solution**: Use at least 5000 points, or accept that H² = dense for small N

### Accuracy Issues

**Symptom**: Results don't match expected values

**Check**:
1. Are you expecting machine precision? (H² is approximate!)
2. Is your tolerance appropriate? (Try `rel_tol=1e-8`)
3. See [Accuracy Expectations](#accuracy-expectations)

**Validation**:
Run `examples/accuracy_demo.py` to understand expected accuracy.

---

## Advanced Usage

### Custom Kernel Objects

```python
# Create kernel object explicitly
from h2pack import GaussianKernel, MaternKernel

kernel = GaussianKernel(lengthscale=2.0)
H = h2pack.H2Matrix(points, kernel=kernel)

# Matern with specific nu
kernel = MaternKernel(lengthscale=1.0, nu=2.5)
H = h2pack.H2Matrix(points, kernel=kernel)
```

### Matrix-Matrix Multiplication

```python
# Multiple vectors at once
X = np.random.randn(5000, 10)  # 10 vectors
Y = H.matmul(X)  # Y = K * X
```

### Extracting Statistics

```python
stats = H.stats

# All available fields:
print(f"Points: {stats['n_points']}")
print(f"Dimension: {stats['dim']}")
print(f"Levels: {stats['n_levels']}")
print(f"Nodes: {stats['n_nodes']}")
print(f"Max rank: {stats['max_rank']}")
print(f"Avg rank: {stats['avg_rank']}")
print(f"Storage (MB): {stats['storage_mb']}")
print(f"Compression: {stats['compression_ratio']}x")
```

### Using Utility Functions

```python
from h2pack import utils

# Print formatted statistics
utils.print_stats(H)

# Generate test points
points_grid = utils.generate_grid_3d(n=50)  # 50×50×50 grid
points_random = utils.generate_random_3d(n=10000)  # Random points
```

---

## Further Reading

- **Accuracy Validation**: See [ACCURACY_VALIDATION.md](ACCURACY_VALIDATION.md)
- **API Reference**: See [API.md](API.md)
- **Examples**: See `examples/` directory
- **Phase Reports**: See `PHASE3_SUMMARY.md` and `PHASE4_SUMMARY.md`

---

## Getting Help

- **Documentation**: This guide + API.md
- **Examples**: `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/scalable-matrix/H2Pack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/scalable-matrix/H2Pack/discussions)

---

*User Guide Version 1.0 - January 2026*
