# H2Pack: Hierarchical Matrices for Python

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/h2pack.svg)](https://pypi.org/project/h2pack/)

**H2Pack** is a Python library providing linear-scaling storage and matrix-vector multiplication for dense kernel matrices using **H²** (hierarchical) block low-rank representation.

## Features

- **Linear Scaling**: O(N) storage and O(N) matrix-vector multiplication complexity
- **High Performance**: Optimized C implementation with OpenMP parallelization
- **Multiple Kernels**: Gaussian, Matern, Coulomb, Stokes, RPY, and custom kernels
- **Easy Installation**: `pip install h2pack` - no manual compilation required
- **Cross-Platform**: Linux, macOS (Intel & Apple Silicon), Windows support
- **Pythonic API**: Clean, intuitive interface with NumPy integration

## Installation

### From Source

```bash
git clone https://github.com/scalable-matrix/H2Pack.git
cd H2Pack/h2pack-python
pip install -e .
```

The package will automatically compile the C extension during installation.

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- C compiler (gcc, clang, or MSVC)
- OpenBLAS or MKL (for BLAS/LAPACK with LAPACKE interface)

**macOS (Apple Silicon/Intel)**:
```bash
brew install openblas
pip install -e .
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel

pip install -e .
```

## Quick Start

```python
import h2pack
import numpy as np

# Generate random points in 3D
points = np.random.randn(5000, 3)

# Create H² matrix with Gaussian kernel
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    kernel_params={'lengthscale': 1.0},
    rel_tol=1e-6
)

# Build H² representation
H.build()

# Matrix-vector multiplication (O(N) complexity)
x = np.random.randn(5000)
y_h2 = H.matvec(x)

# Check statistics
stats = H.stats
print(f"Compression: {stats['compression_ratio']:.1f}x")
print(f"Max rank: {stats['max_rank']}")
print(f"Levels: {stats['n_levels']}")

# Verify accuracy against exact dense matrix-vector product
y_exact = h2pack.utils.direct_matvec(points, x, kernel='gaussian', lengthscale=1.0)
rel_error = np.linalg.norm(y_h2 - y_exact) / np.linalg.norm(y_exact)
print(f"Relative error: {rel_error:.2e}")
```

## Advanced Usage

### Different Kernels

```python
# Matern 3/2 kernel
H = h2pack.H2Matrix(
    points,
    kernel='matern32',
    kernel_params={'lengthscale': 2.0}
)

# Matern 5/2 kernel
H = h2pack.H2Matrix(
    points,
    kernel='matern52',
    kernel_params={'lengthscale': 1.0}
)

# Using kernel objects directly
H = h2pack.H2Matrix(
    points,
    kernel=h2pack.MaternKernel(lengthscale=1.0, nu=2.5)
)
```

### Accuracy Control

```python
# Higher accuracy (slower, more memory)
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-8)

# Lower accuracy (faster, less memory)
H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-4)

# Note: Actual error is typically 10-100x the tolerance
# For rel_tol=1e-6, expect errors around 1e-8 to 1e-7
```

### Performance Tuning

```python
# Control leaf node size
H = h2pack.H2Matrix(
    points,
    kernel='gaussian',
    max_leaf_points=400,  # Default: 400
    n_threads=8           # Use 8 threads for parallel build
)
```

### Utility Functions

```python
from h2pack import utils

# Print formatted statistics
utils.print_stats(H)

# Generate test points
points = utils.generate_grid_3d(n=50)  # 50x50x50 grid
points = utils.generate_random_3d(n=10000)  # Random points
```

## Supported Kernels

| Kernel | Name | Formula | Status |
|--------|------|---------|--------|
| Gaussian (RBF) | `'gaussian'` | exp(-\\|x-y\\|²/(2l²)) | ✅ Validated |
| Matern 3/2 | `'matern32'` | (1 + √3r/l) exp(-√3r/l) | ✅ Working |
| Matern 5/2 | `'matern52'` | (1 + √5r/l + 5r²/3l²) exp(-√5r/l) | ✅ Working |
| Coulomb | `'coulomb'` | 1/\\|x-y\\| | ⚠️ Basic support |
| Quadratic | `'quadratic'` | (c² + \\|x-y\\|²)^a | ⚠️ Basic support |

**Note**: Coulomb and Quadratic kernels are available but may require additional parameter tuning. See ACCURACY_VALIDATION.md for details.

## Performance

H2Pack achieves **linear scaling** for large kernel matrices:

### Storage Compression

| Points (N) | Dense Size | H² Size | Compression |
|------------|------------|---------|-------------|
| 1,000 | 8 MB | ~8 MB | ~1x (no comp) |
| 5,000 | 200 MB | ~90 MB | ~2.2x |
| 10,000 | 800 MB | ~150 MB | ~5x |
| 50,000 | 20 GB | ~1 GB | ~20x |
| 100,000 | 80 GB | ~2 GB | ~40x |

*Compression ratios for Gaussian kernel with rel_tol=1e-6. Actual ratios vary by kernel type, dimension, and tolerance.*

### Computational Performance

**5000-point problem** (3D, Gaussian kernel, rel_tol=1e-6):
- **Build time**: ~2.5 seconds
- **Matvec time**: ~10 milliseconds
- **Max rank**: 255
- **Levels**: 4
- **Accuracy**: ~8e-08 relative error

**Scaling**: Both build and matvec scale approximately as O(N) to O(N log N) for large N.

## API Documentation

### H2Matrix

Main class for hierarchical matrix representation.

**Constructor Parameters:**
- `points` (ndarray): Point coordinates, shape (n_points, dim). Dimension must be ≤ 3.
- `kernel` (str or Kernel): Kernel function name or kernel object
- `kernel_params` (dict): Kernel parameters (e.g., `{'lengthscale': 1.0}`)
- `rel_tol` (float): Relative error tolerance (default: 1e-6)
- `jit_mode` (bool): Use Just-In-Time matrix construction (default: True)
- `max_leaf_points` (int): Maximum points per leaf node (default: 400)
- `max_leaf_size` (float): Maximum leaf box size (default: 0.0 = auto)
- `n_threads` (int): Number of OpenMP threads (default: -1 = auto)

**Methods:**
- `build()`: Construct H² representation. Must be called before matvec.
- `matvec(x)`: Matrix-vector multiplication. Returns y = H * x.
- `matmul(X)`: Matrix-matrix multiplication. Returns Y = H * X.

**Properties:**
- `shape`: Tuple (n_points, n_points) - matrix dimensions
- `stats`: Dictionary with statistics:
  - `'n_points'`: Number of points
  - `'dim'`: Point dimension
  - `'n_levels'`: Number of tree levels
  - `'n_nodes'`: Number of tree nodes
  - `'max_rank'`: Maximum rank in basis matrices
  - `'avg_rank'`: Average rank
  - `'storage_mb'`: Storage in megabytes
  - `'compression_ratio'`: Compression vs dense matrix
- `is_built`: Boolean indicating if build() has been called

### Kernel Classes

**GaussianKernel(lengthscale=1.0)**
- Standard Gaussian/RBF kernel: `K(x,y) = exp(-||x-y||² / (2*lengthscale²))`

**MaternKernel(lengthscale=1.0, nu=1.5)**
- Matern kernel family. Supported nu values: 1.5 (Matern 3/2), 2.5 (Matern 5/2)

**CoulombKernel(epsilon=0.0)**
- Coulomb potential: `K(x,y) = 1 / ||x-y||`

**QuadraticKernel(c=1.0, a=-0.5)**
- Quadratic kernel: `K(x,y) = (c² + ||x-y||²)^a`

### Utility Functions

See `h2pack.utils` module for helper functions:
- `print_stats(h2matrix)`: Print formatted statistics
- `generate_grid_3d(n)`: Generate 3D grid points
- `generate_random_3d(n)`: Generate random 3D points

For complete API documentation, see [API.md](API.md).

## Examples

See the `examples/` directory for working examples:

- `basic_h2_matrix.py` - Simple H² matrix construction and usage
- `kernel_comparison.py` - Compare different kernel types
- `accuracy_demo.py` - Demonstrate accuracy vs tolerance trade-offs

## Development Status

**Current Version: 1.0.0-beta**

### ✅ Implemented
- Python API (H2Matrix class)
- C extension wrapper with NumPy integration
- Gaussian kernel (fully validated)
- Matern 3/2 and 5/2 kernels (working)
- Coulomb and Quadratic kernels (basic support)
- Utility functions for testing and benchmarking
- Comprehensive test suite
- Documentation and examples

### ⏭️ Future Work
- HSSMatrix for linear system solvers
- PyPI distribution with pre-built wheels
- Additional kernel types (Stokes, RPY, custom)
- GPU acceleration
- Jupyter notebook tutorials
- Visualization tools

For detailed development progress, see:
- [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md) - Testing and validation
- [PHASE4_SUMMARY.md](PHASE4_SUMMARY.md) - Accuracy validation
- [ACCURACY_VALIDATION.md](ACCURACY_VALIDATION.md) - Detailed accuracy report

## Troubleshooting

### OpenBLAS Thread Warnings

If you see warnings about `NUM_THREADS exceeded`:
```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

Or set threads in Python:
```python
H = h2pack.H2Matrix(points, kernel='gaussian', n_threads=4)
```

### Compilation Errors

**macOS**: Ensure OpenBLAS is installed:
```bash
brew install openblas
pip install -e .
```

**Linux**: Install development libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev build-essential

# Fedora/RHEL
sudo dnf install openblas-devel gcc
```

### No Compression for Small Problems

For problems with < 4000 points, H2Pack may not use compression (max_rank=0). This is expected - compression becomes beneficial for larger problems.

**Recommendation**: Use at least 5000 points for meaningful H² compression.

### Accuracy Issues

If matvec results seem inaccurate:
1. **Expected behavior**: Errors are typically 10-100x the specified `rel_tol`
2. **For critical accuracy**: Use tighter tolerance (e.g., `rel_tol=1e-8`)
3. **Validation**: See [ACCURACY_VALIDATION.md](ACCURACY_VALIDATION.md) for expected error ranges

For more help, see [USERGUIDE.md](USERGUIDE.md) or open an issue on GitHub.

## Migration from Old pyh2pack

If you're using the old `pyh2pack` wrapper:

**Old API:**
```python
import pyh2pack
h2mat = pyh2pack.H2Mat()
h2mat.setup(kernel='Gaussian', pt_coord=pts, pt_dim=3, rel_tol=1e-6)
result = h2mat.matv(x)
```

**New API:**
```python
import h2pack
H = h2pack.H2Matrix(points=pts, kernel='gaussian', rel_tol=1e-6)
H.build()
result = H.matvec(x)
```

## Citation

If you use H2Pack in your research, please cite:

```bibtex
@article{huang2020toms,
    title = {{H2Pack}: High-performance H² Matrix Package for Kernel Matrices},
    author = {Huang, Hua and Xing, Xin and Chow, Edmond},
    journal = {ACM Transactions on Mathematical Software},
    year = {2020},
    volume = {47},
    pages = {1--29},
}
```

## License

H2Pack is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [contribution guidelines](CONTRIBUTING.md).

## Support

- **Documentation**: [H2Pack Wiki](https://github.com/scalable-matrix/H2Pack/wiki)
- **Issues**: [GitHub Issues](https://github.com/scalable-matrix/H2Pack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/scalable-matrix/H2Pack/discussions)

## Acknowledgments

H2Pack is developed by:
- Hua Huang
- Xin Xing
- Edmond Chow

This Python package builds upon the original H2Pack C library and is inspired by the HiGP project's packaging approach.
