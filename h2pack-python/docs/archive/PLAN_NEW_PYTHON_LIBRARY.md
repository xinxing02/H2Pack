# Plan: Modern H2Pack Python Library

## Executive Summary

Create a modern, pip-installable Python library for H2Pack that:
1. Eliminates manual build requirements for end users
2. Follows modern Python packaging standards (setuptools, pyproject.toml)
3. Provides a clean, Pythonic API similar to HiGP's approach
4. Supports multiple platforms (Linux, macOS Intel/ARM, Windows)
5. Includes comprehensive documentation and examples

## Analysis of Current State

### Current pyh2pack Issues
1. **Manual build required**: Users must build H2Pack C library first
2. **Old-style setup.py**: Uses deprecated `distutils`, hardcoded paths
3. **Platform-specific configuration**: Requires manual path editing
4. **No dependency management**: OpenBLAS/gfortran paths hardcoded
5. **Limited API**: Direct C wrapper without Python conveniences
6. **No PyPI distribution**: Cannot `pip install h2pack`

### HiGP's Successful Approach (What to Learn)
1. **Self-contained build**: All C++ sources included in package
2. **Smart platform detection**: Automatic BLAS library detection
3. **Modern setuptools**: Uses Extension with automatic compilation
4. **Pythonic wrapper layer**: Python classes wrapping C extension
5. **Proper packaging**: README as long_description, version management
6. **CI/CD for releases**: Travis CI builds wheels for multiple platforms
7. **Clean separation**: C extension (`higp_cext`) + Python layer (`higp.modules`)

## Proposed Architecture

### Directory Structure
```
h2pack/
├── pyproject.toml              # Modern build system configuration
├── setup.py                    # Build configuration
├── README.md                   # User documentation
├── LICENSE                     # License file
├── MANIFEST.in                 # Include C sources in sdist
│
├── h2pack/                     # Python package
│   ├── __init__.py            # Package initialization
│   ├── core.py                # High-level Python API
│   ├── kernels.py             # Kernel function definitions
│   ├── utils.py               # Utility functions
│   └── _version.py            # Version information
│
├── src/                        # C extension sources
│   ├── h2pack_cext.c          # Python C extension wrapper
│   ├── h2pack_cext.h          # C extension header
│   │
│   └── h2pack/                # H2Pack C library sources (copied)
│       ├── H2Pack_build.c
│       ├── H2Pack_matvec.c
│       ├── H2Pack_matmul.c
│       ├── H2Pack_partition.c
│       ├── AFN_precond.c
│       └── ... (all H2Pack C sources)
│
├── examples/                   # Example scripts
│   ├── basic_h2_matrix.py
│   ├── hss_solver.py
│   ├── kernel_comparison.py
│   └── performance_benchmark.py
│
├── tests/                      # Unit tests
│   ├── test_h2_matrix.py
│   ├── test_kernels.py
│   ├── test_matvec.py
│   └── test_hss.py
│
└── ci/                         # CI/CD scripts
    ├── build-wheels-linux.sh
    ├── build-wheels-macos.sh
    └── test-package.sh
```

## Implementation Plan

### Phase 1: Project Setup and Structure (Week 1)

**Task 1.1: Create New Package Structure**
- Create new directory `h2pack-python/` (separate from current pyh2pack)
- Set up modern Python package structure as outlined above
- Initialize git repository for version control

**Task 1.2: Copy and Organize C Sources**
- Copy all H2Pack C source files from `H2Pack/src/` to `src/h2pack/`
- Include necessary headers from `H2Pack/include/`
- Copy ASTER library sources if needed
- Create MANIFEST.in to include all C sources in source distribution

**Task 1.3: Create Modern Build Configuration**
- Write `pyproject.toml` with PEP 517/518 compliance
- Write `setup.py` based on HiGP's approach but adapted for H2Pack
- Implement automatic BLAS library detection (MKL, OpenBLAS, Accelerate)
- Implement platform-specific compiler flags

### Phase 2: C Extension Development (Week 2-3)

**Task 2.1: Design C Extension API**
Expose core H2Pack functionality:
- `H2Matrix` class: Wrapper for H2Pack_p structure
  - `__init__(points, kernel, rel_tol, ...)`
  - `build()`: Construct H2 representation
  - `matvec(x)`: Matrix-vector multiplication
  - `matmul(X)`: Matrix-matrix multiplication
  - `get_stats()`: Return storage, timing info

- `HSSMatrix` class: Wrapper for HSS matrices
  - `__init__(points, kernel, rel_tol, ...)`
  - `build()`: Construct HSS representation
  - `factorize()`: ULV factorization
  - `solve(b)`: Linear system solve

- Kernel functions:
  - `GaussianKernel`
  - `MaternKernel`
  - `CoulombKernel`
  - `StokesKernel`
  - `RPYKernel`
  - `CustomKernel`

**Task 2.2: Write C Extension Wrapper**
- Create `src/h2pack_cext.c` based on HiGP's pattern
- Use Python C API with NumPy arrays
- Implement proper error handling and memory management
- Support both single and double precision
- Add docstrings for all functions

**Task 2.3: Handle Platform-Specific Builds**
Implement in `setup.py`:
- **macOS**:
  - Detect Apple Silicon vs Intel
  - Use Accelerate framework or OpenBLAS
  - Handle OpenMP with `-Xpreprocessor -fopenmp`
  - Detect OpenMP from PyTorch or Homebrew
- **Linux**:
  - Detect OpenBLAS or MKL via lsof (like HiGP)
  - Support both LP64 and ILP64 interfaces
  - Use GCC with standard flags
- **Windows**:
  - MSVC compiler support
  - MKL or OpenBLAS linking
  - OpenMP support

### Phase 3: Python API Layer (Week 3-4)

**Task 3.1: Create High-Level Python Classes**

```python
# h2pack/core.py

class H2Matrix:
    """High-level interface to H2 matrices."""

    def __init__(self, points, kernel='gaussian', rel_tol=1e-6,
                 kernel_params=None, jit_mode=True, **kwargs):
        """
        Initialize H2 matrix.

        Parameters
        ----------
        points : ndarray, shape (n_points, dim)
            Point coordinates
        kernel : str or callable
            Kernel function name or custom kernel
        rel_tol : float
            Relative error tolerance
        kernel_params : dict, optional
            Kernel parameters (e.g., {'lengthscale': 1.0})
        jit_mode : bool
            Use Just-In-Time mode for memory efficiency
        """

    def build(self):
        """Construct H2 matrix representation."""

    def matvec(self, x):
        """Matrix-vector multiplication: y = H * x"""

    def matmul(self, X):
        """Matrix-matrix multiplication: Y = H * X"""

    @property
    def shape(self):
        """Matrix shape (n, n)"""

    @property
    def stats(self):
        """Return statistics (storage, ranks, timing)"""
```

**Task 3.2: Implement Kernel Functions**

```python
# h2pack/kernels.py

class Kernel:
    """Base class for kernel functions."""

class GaussianKernel(Kernel):
    def __init__(self, lengthscale=1.0):
        self.lengthscale = lengthscale

class MaternKernel(Kernel):
    def __init__(self, lengthscale=1.0, nu=1.5):
        self.lengthscale = lengthscale
        self.nu = nu

# Custom kernel support
def custom_kernel(func, params):
    """Wrap user-defined kernel function."""
```

**Task 3.3: Create Utility Functions**

```python
# h2pack/utils.py

def generate_test_points(n_points, dim, distribution='uniform'):
    """Generate test point sets."""

def benchmark_performance(matrix_sizes, kernel_types):
    """Benchmark H2Pack performance."""

def visualize_tree(h2matrix):
    """Visualize H2 tree structure."""
```

### Phase 4: Examples and Documentation (Week 4-5)

**Task 4.1: Create Comprehensive Examples**
- `examples/basic_h2_matrix.py`: Simple H2 matrix construction and matvec
- `examples/kernel_comparison.py`: Compare different kernels
- `examples/hss_linear_solver.py`: Solve linear system with HSS
- `examples/performance_scaling.py`: Demonstrate linear scaling
- `examples/custom_kernel.py`: User-defined kernel function
- `examples/reuse_proxy_points.py`: Proxy point reuse

**Task 4.2: Write Documentation**
- README.md with quick start guide
- API reference documentation
- Kernel function guide
- Performance tuning guide
- Migration guide from old pyh2pack

**Task 4.3: Add Type Hints and Docstrings**
- Type hints for all Python functions
- NumPy-style docstrings
- Sphinx-compatible documentation

### Phase 5: Testing Infrastructure (Week 5-6)

**Task 5.1: Unit Tests**
```python
# tests/test_h2_matrix.py
def test_h2_construction():
    """Test H2 matrix construction."""

def test_matvec_accuracy():
    """Test matvec accuracy against direct computation."""

def test_different_kernels():
    """Test various kernel functions."""

# tests/test_hss.py
def test_hss_factorization():
    """Test HSS ULV factorization."""

def test_hss_solve():
    """Test HSS linear solver."""
```

**Task 5.2: Integration Tests**
- Compare with original H2Pack C library
- Verify results match example outputs
- Test on different point distributions
- Test edge cases (small/large matrices)

**Task 5.3: Performance Tests**
- Verify linear scaling O(N)
- Compare with direct methods
- Benchmark across platforms

### Phase 6: CI/CD and Distribution (Week 6-7)

**Task 6.1: Set Up GitHub Actions**
```yaml
# .github/workflows/build-test.yml
- Build and test on multiple platforms
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Platforms: Ubuntu, macOS (Intel + ARM), Windows
```

**Task 6.2: Create Wheel Building Pipeline**
- Use cibuildwheel for multi-platform wheels
- Or use manylinux Docker containers (like HiGP)
- Build wheels for:
  - Linux: manylinux2014_x86_64, manylinux2014_aarch64
  - macOS: macosx_10_9_x86_64, macosx_11_0_arm64
  - Windows: win_amd64

**Task 6.3: PyPI Publishing**
- Set up PyPI account and project
- Create release workflow
- Test on TestPyPI first
- Publish to PyPI

## Technical Details

### setup.py Structure (Based on HiGP)

```python
import os
import platform
import numpy
import setuptools
from pathlib import Path

workdir = os.path.abspath(os.path.dirname(__file__))

# Compiler flags
cflags = ["-g", "-std=c11", "-O3", "-fPIC"]
cflags += ["-Wno-unused-result", "-Wno-unused-function"]
lflags = ["-lm"]

# Platform-specific configuration
if platform.system() == 'Darwin':
    if platform.machine() == 'arm64':
        # Apple Silicon
        cflags += ["-Xpreprocessor", "-fopenmp"]
        # Try PyTorch OpenMP, then Homebrew
        # Use Accelerate framework
        cflags += ["-DUSE_ACCELERATE_LP64"]
        lflags += ["-framework", "Accelerate"]
    else:
        # Intel Mac
        cflags += ["-fopenmp"]
elif platform.system() == 'Linux':
    cflags += ["-fopenmp"]
    lflags += ["-fopenmp"]
    # Detect OpenBLAS/MKL via lsof
elif platform.system() == 'Windows':
    # MSVC configuration
    pass

# Collect all C sources
h2pack_sources = [
    str(p) for p in Path(workdir, 'src/h2pack').glob('*.c')
]
extension_sources = [
    workdir + '/src/h2pack_cext.c'
] + h2pack_sources

h2pack_cext = setuptools.Extension(
    'h2pack._h2pack_cext',
    sources=extension_sources,
    include_dirs=[
        numpy.get_include(),
        workdir + '/src',
        workdir + '/src/h2pack'
    ],
    extra_compile_args=cflags,
    extra_link_args=lflags,
    language="c"
)

setuptools.setup(
    name="h2pack",
    version="1.0.0",
    description="Python interface for H2Pack: hierarchical matrices",
    long_description=Path(workdir, "README.md").read_text(),
    long_description_content_type='text/markdown',
    author="H2Pack Team",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.20.0"],
    python_requires=">=3.8",
    ext_modules=[h2pack_cext],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "numpy>=1.20.0"]
build-backend = "setuptools.build_meta"

[project]
name = "h2pack"
version = "1.0.0"
description = "Python interface for H2Pack hierarchical matrices"
readme = "README.md"
requires-python = ">=3.8"
license = {file = "LICENSE"}
authors = [
    {name = "H2Pack Team"},
]
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
test = ["pytest>=6.0", "pytest-cov"]
docs = ["sphinx", "sphinx-rtd-theme"]
examples = ["matplotlib", "scipy"]

[project.urls]
Homepage = "https://github.com/scalable-matrix/H2Pack"
Documentation = "https://h2pack.readthedocs.io"
```

## API Design Examples

### Basic Usage
```python
import h2pack
import numpy as np

# Generate test points
points = np.random.randn(10000, 3)

# Create H2 matrix with Gaussian kernel
H = h2pack.H2Matrix(
    points=points,
    kernel='gaussian',
    lengthscale=1.0,
    rel_tol=1e-6
)

# Build H2 representation
H.build()

# Matrix-vector multiplication
x = np.random.randn(10000)
y = H.matvec(x)

# Get statistics
print(H.stats)
# Output: {'compression_ratio': 15.2, 'avg_rank': 8, 'build_time': 0.5}
```

### Advanced Usage
```python
# HSS matrix for linear solve
H = h2pack.HSSMatrix(points, kernel='matern', nu=2.5)
H.build()
H.factorize()

# Solve linear system
b = np.random.randn(len(points))
x = H.solve(b)

# Custom kernel
def my_kernel(x, y, params):
    r = np.linalg.norm(x - y)
    return np.exp(-r / params['scale'])

H = h2pack.H2Matrix(
    points=points,
    kernel=my_kernel,
    kernel_params={'scale': 2.0}
)
```

## Migration Strategy

### For Existing pyh2pack Users

Provide migration guide:
```python
# Old pyh2pack
import pyh2pack
h2mat = pyh2pack.H2Mat()
h2mat.setup(kernel='Gaussian', pt_coord=pts, pt_dim=3, ...)
h2mat.matv(x)

# New h2pack
import h2pack
H = h2pack.H2Matrix(points=pts, kernel='gaussian', ...)
H.build()
y = H.matvec(x)
```

Keep old pyh2pack for now, mark as deprecated.

## Success Criteria

1. **Installation**: `pip install h2pack` works on all major platforms
2. **No dependencies**: No manual BLAS/OpenMP installation required
3. **Performance**: Match or exceed current pyh2pack performance
4. **API**: Clean, Pythonic API with comprehensive docs
5. **Testing**: >90% code coverage, CI passing on all platforms
6. **Distribution**: Wheels available on PyPI for common platforms
7. **Documentation**: Complete docs with examples

## Timeline

- **Week 1**: Project setup and structure
- **Week 2-3**: C extension development
- **Week 3-4**: Python API layer
- **Week 4-5**: Examples and documentation
- **Week 5-6**: Testing infrastructure
- **Week 6-7**: CI/CD and distribution
- **Week 8**: Polish, bug fixes, release prep

**Total estimated time**: 8 weeks for full implementation

## Risks and Mitigation

### Risk 1: Platform-specific build issues
**Mitigation**: Extensive testing on CI, fallback configurations

### Risk 2: BLAS library compatibility
**Mitigation**: Support multiple BLAS libraries, clear error messages

### Risk 3: Performance regression
**Mitigation**: Comprehensive benchmarks, comparison with C version

### Risk 4: Breaking changes from C library
**Mitigation**: Pin to stable H2Pack version, version compatibility matrix

## Future Enhancements

1. **GPU support**: CUDA/ROCm kernels for matrix operations
2. **Sparse matrix support**: Integration with scipy.sparse
3. **Parallel I/O**: Save/load H2 matrices in parallel
4. **Advanced preconditioners**: More preconditioner options
5. **Visualization tools**: Interactive tree structure visualization
6. **JAX/PyTorch integration**: Automatic differentiation support

## References

- HiGP setup.py: `/Users/xin/Programs/tSNE/HiGP/py-interface/setup.py`
- Current pyh2pack: `/Users/xin/Programs/tSNE/H2Pack/pyh2pack/`
- H2Pack C sources: `/Users/xin/Programs/tSNE/H2Pack/src/`
- Python packaging guide: https://packaging.python.org/
- Building C extensions: https://docs.python.org/3/extending/building.html
