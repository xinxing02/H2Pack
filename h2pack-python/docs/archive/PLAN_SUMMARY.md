# Modern H2Pack Python Library - Plan Summary

## Goal
Create a pip-installable H2Pack Python library that eliminates manual build hassles, following the successful HiGP approach.

## Key Improvements Over Current pyh2pack

| Current pyh2pack | New h2pack |
|-----------------|------------|
| Requires pre-built C library | Self-contained, builds during `pip install` |
| Manual path configuration | Automatic BLAS/OpenMP detection |
| Old distutils setup | Modern setuptools + pyproject.toml |
| Direct C wrapper | Pythonic API with Python layer |
| Local install only | PyPI distribution with wheels |
| Platform-specific setup | Cross-platform support |
| Minimal examples | Comprehensive docs & examples |

## Architecture

```
h2pack/
├── pyproject.toml          # Modern build config
├── setup.py                # Auto-detecting build (HiGP-style)
├── h2pack/                 # Python package
│   ├── __init__.py
│   ├── core.py            # H2Matrix, HSSMatrix classes
│   ├── kernels.py         # Kernel definitions
│   └── utils.py           # Utilities
├── src/                    # C extension
│   ├── h2pack_cext.c      # Python C API wrapper
│   └── h2pack/            # All H2Pack C sources (bundled)
├── examples/               # Usage examples
└── tests/                  # Unit tests
```

## Key Learning from HiGP

1. **Bundle all C sources**: Include H2Pack C files in package
2. **Smart build system**: Auto-detect BLAS (MKL/OpenBLAS/Accelerate)
3. **Platform handling**:
   - macOS: Detect Apple Silicon, use Accelerate, find OpenMP
   - Linux: lsof-based BLAS detection
   - Windows: MSVC support
4. **Two-layer design**:
   - Low-level C extension (`_h2pack_cext`)
   - High-level Python classes (`h2pack.H2Matrix`)
5. **CI/CD**: Build wheels for multiple platforms

## User Experience

### Before (current pyh2pack):
```bash
# User must do:
cd H2Pack/src
make -f GCC-OpenBLAS.make
cd ../pyh2pack
# Edit setup.py to fix paths
pip install -e .
```

### After (new h2pack):
```bash
# User just does:
pip install h2pack
```

### Usage Comparison

**Old API**:
```python
import pyh2pack
h2mat = pyh2pack.H2Mat()
h2mat.setup(kernel='Gaussian', pt_coord=pts, pt_dim=3, rel_tol=1e-6)
result = h2mat.matv(x)
```

**New API**:
```python
import h2pack
H = h2pack.H2Matrix(points=pts, kernel='gaussian', rel_tol=1e-6)
H.build()
result = H.matvec(x)
print(H.stats)  # Access statistics
```

## Implementation Phases

### Phase 1: Setup (Week 1)
- Create new package structure
- Copy C sources
- Write modern build config

### Phase 2: C Extension (Week 2-3)
- Write Python C API wrapper
- Platform-specific builds
- Memory management

### Phase 3: Python Layer (Week 3-4)
- High-level H2Matrix class
- Kernel function classes
- Utility functions

### Phase 4: Documentation (Week 4-5)
- Examples and tutorials
- API documentation
- Migration guide

### Phase 5: Testing (Week 5-6)
- Unit tests
- Integration tests
- Performance benchmarks

### Phase 6: Distribution (Week 6-7)
- CI/CD setup (GitHub Actions)
- Build wheels (cibuildwheel)
- PyPI publishing

## Technical Highlights

### setup.py Key Features

```python
# Auto-detect platform
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    # Apple Silicon: Use Accelerate
    cflags += ["-DUSE_ACCELERATE_LP64"]
    lflags += ["-framework", "Accelerate"]
    # Find OpenMP from PyTorch or Homebrew
elif platform.system() == 'Linux':
    # Detect BLAS via lsof (HiGP approach)
    has_mkl, has_openblas, lib_dir, lib_name = get_numpy_linalg_lib()

# Bundle all sources
sources = glob('src/h2pack/*.c') + ['src/h2pack_cext.c']
```

### Pythonic API Design

```python
class H2Matrix:
    """Hierarchical matrix representation."""

    def __init__(self, points, kernel='gaussian', rel_tol=1e-6, **kwargs):
        self._matrix = _h2pack_cext.H2Matrix(...)  # C extension

    def build(self):
        """Construct H2 representation."""
        return self._matrix.build()

    def matvec(self, x):
        """Matrix-vector product."""
        return self._matrix.matvec(x)

    @property
    def stats(self):
        """Storage and performance statistics."""
        return self._matrix.get_stats()
```

## Success Metrics

- ✅ `pip install h2pack` works on Linux, macOS, Windows
- ✅ No manual BLAS configuration needed
- ✅ Performance matches current pyh2pack
- ✅ Clean API with comprehensive docs
- ✅ 90%+ test coverage
- ✅ Wheels on PyPI for Python 3.8-3.12
- ✅ CI passing on all platforms

## Timeline

**8 weeks total**: 2 weeks per phase (Setup + C Extension + Python Layer + Docs/Tests/CI)

## Next Steps

1. **Review and approve plan**
2. **Create new repository** (h2pack-python)
3. **Start Phase 1**: Set up structure and copy sources
4. **Iterate**: Build, test, refine each phase
5. **Release**: Publish v1.0.0 to PyPI

## Questions to Consider

1. Should we keep old pyh2pack for backward compatibility?
2. What's the target Python version range? (suggest 3.8+)
3. Do we want GPU support in v1.0 or later?
4. Should we include visualization tools?
5. Integration with other libraries (scipy, scikit-learn)?

---

**Note**: This plan is based on analyzing HiGP's successful approach at `/Users/xin/Programs/tSNE/HiGP/` and adapting it for H2Pack's specific needs.
