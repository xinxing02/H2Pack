# H2Pack Setup Guide for Apple Silicon MacBook Pro

## Overview

This document describes how to build and install H2Pack on Apple Silicon (ARM64) Macs. The setup has been tested and verified on macOS with Apple Clang compiler, Homebrew OpenBLAS, and OpenMP.

## Prerequisites

Install the required dependencies via Homebrew:

```bash
brew install openblas
brew install libomp
brew install gcc  # For gfortran
```

Verify installations:
```bash
brew --prefix openblas    # Should show /opt/homebrew/opt/openblas
brew --prefix libomp      # Should show /opt/homebrew/opt/libomp
which gfortran            # Should show /opt/homebrew/bin/gfortran
```

Python requirements:
- Python 3.10 or newer
- NumPy 1.24.4 or newer

## Configuration Changes for Apple Silicon

The following files have been modified to support Apple Silicon:

### 1. `src/common.make`

Added Apple Clang compiler detection and proper OpenMP/OpenBLAS linking:

```makefile
# Apple Clang support for macOS (including Apple Silicon)
ifeq ($(shell $(CC) --version 2>&1 | grep -c "Apple clang"), 1)
CFLAGS += -Xpreprocessor -fopenmp -march=native -Wno-unused-result -Wno-unused-function
INCS   += -I/opt/homebrew/opt/libomp/include
LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
endif

# OpenBLAS configuration
OPENBLAS_INSTALL_DIR = /opt/homebrew/opt/openblas
ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib -lopenblas
endif
```

Also updated the shared library build rule to include LDFLAGS:
```makefile
$(LIB_SO): $(C_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)
```

### 2. `pyh2pack/setup.py`

Updated paths and compiler flags for macOS:

```python
H2PACK_DIR = ".."
OPENBLAS_INSTALL_DIR = "/opt/homebrew/opt/openblas"

extra_cflags  = ["-I"+H2PACK_DIR+"/include"]
extra_cflags += ["-I"+OPENBLAS_INSTALL_DIR+"/include"]
extra_cflags += ["-I/opt/homebrew/opt/libomp/include"]
extra_cflags += ["-g", "-std=gnu99", "-O3"]
extra_cflags += ["-DUSE_OPENBLAS", "-Xpreprocessor", "-fopenmp", "-march=native"]
extra_cflags += ["-Wno-unused-result", "-Wno-unused-function"]

LIB = [H2PACK_DIR+"/lib/libH2Pack.a", OPENBLAS_INSTALL_DIR+"/lib/libopenblas.a"]
extra_lflags = LIB + ["-g", "-O3", "-L/opt/homebrew/opt/libomp/lib", "-lomp", "-L/opt/homebrew/lib/gcc/current", "-lgfortran", "-lm"]
```

## Build Instructions

### Step 1: Build the H2Pack C Library

```bash
cd src
make -f GCC-OpenBLAS.make clean
make -f GCC-OpenBLAS.make
```

This will create:
- `lib/libH2Pack.a` (static library)
- `lib/libH2Pack.so` (shared library)
- Headers in `include/`

Expected output: Compilation warnings about SLEEF library are normal and can be ignored.

### Step 2: Build and Install the Python Interface

```bash
cd pyh2pack
pip install --no-build-isolation --no-deps -e .
```

The `--no-build-isolation` flag ensures the build uses your local environment.
The `-e` flag installs in "editable" mode for development.

### Step 3: Verify Installation

Test the Python interface:
```bash
python3 -c "import pyh2pack; print('PyH2Pack imported successfully!')"
```

Run example scripts:
```bash
python3 example.py              # Basic H2 matrix example
python3 example_hss.py          # HSS matrix example
python3 example_samplept.py     # Sample point method example
```

## Common Issues and Solutions

### Issue: Symbol not found errors for gfortran

**Error**: `symbol not found in flat namespace '__gfortran_concat_string'`

**Solution**: Make sure gfortran is linked in `setup.py`:
```python
extra_lflags = LIB + [..., "-L/opt/homebrew/lib/gcc/current", "-lgfortran", "-lm"]
```

### Issue: OpenMP not found

**Error**: `omp.h` file not found

**Solution**: Install OpenMP via Homebrew:
```bash
brew install libomp
```

### Issue: Version mismatch warnings

**Warning**: `building for macOS-11.0, but linking with dylib ... which was built for newer version 15.0`

**Solution**: These warnings are harmless and can be ignored. They occur because the libraries were built for a newer macOS version, but are backward compatible.

## Rebuilding After Changes

If you modify the C source code:

```bash
# Rebuild C library
cd src
make -f GCC-OpenBLAS.make clean
make -f GCC-OpenBLAS.make

# Rebuild Python interface
cd ../pyh2pack
pip install --no-build-isolation --no-deps --force-reinstall -e .
```

## Performance Notes

On Apple Silicon, H2Pack uses:
- **NEON SIMD instructions** via `-march=native` (ARM64 optimized)
- **OpenMP parallelization** for multi-core performance
- **ARM-optimized OpenBLAS** for BLAS/LAPACK operations

The SLEEF library warnings indicate that SIMD optimizations fall back to loop implementations, which is normal and doesn't significantly impact performance for most use cases.

## Example Output

Running `example.py` should produce output similar to:

```
==================== H2Pack H2 tree info ====================
  * Number of points               : 80000
  * Kernel matrix size             : 80000
  * Maximum points in a leaf node  : 400
  * Number of levels (root at 0)   : 4
  * Number of nodes                : 585
==================== H2Pack storage info ====================
  * H2 representation U, B, D      : 4.20, 9.71, 1038.45 (MB)
  * Max / Avg compressed rank      : 11, 7
==================== H2Pack timing info =====================
  * H2 construction time (sec)     = 0.009
  * H2 matvec average time (sec)   = 0.189, 5.68 GB/s
=============================================================
```

## System Information

This setup was tested on:
- **Hardware**: Apple Silicon (ARM64) MacBook Pro
- **OS**: macOS 15.0 (Sequoia)
- **Compiler**: Apple Clang 17.0.0
- **OpenBLAS**: 0.3.30
- **OpenMP**: libomp 21.1.7
- **GCC/gfortran**: 15.2.0
- **Python**: 3.10.10
- **NumPy**: 1.26.4

## Using H2Pack in Your Code

Basic Python usage:

```python
import pyh2pack
import numpy as np

# Example: Setup H2 matrix for Gaussian kernel
# See example.py for complete working examples
```

For detailed API documentation, refer to:
- `pyh2pack/example.py` - Basic H2 matrix operations
- `pyh2pack/example_hss.py` - HSS matrix operations
- `pyh2pack/example_samplept.py` - Sample point method
- Main H2Pack README: [https://github.com/scalable-matrix/H2Pack](https://github.com/scalable-matrix/H2Pack)

## Additional Resources

- **H2Pack Wiki**: https://github.com/scalable-matrix/H2Pack/wiki
- **Python Interface Guide**: https://github.com/scalable-matrix/H2Pack/wiki/Using-H2Pack-in-Python
- **Kernel Functions**: https://github.com/scalable-matrix/H2Pack/wiki/Using-and-Writing-Kernel-Functions

## Troubleshooting

If you encounter issues:

1. Verify all prerequisites are installed correctly
2. Check that paths in `src/common.make` and `pyh2pack/setup.py` point to actual library locations
3. Ensure you're using the ARM64 Homebrew (`/opt/homebrew`) not Intel (`/usr/local`)
4. Try a clean rebuild: `make clean` before `make`

For additional help, refer to the main README.md or open an issue on the H2Pack GitHub repository.
