"""
Setup script for h2pack Python package.

This setup.py is inspired by HiGP's approach and provides:
- Automatic BLAS library detection (OpenBLAS, MKL, Accelerate)
- Platform-specific compiler configuration (Linux, macOS, Windows)
- OpenMP detection and configuration
- Self-contained build (all C sources included)
"""

import os
import sys
import platform
import numpy
import setuptools
from pathlib import Path
from glob import glob

workdir = os.path.abspath(os.path.dirname(__file__))

# Basic C compiler flags
cflags = ["-g", "-std=c11", "-O3", "-fPIC"]
cflags += ["-Wno-unused-result", "-Wno-unused-function", "-Wno-unused-variable"]
lflags = ["-lm"]

# Platform-specific configuration
skip_blas_detection = False

if platform.system() == 'Darwin':
    # macOS configuration
    if platform.machine() == 'arm64':
        print("=" * 60)
        print("Building for Apple Silicon (ARM64)")
        print("=" * 60)

        os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '11.0')

        # OpenMP configuration for Apple Clang
        cflags += ["-Xpreprocessor", "-fopenmp"]

        omp_found = False

        # Try to find OpenMP from PyTorch first
        try:
            import torch
            torch_path = os.path.dirname(torch.__file__)
            torch_omp_include = os.path.join(torch_path, 'include')
            torch_omp_lib = os.path.join(torch_path, 'lib')
            torch_omp_header = os.path.join(torch_omp_include, 'omp.h')
            torch_omp_dylib = os.path.join(torch_omp_lib, 'libomp.dylib')

            if os.path.exists(torch_omp_header) and os.path.exists(torch_omp_dylib):
                print(f"Found OpenMP in PyTorch: {torch_omp_lib}")
                cflags += [f"-I{torch_omp_include}"]
                lflags += [f"-L{torch_omp_lib}", "-lomp"]
                omp_found = True
        except ImportError:
            pass

        # Fallback to Homebrew OpenMP
        if not omp_found:
            homebrew_omp_paths = [
                "/opt/homebrew/opt/libomp",  # ARM64 Homebrew
                "/usr/local/opt/libomp"      # Intel Homebrew (fallback)
            ]

            for homebrew_path in homebrew_omp_paths:
                omp_header = os.path.join(homebrew_path, "include", "omp.h")
                if os.path.exists(omp_header):
                    print(f"Found OpenMP in Homebrew: {homebrew_path}")
                    cflags += [f"-I{homebrew_path}/include"]
                    lflags += [f"-L{homebrew_path}/lib", "-lomp"]
                    omp_found = True
                    break

            if not omp_found:
                print("WARNING: OpenMP not found!")
                print("Please install with: brew install libomp")
                print("Or ensure PyTorch is installed with OpenMP support")

        # Use OpenBLAS for BLAS/LAPACK (provides LAPACKE interface)
        cflags += ["-DUSE_OPENBLAS_LP64", "-DUSE_OPENBLAS"]
        cflags += ["-I/opt/homebrew/opt/openblas/include"]
        lflags += ["-L/opt/homebrew/opt/openblas/lib", "-lopenblas"]
        skip_blas_detection = True

        print("Using OpenBLAS for BLAS/LAPACK (LAPACKE interface)")

    else:
        # Intel Mac
        print("=" * 60)
        print("Building for Intel macOS")
        print("=" * 60)

        cflags += ["-fopenmp", "-march=native"]
        lflags += ["-fopenmp"]

elif platform.system() == 'Linux':
    # Linux configuration
    print("=" * 60)
    print("Building for Linux")
    print("=" * 60)

    cflags += ["-fopenmp"]
    lflags += ["-fopenmp"]

    # For release builds, target Haswell (AVX2) for broad compatibility
    if "BUILD_H2PACK_RELEASE" in os.environ:
        cflags += ["-march=haswell"]
        print("Release build: targeting Haswell (AVX2) architecture")
    else:
        cflags += ["-march=native"]
        print("Development build: using native architecture")

elif platform.system() == 'Windows':
    # Windows/MSVC configuration
    print("=" * 60)
    print("Building for Windows")
    print("=" * 60)

    # MSVC uses different flags
    cflags = ["/O2", "/openmp"]
    lflags = []

else:
    print(f"Warning: Unknown platform {platform.system()}")

# Detect BLAS library (Linux and Intel Mac)
def get_numpy_linalg_lib():
    """Detect which BLAS library numpy is using."""
    pid = os.getpid()
    lsof_cmd = f'lsof -p {pid} | grep -E "libmkl_rt|libopenblas"'
    tmp_file = "/tmp/h2pack_numpy_load.txt"

    os.system(f'{lsof_cmd} > {tmp_file} 2>/dev/null')

    try:
        with open(tmp_file, 'r') as f:
            lsof_line = f.readline()
    except FileNotFoundError:
        return 0, 0, None, None
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    has_mkl = 0
    has_openblas = 0
    lib_path = None
    lib_name = None

    if "libmkl_rt" in lsof_line:
        has_mkl = 32  # MKL always uses LP64 (32-bit integer interface)
        print("Detected MKL BLAS library from NumPy")
    elif "libopenblas" in lsof_line and "libopenblas64" not in lsof_line:
        has_openblas = 32  # OpenBLAS LP64
        print("Detected OpenBLAS (LP64) from NumPy")
    elif "libopenblas64" in lsof_line:
        has_openblas = 64  # OpenBLAS ILP64
        print("Detected OpenBLAS (ILP64) from NumPy")

    # Extract library path
    for substr in lsof_line.split():
        substr = substr.strip()
        if os.path.isfile(substr) and ('libmkl' in substr or 'libopenblas' in substr):
            lib_path = substr
            lib_dir = os.path.dirname(lib_path)
            base_name = os.path.basename(lib_path)
            lib_name_noext = base_name.split(".so")[0]
            lib_name = lib_name_noext[3:]  # Remove 'lib' prefix
            print(f"BLAS library path: {lib_dir}")
            break

    return has_mkl, has_openblas, lib_dir, lib_name

if not skip_blas_detection:
    has_mkl, has_openblas, lib_dir, lib_name = get_numpy_linalg_lib()

    if lib_dir and lib_name:
        lflags += ["-L", lib_dir, "-l", lib_name]

        if has_mkl > 0:
            cflags += ["-DUSE_MKL"]
        elif has_openblas == 32:
            cflags += ["-DUSE_OPENBLAS_LP64", "-DUSE_OPENBLAS"]
        elif has_openblas == 64:
            cflags += ["-DUSE_OPENBLAS_ILP64", "-DUSE_OPENBLAS"]
    else:
        print("WARNING: Could not detect BLAS library!")
        print("Will attempt to build without explicit BLAS linking")

# Collect all H2Pack C sources (use relative paths)
h2pack_c_sources = sorted(glob('src/h2pack/*.c'))

# C extension wrapper (use relative paths)
extension_sources = [
    'src/h2pack_cext.c'
] + h2pack_c_sources

print(f"\nFound {len(h2pack_c_sources)} H2Pack C source files")
print(f"Total source files to compile: {len(extension_sources)}")

# Include directories (use relative paths where possible)
include_dirs = [
    numpy.get_include(),  # This must be absolute
    'src',
    'src/h2pack',
    'src/h2pack/ASTER/include',
]

print("\nCompiler flags:")
print(f"  CFLAGS: {' '.join(cflags)}")
print(f"  LDFLAGS: {' '.join(lflags)}")
print("=" * 60)

# Define the C extension module
h2pack_cext = setuptools.Extension(
    'h2pack._h2pack_cext',
    sources=extension_sources,
    include_dirs=include_dirs,
    extra_compile_args=cflags,
    extra_link_args=lflags,
    language="c"
)

# Read long description from README
readme_path = Path(workdir) / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = "Python interface for H2Pack hierarchical matrices"

# Setup configuration
setuptools.setup(
    name="h2pack",
    version="1.0.0",
    description="Python interface for H2Pack hierarchical matrices",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="H2Pack Team",
    author_email="h2pack@example.com",
    url="https://github.com/scalable-matrix/H2Pack",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.20.0"],
    python_requires=">=3.8",
    ext_modules=[h2pack_cext],  # C extension enabled
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
)

print("\nSetup configuration complete!")
print("=" * 60)
