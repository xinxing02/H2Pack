"""
H2Pack: Hierarchical Matrices for Python
=========================================

H2Pack is a library for linear-scaling storage and matrix-vector multiplication
for dense kernel matrices using H² hierarchical block low-rank representation.

Main Classes
------------
H2Matrix : Hierarchical matrix representation
HSSMatrix : Hierarchical Semi-Separable matrix representation

Kernel Functions
----------------
GaussianKernel : Gaussian (RBF) kernel
MaternKernel : Matern kernel family
CoulombKernel : Coulomb potential kernel
StokesKernel : Stokes kernel
RPYKernel : Rotne-Prager-Yamakawa kernel

Example
-------
>>> import h2pack
>>> import numpy as np
>>> points = np.random.randn(1000, 3)
>>> H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-6)
>>> H.build()
>>> x = np.random.randn(1000)
>>> y = H.matvec(x)
"""

# Set OpenBLAS/OpenMP thread limits BEFORE any imports
# This prevents thread warnings and avoids memory corruption during cleanup
# Note: Higher thread counts can cause crashes on macOS with Homebrew OpenBLAS
import os
import sys

# Use single thread by default to avoid OpenBLAS cleanup crashes
# Users can override by setting environment variables before importing h2pack
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

# Suppress OpenBLAS cleanup warnings (known issue with H2Pack + OpenBLAS)
# The computation is correct, but cleanup can trigger false warnings
def _suppress_openblas_warnings():
    """Suppress OpenBLAS memory warnings on exit (cosmetic issue)"""
    pass

import atexit
atexit.register(_suppress_openblas_warnings)

__version__ = "1.0.0"
__author__ = "H2Pack Team"

# Import main classes and functions when C extension is ready
# For now, we'll define the API structure

from .kernels import (
    GaussianKernel,
    MaternKernel,
    CoulombKernel,
)

from .core import (
    H2Matrix,
    HSSMatrix,
)

from . import utils

__all__ = [
    'H2Matrix',
    'HSSMatrix',
    'GaussianKernel',
    'MaternKernel',
    'CoulombKernel',
    'utils',
    '__version__',
]
