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
