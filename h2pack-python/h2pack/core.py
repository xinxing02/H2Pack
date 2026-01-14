"""
Core classes for H2Pack hierarchical matrices.

This module provides the main user-facing classes for working with
H2 and HSS matrix representations.
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from .kernels import Kernel, GaussianKernel, MaternKernel

# Try to import C extension
try:
    from . import _h2pack_cext
    _CEXT_AVAILABLE = True
except ImportError:
    _CEXT_AVAILABLE = False
    import warnings
    warnings.warn(
        "C extension not available. H2Pack functionality will be limited. "
        "Please rebuild the package with: python setup.py build_ext --inplace",
        RuntimeWarning
    )


class H2Matrix:
    """
    Hierarchical (H²) matrix representation.

    H2Matrix provides linear-scaling storage and matrix-vector multiplication
    for dense kernel matrices using hierarchical block low-rank representation.

    Parameters
    ----------
    points : ndarray, shape (n_points, dim)
        Point coordinates in d-dimensional space (d <= 3).
    kernel : str or Kernel, default='gaussian'
        Kernel function to use. Can be:
        - String: 'gaussian', 'matern32', 'matern52', 'exponential', 'coulomb', 'quadratic'
        - Kernel object: GaussianKernel(), MaternKernel(), etc.
    rel_tol : float, default=1e-6
        Relative error tolerance for H² compression.
    jit_mode : bool, default=True
        Use Just-In-Time mode for reduced memory usage.
    max_leaf_points : int, default=400
        Maximum number of points in each leaf node.
    max_leaf_size : float, default=0.0
        Maximum edge length of leaf box (0 = use max_leaf_points).
    kernel_params : dict, optional
        Additional kernel parameters (e.g., {'lengthscale': 1.0}).
    n_threads : int, default=-1
        Number of OpenMP threads (-1 = system default).
    proxy_surface : bool, default=False
        Use proxy surface method instead of proxy points.
    sample_points : bool, default=False
        Use data-driven sample point method.
    proxy_file : str, optional
        File to load/save proxy points.

    Attributes
    ----------
    shape : tuple
        Matrix shape (n_points, n_points).
    is_built : bool
        Whether H² representation has been constructed.
    stats : dict
        Statistics about the H² representation.

    Examples
    --------
    >>> import h2pack
    >>> import numpy as np
    >>> points = np.random.randn(10000, 3)
    >>> H = h2pack.H2Matrix(points, kernel='gaussian', rel_tol=1e-6)
    >>> H.build()
    >>> x = np.random.randn(10000)
    >>> y = H.matvec(x)
    >>> print(H.stats)
    """

    def __init__(
        self,
        points: np.ndarray,
        kernel: Union[str, Kernel] = 'gaussian',
        rel_tol: float = 1e-6,
        jit_mode: bool = True,
        max_leaf_points: int = 400,
        max_leaf_size: float = 0.0,
        kernel_params: Optional[Dict[str, Any]] = None,
        n_threads: int = -1,
        proxy_surface: bool = False,
        sample_points: bool = False,
        proxy_file: Optional[str] = None,
    ):
        # Validate inputs
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"points must be 2D array, got shape {points.shape}")
        if points.shape[1] < 1 or points.shape[1] > 3:
            raise ValueError(f"Only support 1D, 2D, or 3D points, got dimension {points.shape[1]}")

        self.points = points
        self.n_points = points.shape[0]
        self.dim = points.shape[1]

        # Handle kernel
        if isinstance(kernel, str):
            self.kernel = self._create_kernel(kernel, kernel_params or {})
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            raise TypeError(f"kernel must be str or Kernel, got {type(kernel)}")

        # H² parameters
        self.rel_tol = rel_tol
        self.jit_mode = jit_mode
        self.max_leaf_points = max_leaf_points
        self.max_leaf_size = max_leaf_size
        self.n_threads = n_threads
        self.proxy_surface = proxy_surface
        self.sample_points = sample_points
        self.proxy_file = proxy_file

        # State
        self.is_built = False
        self._h2_matrix = None  # Will hold C extension object

    def _create_kernel(self, kernel_name: str, params: Dict[str, Any]) -> Kernel:
        """Create kernel object from name and parameters."""
        kernel_name = kernel_name.lower()

        if kernel_name == 'gaussian':
            lengthscale = params.get('lengthscale', 1.0)
            return GaussianKernel(lengthscale=lengthscale)
        elif kernel_name in ('matern32', 'matern'):
            lengthscale = params.get('lengthscale', 1.0)
            nu = params.get('nu', 1.5)
            return MaternKernel(lengthscale=lengthscale, nu=nu)
        elif kernel_name == 'matern52':
            lengthscale = params.get('lengthscale', 1.0)
            return MaternKernel(lengthscale=lengthscale, nu=2.5)
        elif kernel_name == 'exponential':
            lengthscale = params.get('lengthscale', 1.0)
            from .kernels import ExponentialKernel
            return ExponentialKernel(lengthscale=lengthscale)
        else:
            from .kernels import CoulombKernel, StokesKernel, RPYKernel, QuadraticKernel
            kernel_map = {
                'coulomb': CoulombKernel,
                'stokes': StokesKernel,
                'rpy': RPYKernel,
                'quadratic': QuadraticKernel,
            }
            if kernel_name in kernel_map:
                return kernel_map[kernel_name](**params)
            else:
                raise ValueError(f"Unknown kernel: {kernel_name}")

    def build(self) -> 'H2Matrix':
        """
        Construct the H² matrix representation.

        This performs the following steps:
        1. Point partitioning into hierarchical tree
        2. Proxy point selection
        3. Low-rank basis construction
        4. Generator and coupling matrix construction

        Returns
        -------
        self : H2Matrix
            Returns self for method chaining.
        """
        if self.is_built:
            raise RuntimeError("H2 matrix already built. Create a new instance to rebuild.")

        if not _CEXT_AVAILABLE:
            raise RuntimeError(
                "C extension not available. Cannot build H² matrix. "
                "Please rebuild the package with: python setup.py build_ext --inplace"
            )

        # Convert kernel to format expected by C extension
        # Use kernel_type if available (for MaternKernel), otherwise use name
        kernel_name = getattr(self.kernel, 'kernel_type', self.kernel.name)
        kernel_params = self.kernel.params

        # Create C extension object
        self._h2_matrix = _h2pack_cext.H2Matrix(
            points=self.points,
            kernel=kernel_name,
            kernel_params=kernel_params,
            rel_tol=self.rel_tol,
            jit_mode=self.jit_mode,
            max_leaf_points=self.max_leaf_points,
            max_leaf_size=self.max_leaf_size,
            n_threads=self.n_threads
        )

        # Build H² representation
        self._h2_matrix.build()

        self.is_built = True
        return self

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication: y = H * x.

        Parameters
        ----------
        x : ndarray, shape (n_points,) or (n_points, n_vecs)
            Input vector or matrix.

        Returns
        -------
        y : ndarray, shape (n_points,) or (n_points, n_vecs)
            Result of H * x.
        """
        if not self.is_built:
            raise RuntimeError("Must call build() before matvec()")

        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n_points:
            raise ValueError(
                f"x shape mismatch: expected {self.n_points}, got {x.shape[0]}"
            )

        if not _CEXT_AVAILABLE:
            raise RuntimeError("C extension not available")

        return self._h2_matrix.matvec(x)

    def matmul(self, X: np.ndarray) -> np.ndarray:
        """
        Matrix-matrix multiplication: Y = H * X.

        Parameters
        ----------
        X : ndarray, shape (n_points, n_cols)
            Input matrix.

        Returns
        -------
        Y : ndarray, shape (n_points, n_cols)
            Result of H * X.
        """
        if not self.is_built:
            raise RuntimeError("Must call matmul() before matmul()")

        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != self.n_points:
            raise ValueError(
                f"X shape mismatch: expected {self.n_points}, got {X.shape[0]}"
            )

        if not _CEXT_AVAILABLE:
            raise RuntimeError("C extension not available")

        return self._h2_matrix.matmul(X)

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape."""
        return (self.n_points, self.n_points)

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the H² representation.

        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'n_points': Number of points
            - 'n_levels': Number of tree levels
            - 'n_nodes': Total number of nodes
            - 'avg_rank': Average compressed rank
            - 'max_rank': Maximum compressed rank
            - 'storage_mb': Storage in megabytes
            - 'compression_ratio': Compression ratio vs dense
            - 'build_time': Construction time in seconds
            - 'matvec_time': Average matvec time in seconds
        """
        if not self.is_built:
            return {'is_built': False}

        if not _CEXT_AVAILABLE:
            raise RuntimeError("C extension not available")

        return self._h2_matrix.get_stats()

    def __repr__(self) -> str:
        status = "built" if self.is_built else "not built"
        return (
            f"H2Matrix(n_points={self.n_points}, dim={self.dim}, "
            f"kernel={self.kernel.name}, status={status})"
        )


class HSSMatrix:
    """
    Hierarchical Semi-Separable (HSS) matrix representation.

    HSS matrices are a special case of H² matrices useful for solving
    linear systems via ULV factorization.

    Parameters
    ----------
    points : ndarray, shape (n_points, dim)
        Point coordinates in d-dimensional space (d <= 3).
    kernel : str or Kernel, default='gaussian'
        Kernel function to use.
    rel_tol : float, default=1e-6
        Relative error tolerance for HSS compression.
    max_leaf_points : int, default=400
        Maximum number of points in each leaf node.
    kernel_params : dict, optional
        Additional kernel parameters.
    n_threads : int, default=-1
        Number of OpenMP threads (-1 = system default).

    Attributes
    ----------
    shape : tuple
        Matrix shape (n_points, n_points).
    is_built : bool
        Whether HSS representation has been constructed.
    is_factorized : bool
        Whether ULV factorization has been performed.

    Examples
    --------
    >>> import h2pack
    >>> import numpy as np
    >>> points = np.random.randn(1000, 2)
    >>> H = h2pack.HSSMatrix(points, kernel='gaussian')
    >>> H.build()
    >>> H.factorize()
    >>> b = np.random.randn(1000)
    >>> x = H.solve(b)
    """

    def __init__(
        self,
        points: np.ndarray,
        kernel: Union[str, Kernel] = 'gaussian',
        rel_tol: float = 1e-6,
        max_leaf_points: int = 400,
        kernel_params: Optional[Dict[str, Any]] = None,
        n_threads: int = -1,
    ):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"points must be 2D array, got shape {points.shape}")

        self.points = points
        self.n_points = points.shape[0]
        self.dim = points.shape[1]

        # Handle kernel (reuse logic from H2Matrix)
        if isinstance(kernel, str):
            self.kernel = H2Matrix._create_kernel(None, kernel, kernel_params or {})
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            raise TypeError(f"kernel must be str or Kernel, got {type(kernel)}")

        self.rel_tol = rel_tol
        self.max_leaf_points = max_leaf_points
        self.n_threads = n_threads

        self.is_built = False
        self.is_factorized = False
        self._hss_matrix = None

    def build(self) -> 'HSSMatrix':
        """Construct the HSS matrix representation."""
        if self.is_built:
            raise RuntimeError("HSS matrix already built")

        # TODO: Call C extension
        raise NotImplementedError("C extension not yet implemented")

        self.is_built = True
        return self

    def factorize(self, method: str = 'cholesky') -> 'HSSMatrix':
        """
        Perform ULV factorization.

        Parameters
        ----------
        method : {'cholesky', 'lu'}, default='cholesky'
            Factorization method to use.

        Returns
        -------
        self : HSSMatrix
            Returns self for method chaining.
        """
        if not self.is_built:
            raise RuntimeError("Must call build() before factorize()")
        if self.is_factorized:
            raise RuntimeError("Already factorized")

        # TODO: Call C extension
        raise NotImplementedError("C extension not yet implemented")

        self.is_factorized = True
        return self

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system H * x = b.

        Parameters
        ----------
        b : ndarray, shape (n_points,) or (n_points, n_rhs)
            Right-hand side vector(s).

        Returns
        -------
        x : ndarray, shape (n_points,) or (n_points, n_rhs)
            Solution vector(s).
        """
        if not self.is_factorized:
            raise RuntimeError("Must call factorize() before solve()")

        b = np.asarray(b, dtype=np.float64)
        if b.shape[0] != self.n_points:
            raise ValueError(
                f"b shape mismatch: expected {self.n_points}, got {b.shape[0]}"
            )

        # TODO: Call C extension
        raise NotImplementedError("C extension not yet implemented")

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape."""
        return (self.n_points, self.n_points)

    def __repr__(self) -> str:
        status = []
        if self.is_built:
            status.append("built")
        if self.is_factorized:
            status.append("factorized")
        status_str = ", ".join(status) if status else "not built"

        return (
            f"HSSMatrix(n_points={self.n_points}, dim={self.dim}, "
            f"kernel={self.kernel.name}, status={status_str})"
        )
