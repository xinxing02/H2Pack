"""
Kernel functions for H2Pack.

This module provides various kernel functions that can be used with H2Matrix.
"""

import numpy as np
from typing import Optional, Dict, Any


class Kernel:
    """Base class for kernel functions."""

    def __init__(self, name: str):
        self.name = name
        self.params = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel.

    K(x, y) = exp(-||x - y||^2 / (2 * lengthscale^2))

    Parameters
    ----------
    lengthscale : float, default=1.0
        Length scale parameter for the kernel.
    """

    def __init__(self, lengthscale: float = 1.0):
        super().__init__("Gaussian")
        self.lengthscale = lengthscale
        self.params = {'lengthscale': lengthscale}


class MaternKernel(Kernel):
    """
    Matern kernel family.

    K(x, y) = (2^(1-nu) / Gamma(nu)) * (sqrt(2*nu) * r / l)^nu * K_nu(sqrt(2*nu) * r / l)

    where r = ||x - y||, l is the lengthscale, and K_nu is the modified Bessel function.

    Parameters
    ----------
    lengthscale : float, default=1.0
        Length scale parameter for the kernel.
    nu : float, default=1.5
        Smoothness parameter. Common values: 0.5, 1.5, 2.5
    """

    def __init__(self, lengthscale: float = 1.0, nu: float = 1.5):
        super().__init__("Matern")
        self.lengthscale = lengthscale
        self.nu = nu
        self.params = {'lengthscale': lengthscale, 'nu': nu}

        # Map nu to internal kernel names
        if np.isclose(nu, 1.5):
            self.kernel_type = "Matern32"
        elif np.isclose(nu, 2.5):
            self.kernel_type = "Matern52"
        else:
            raise ValueError(f"Currently only nu=1.5 and nu=2.5 are supported, got nu={nu}")


class CoulombKernel(Kernel):
    """
    Coulomb potential kernel.

    K(x, y) = 1 / ||x - y||

    Parameters
    ----------
    epsilon : float, default=0.0
        Regularization parameter to avoid singularity.
    """

    def __init__(self, epsilon: float = 0.0):
        super().__init__("Coulomb")
        self.epsilon = epsilon
        self.params = {'epsilon': epsilon}


class StokesKernel(Kernel):
    """
    Stokes kernel for fluid dynamics.

    Parameters
    ----------
    viscosity : float, default=1.0
        Fluid viscosity parameter.
    """

    def __init__(self, viscosity: float = 1.0):
        super().__init__("Stokes")
        self.viscosity = viscosity
        self.params = {'viscosity': viscosity}


class RPYKernel(Kernel):
    """
    Rotne-Prager-Yamakawa kernel for hydrodynamic interactions.

    Parameters
    ----------
    radius : float, default=1.0
        Particle radius.
    viscosity : float, default=1.0
        Fluid viscosity.
    periodic : bool, default=False
        Whether to use periodic boundary conditions.
    """

    def __init__(self, radius: float = 1.0, viscosity: float = 1.0,
                 periodic: bool = False):
        super().__init__("RPY")
        self.radius = radius
        self.viscosity = viscosity
        self.periodic = periodic
        self.params = {
            'radius': radius,
            'viscosity': viscosity,
            'periodic': periodic
        }


class ExponentialKernel(Kernel):
    """
    Exponential kernel.

    K(x, y) = exp(-||x - y|| / lengthscale)

    Parameters
    ----------
    lengthscale : float, default=1.0
        Length scale parameter for the kernel.
    """

    def __init__(self, lengthscale: float = 1.0):
        super().__init__("Exponential")
        self.lengthscale = lengthscale
        self.params = {'lengthscale': lengthscale}


class QuadraticKernel(Kernel):
    """
    Quadratic kernel.

    K(x, y) = (1 + c * ||x - y||^2)^a

    Parameters
    ----------
    c : float, default=1.0
        Scaling parameter.
    a : float, default=-0.5
        Power parameter.
    """

    def __init__(self, c: float = 1.0, a: float = -0.5):
        super().__init__("Quadratic")
        self.c = c
        self.a = a
        self.params = {'c': c, 'a': a}


class CustomKernel(Kernel):
    """
    Custom user-defined kernel function.

    Parameters
    ----------
    eval_func : callable
        Function that evaluates the kernel. Should have signature:
        eval_func(X, Y, params) -> K
        where X and Y are coordinate arrays and K is the kernel matrix.
    params : dict
        Parameters for the kernel function.
    name : str, default='Custom'
        Name for the custom kernel.
    """

    def __init__(self, eval_func, params: Optional[Dict[str, Any]] = None,
                 name: str = "Custom"):
        super().__init__(name)
        self.eval_func = eval_func
        self.params = params or {}

    def evaluate(self, X, Y):
        """Evaluate custom kernel function."""
        return self.eval_func(X, Y, self.params)
