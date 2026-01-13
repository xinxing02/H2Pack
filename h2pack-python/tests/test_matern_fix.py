"""Quick test for Matern kernel fix"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

print("Testing Matern32 kernel...")
np.random.seed(42)
points = np.random.randn(100, 3)

try:
    H = h2pack.H2Matrix(
        points,
        kernel='matern32',
        kernel_params={'lengthscale': 1.0},
        rel_tol=1e-6,
        n_threads=4
    )
    H.build()
    print("✓ Matern32 builds successfully!")

    # Test matvec
    x = np.ones(100)
    y = H.matvec(x)
    print(f"✓ Matvec works: result norm = {np.linalg.norm(y):.6f}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTesting Matern52 kernel...")
try:
    H2 = h2pack.H2Matrix(
        points,
        kernel='matern52',
        kernel_params={'lengthscale': 1.0},
        rel_tol=1e-6,
        n_threads=4
    )
    H2.build()
    print("✓ Matern52 builds successfully!")

    x = np.ones(100)
    y = H2.matvec(x)
    print(f"✓ Matvec works: result norm = {np.linalg.norm(y):.6f}")
except Exception as e:
    print(f"✗ Failed: {e}")
