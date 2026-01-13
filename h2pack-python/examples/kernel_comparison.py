"""
Kernel Comparison Example

This script demonstrates how to use different kernel types with H2Pack
and compares their characteristics.

Author: H2Pack Development Team
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h2pack
import os
import time

# Set thread limits
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def compare_kernels(n_points=5000, lengthscale=1.0, rel_tol=1e-6):
    """
    Compare different kernel types.

    Parameters
    ----------
    n_points : int
        Number of points to use
    lengthscale : float
        Kernel lengthscale parameter
    rel_tol : float
        H2 relative tolerance
    """
    print("="*70)
    print(f"KERNEL COMPARISON: {n_points} points, lengthscale={lengthscale}")
    print("="*70)
    print()

    # Generate random points
    np.random.seed(42)
    points = np.random.randn(n_points, 3)

    # Test vector
    x = np.random.randn(n_points)

    # Kernels to test
    kernels = [
        ('gaussian', {'lengthscale': lengthscale}),
        ('matern32', {'lengthscale': lengthscale}),
        ('matern52', {'lengthscale': lengthscale}),
    ]

    results = []

    for kernel_name, kernel_params in kernels:
        print(f"\n{kernel_name.upper()} Kernel")
        print("-" * 70)

        try:
            # Build H2 matrix
            H = h2pack.H2Matrix(
                points,
                kernel=kernel_name,
                kernel_params=kernel_params,
                rel_tol=rel_tol,
                n_threads=4
            )

            t0 = time.time()
            H.build()
            build_time = time.time() - t0

            # Get statistics
            stats = H.stats

            # Test matvec
            t0 = time.time()
            y = H.matvec(x)
            matvec_time = time.time() - t0

            # Store results
            results.append({
                'kernel': kernel_name,
                'build_time': build_time,
                'matvec_time': matvec_time,
                'n_levels': stats['n_levels'],
                'max_rank': stats['max_rank'],
                'avg_rank': stats.get('avg_rank', 0),
                'compression': stats.get('compression_ratio', 0),
                'storage_mb': stats.get('storage_mb', 0),
                'result_norm': np.linalg.norm(y)
            })

            # Print results
            print(f"  Build time:      {build_time:.3f} seconds")
            print(f"  Matvec time:     {matvec_time*1000:.2f} milliseconds")
            print(f"  Structure:       {stats['n_levels']} levels, {stats['n_nodes']} nodes")
            print(f"  Ranks:           max={stats['max_rank']}, avg={stats.get('avg_rank', 0):.1f}")
            print(f"  Compression:     {stats.get('compression_ratio', 0):.2f}x")
            print(f"  Storage:         {stats.get('storage_mb', 0):.1f} MB")
            print(f"  Result norm:     {np.linalg.norm(y):.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary comparison
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Kernel':<12} {'Build(s)':<10} {'Matvec(ms)':<12} {'Rank':<8} {'Compression':<12}")
        print("-"*70)

        for r in results:
            print(f"{r['kernel']:<12} {r['build_time']:<10.3f} {r['matvec_time']*1000:<12.2f} "
                  f"{r['max_rank']:<8} {r['compression']:<12.2f}x")

        print("="*70)

def demonstrate_lengthscale_effect():
    """Demonstrate how lengthscale affects compression."""
    print("\n" * 2)
    print("="*70)
    print("LENGTHSCALE EFFECT ON COMPRESSION")
    print("="*70)
    print()

    n_points = 5000
    np.random.seed(42)
    points = np.random.randn(n_points, 3)

    lengthscales = [0.5, 1.0, 2.0, 4.0]

    print(f"{'Lengthscale':<15} {'Max Rank':<12} {'Compression':<15} {'Notes'}")
    print("-"*70)

    for l in lengthscales:
        H = h2pack.H2Matrix(
            points,
            kernel='gaussian',
            kernel_params={'lengthscale': l},
            rel_tol=1e-6,
            n_threads=4
        )
        H.build()

        stats = H.stats
        max_rank = stats['max_rank']
        compression = stats.get('compression_ratio', 0)

        if l <= 0.5:
            note = "Short-range, high compression"
        elif l <= 1.5:
            note = "Medium-range"
        else:
            note = "Long-range, low compression"

        print(f"{l:<15.1f} {max_rank:<12} {compression:<15.2f}x {note}")

    print("\nNote: Smaller lengthscales → shorter range → better compression")
    print("="*70)

def main():
    """Run all demonstrations."""
    print("\n")
    print("="*70)
    print("H2PACK KERNEL COMPARISON")
    print("="*70)
    print()
    print("This example demonstrates:")
    print("  1. Different kernel types (Gaussian, Matern 3/2, Matern 5/2)")
    print("  2. Performance characteristics of each kernel")
    print("  3. Effect of lengthscale on compression")
    print()

    # Main comparison
    compare_kernels(n_points=5000, lengthscale=1.0, rel_tol=1e-6)

    # Lengthscale effect
    demonstrate_lengthscale_effect()

    print("\n✓ All demonstrations complete!")
    print("\nNext steps:")
    print("  - Try different problem sizes")
    print("  - Experiment with different tolerances (rel_tol)")
    print("  - See examples/accuracy_demo.py for accuracy validation")
    print()

if __name__ == "__main__":
    main()
