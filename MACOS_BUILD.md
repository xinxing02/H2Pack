# Building H2Pack on macOS Apple Silicon

## The Apple Clang Problem

**Issue**: macOS ships with `gcc` at `/usr/bin/gcc`, but this is actually **Apple Clang**, not GCC.

```bash
$ which gcc
/usr/bin/gcc

$ gcc --version
Apple clang version 17.0.0
```

**Problems with Apple Clang + OpenMP**:
1. Requires Homebrew `libomp` library
2. Needs special flags: `-Xpreprocessor -fopenmp -lomp`
3. Known race conditions on Apple Silicon causing hangs/crashes
4. Required workaround: `OMP_NUM_THREADS=1` (single-threaded only)

**Solution**: Use Homebrew GCC instead, which has built-in OpenMP support.

## Prerequisites

```bash
# Install Homebrew GCC
brew install gcc

# Install OpenBLAS
brew install openblas

# Verify installations
/opt/homebrew/bin/gcc-15 --version  # Should show "Homebrew GCC"
ls /opt/homebrew/opt/openblas/lib/libopenblas.dylib
```

## Build Instructions

### Step 1: Build H2Pack Library

```bash
cd H2Pack/src

# Use Homebrew GCC explicitly (not /usr/bin/gcc!)
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make

# Verify build
ls -lh ../lib/libH2Pack.a ../lib/libH2Pack.so
```

### Step 2: Build Examples

```bash
cd ../examples

# Use same Homebrew GCC
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make example_H2.exe

# Verify
file example_H2.exe
# Should show: Mach-O 64-bit executable arm64
```

### Step 3: Run with OpenMP

```bash
# Set library paths
export DYLD_LIBRARY_PATH=$PWD/../lib:$DYLD_LIBRARY_PATH

# Configure threading
export OMP_NUM_THREADS=4         # H2Pack parallelism
export OPENBLAS_NUM_THREADS=1    # Avoid oversubscription

# Run example
echo "0" | ./example_H2.exe
```

## Makefile Configuration for macOS

The build system has been updated for macOS in:
- `H2Pack/src/GCC-OpenBLAS.make` - Updated OpenBLAS path
- `H2Pack/src/common.make` - Added Homebrew GCC detection and Apple Silicon optimization
- `H2Pack/examples/GCC-OpenBLAS.make` - Updated OpenBLAS path

The makefiles now automatically detect Homebrew GCC and apply:
- `-fopenmp` for OpenMP support
- `-mcpu=apple-m1` for Apple Silicon optimization (instead of generic `-march=native`)
- Proper warning suppressions

## Troubleshooting

### Problem: Still getting hangs with OMP_NUM_THREADS > 1

**Check your compiler**:
```bash
# Inside your build, check what compiler was actually used
otool -L ../lib/libH2Pack.dylib | grep -i omp

# If you see "libomp.dylib" (Apple's), you used Apple Clang
# If you see "libgomp" (GNU's), you used Homebrew GCC correctly
```

**Solution**: Clean rebuild with explicit CC path:
```bash
make clean
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make
```

### Problem: Compiler not found

```bash
# Find installed GCC versions
ls /opt/homebrew/bin/gcc-*

# Use the one you find (e.g., gcc-13, gcc-14, gcc-15)
CC=/opt/homebrew/bin/gcc-15 make -f GCC-OpenBLAS.make
```

### Problem: Library not found when running

```bash
# Make sure DYLD_LIBRARY_PATH is set correctly
export DYLD_LIBRARY_PATH=/path/to/H2Pack/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Verify the library can be found
otool -L ./example_H2.exe
```

## Performance Tips

1. **Set thread counts wisely**:
   ```bash
   # For Apple M1/M2 (8 performance cores)
   export OMP_NUM_THREADS=4  # Use half the performance cores

   # For Apple M1 Pro/Max/Ultra (10+ cores)
   export OMP_NUM_THREADS=8
   ```

2. **Disable BLAS threading**:
   ```bash
   export OPENBLAS_NUM_THREADS=1  # Always set this!
   ```
   H2Pack already parallelizes at a higher level. Nested parallelism causes oversubscription.

3. **Monitor performance**:
   ```bash
   # Time execution
   time ./example_H2.exe < /dev/null

   # Check speedup with different thread counts
   for t in 1 2 4 8; do
       echo "Threads: $t"
       OMP_NUM_THREADS=$t time ./example_H2.exe < /dev/null 2>&1 | grep real
   done
   ```

## Known Issues (Fixed)

### Issue #1: Variable Shadowing in H2Pack_matvec.c

**Location**: Line 1257
**Symptom**: Race condition causing hangs/crashes with multiple threads
**Status**: Fixed - variable renamed and reduction restructured

### Issue #2: Apple Clang Incompatibility

**Root cause**: Apple Clang + libomp has known threading issues on Apple Silicon
**Solution**: Use Homebrew GCC instead
**Status**: Addressed in this build guide

## References

- H2Pack GitHub: https://github.com/scalable-matrix/H2Pack
- OpenMP with Apple Clang: https://mac.r-project.org/openmp/
- Homebrew GCC: https://formulae.brew.sh/formula/gcc
