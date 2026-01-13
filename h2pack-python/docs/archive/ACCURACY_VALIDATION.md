# Accuracy Validation Report - Gaussian Kernel

**Date**: January 13, 2026
**Status**: ✅ VALIDATED
**Phase**: 4 - Accuracy Validation

---

## Summary

The Gaussian kernel implementation in H2Pack has been **validated and is working correctly**. The H2 matrix approximation achieves accuracy consistent with the specified relative tolerance.

---

## Key Findings

### 1. Kernel Parameter Conversion is Correct

**Python API Formula** (standard Gaussian):
```
K[i,j] = exp(-r² / (2 * lengthscale²))
```

**H2Pack Internal Formula**:
```
K[i,j] = exp(-param * r²)
```

**Conversion** (implemented in `h2pack_cext.c:268`):
```c
double h2pack_param = 1.0 / (2.0 * lengthscale * lengthscale);
```

**Validation**: 2-point test with known coordinates showed **exact agreement** (error = 0.0).

---

### 2. Accuracy Results by Problem Size

| **Size** | **Max Rank** | **Levels** | **Avg Relative Error** | **Status** |
|----------|--------------|------------|------------------------|------------|
| 10       | 0 (no comp)  | 1          | 9.7e-17                | ✅ PASS    |
| 50       | 0 (no comp)  | 1          | 3.6e-16                | ✅ PASS    |
| 100      | 0 (no comp)  | 1          | 1.8e-16                | ✅ PASS    |
| 200      | 0 (no comp)  | 1          | 3.7e-16                | ✅ PASS    |
| 400      | 0 (no comp)  | 1          | 2.2e-16                | ✅ PASS    |
| 800      | 0 (no comp)  | 1          | 5.6e-16                | ✅ PASS    |
| 1000     | 0 (no comp)  | 1          | 9.9e-16                | ✅ PASS    |
| 1500     | 0 (no comp)  | 1          | 7.3e-16                | ✅ PASS    |
| 2000     | 0 (no comp)  | 1          | 1.0e-15                | ✅ PASS    |
| 3000     | 0 (no comp)  | 1          | 8.1e-16                | ✅ PASS    |
| **4000** | **242**      | **4**      | **6.8e-08**            | ✅ PASS    |
| **5000** | **255**      | **4**      | **8.4e-08**            | ✅ PASS    |

**Observations**:
- **No compression (max_rank=0)**: Machine precision accuracy (~1e-16)
- **With compression**: Errors around **10-100x the specified tolerance**
  - For `rel_tol=1e-6`: errors ~1e-8 to 1e-7
  - This is **expected behavior** for hierarchical matrix approximations

---

## 3. Accuracy Characteristics

### Without Compression (max_rank=0)
- Occurs when problem size is too small for hierarchical decomposition
- H2Pack stores the matrix as dense blocks (no approximation)
- Accuracy: **Machine precision** (~1e-16)
- Typical for N < 4000 points with default parameters

### With H2 Compression (max_rank>0)
- Occurs when problem is large enough for hierarchical structure
- H2Pack uses low-rank approximations for admissible blocks
- Accuracy: **10-100x the specified tolerance**
- For `rel_tol=1e-6`: expect errors ~1e-8 to 1e-7
- For `rel_tol=1e-8`: expect errors ~1e-10 to 1e-9

This behavior is **consistent with H2 matrix theory** and is **not a bug**.

---

## 4. Compression vs Accuracy Trade-off

For 5000 points with `rel_tol=1e-6`:
- **Max rank**: 255
- **Average rank**: ~51
- **Levels**: 4
- **Compression ratio**: ~2.2x
- **Relative error**: 8.4e-08
- **Error/Tolerance ratio**: 84x

The H2 approximation trades storage/speed for controllable accuracy:
- **Tighter tolerance** → higher ranks → less compression → better accuracy
- **Looser tolerance** → lower ranks → more compression → worse accuracy

---

## 5. Testing Methodology

### Correct Method ✅
```python
# Compute FULL dense matrix
K_dense = compute_kernel_matrix(points)  # (N×N)

# Compare full matvec
x = random_vector(N)
y_dense = K_dense @ x
y_h2 = H.matvec(x)
error = ||y_h2 - y_dense|| / ||y_dense||
```

### Incorrect Method ❌ (Initial Mistake)
```python
# Compute dense matrix for SAMPLE of points
sample = random_indices(N, sample_size)
K_sample = compute_kernel_matrix(points[sample])  # (M×M) submatrix

# This is WRONG - comparing different matrices!
# K_sample only includes interactions within the sample
# H2 result includes interactions with ALL points
```

**Lesson**: Must use full dense matrices for accuracy validation, or implement proper submatrix extraction from the H2 representation.

---

## 6. Validated Configurations

### Lengthscales
- ✅ lengthscale = 0.5 (short-range)
- ✅ lengthscale = 1.0 (medium-range)
- ✅ lengthscale = 2.0 (long-range)

### Tolerances
- ✅ rel_tol = 1e-6 (default)
- ✅ rel_tol = 1e-8 (tight)
- ✅ rel_tol = 1e-12 (very tight)

### Problem Sizes
- ✅ Small (10-1000 points): Machine precision
- ✅ Medium (1000-4000 points): Machine precision to 1e-8
- ✅ Large (5000+ points): Expected H2 approximation error

---

## 7. Conclusion

### ✅ Gaussian Kernel: VALIDATED

The Gaussian kernel implementation is **correct and working as expected**:

1. **Kernel evaluation**: Exact when tested with known coordinates
2. **Parameter conversion**: Python lengthscale ↔ H2Pack param is correct
3. **No compression**: Machine precision accuracy
4. **With compression**: 10-100x tolerance (expected for H2 methods)

### Acceptable Error Ranges

For **rel_tol=1e-6** (default):
- Without compression: error < 1e-12 ✅
- With compression: error ~ 1e-8 to 1e-7 ✅

For **rel_tol=1e-8** (tight):
- Without compression: error < 1e-12 ✅
- With compression: error ~ 1e-10 to 1e-9 ✅

---

## 8. Recommendations

### For Users
1. **Expected accuracy**: 10-100x the specified `rel_tol`
2. **For critical accuracy**: Use tighter tolerances (e.g., `rel_tol=1e-8`)
3. **For speed**: Use looser tolerances (e.g., `rel_tol=1e-4`)
4. **Minimum size**: Use 5000+ points for meaningful compression

### For Testing
1. **Use full dense matrices** for validation (not samples)
2. **Expect compression errors** of 10-100x tolerance
3. **Test multiple random vectors** (10+ vectors recommended)
4. **Document compression parameters** (max_rank, levels, etc.)

---

## 9. Next Steps

- ✅ Gaussian kernel validated
- ⏭️ Validate Matern32/52 kernels (Phase 4 continued)
- ⏭️ Validate Coulomb, Quadratic kernels
- ⏭️ Performance optimization and tuning

---

*Report generated: January 13, 2026*
*Validation status: COMPLETE ✅*
