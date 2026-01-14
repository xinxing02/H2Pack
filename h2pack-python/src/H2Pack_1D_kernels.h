#ifndef __H2PACK_1D_KERNELS_H__
#define __H2PACK_1D_KERNELS_H__

#include <math.h>
#include "h2pack/H2Pack_config.h"

#ifndef KRNL_EVAL_PARAM
#define KRNL_EVAL_PARAM \
    const DTYPE *coord0, const int ld0, const int n0, \
    const DTYPE *coord1, const int ld1, const int n1, \
    const void *param, DTYPE * __restrict mat, const int ldm
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================ //
// ====================   Gaussian Kernel   =================== //
// ============================================================ //

static void Gaussian_1D_eval(KRNL_EVAL_PARAM)
{
    const DTYPE *x0 = coord0;
    const DTYPE *x1 = coord1;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];

    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];

        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE r2 = dx * dx;
            mat_irow[j] = exp(neg_l * r2);
        }
    }
}

// ============================================================ //
// ==================   Exponential Kernel   ================== //
// ============================================================ //

static void Expon_1D_eval(KRNL_EVAL_PARAM)
{
    const DTYPE *x0 = coord0;
    const DTYPE *x1 = coord1;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE neg_l = -param_[0];

    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];

        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE r = fabs(dx);
            mat_irow[j] = exp(neg_l * r);
        }
    }
}

// ============================================================ //
// ===================   Matern 3/2 Kernel   ================== //
// ============================================================ //

#define NSQRT3 -1.7320508075688772

static void Matern32_1D_eval(KRNL_EVAL_PARAM)
{
    const DTYPE *x0 = coord0;
    const DTYPE *x1 = coord1;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE nsqrt3_l = NSQRT3 * param_[0];

    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];

        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE r = fabs(dx);
            DTYPE t = r * nsqrt3_l;
            mat_irow[j] = (1.0 - t) * exp(t);
        }
    }
}

// ============================================================ //
// ===================   Matern 5/2 Kernel   ================== //
// ============================================================ //

#define SQRT5 2.2360679774997898
#define NSQRT5 -2.2360679774997898

static void Matern52_1D_eval(KRNL_EVAL_PARAM)
{
    const DTYPE *x0 = coord0;
    const DTYPE *x1 = coord1;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE l = param_[0];
    const DTYPE nsqrt5_l = NSQRT5 * l;
    const DTYPE five_third_l2 = 5.0 / (3.0 * l * l);

    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];

        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE r = fabs(dx);
            DTYPE t = r * nsqrt5_l;
            DTYPE r2 = r * r;
            mat_irow[j] = (1.0 - t + five_third_l2 * r2) * exp(t);
        }
    }
}

// ============================================================ //
// ===================   Quadratic Kernel   =================== //
// ============================================================ //

static void Quadratic_1D_eval(KRNL_EVAL_PARAM)
{
    const DTYPE *x0 = coord0;
    const DTYPE *x1 = coord1;
    const DTYPE *param_ = (DTYPE*) param;
    const DTYPE c = param_[0];
    const DTYPE a = param_[1];

    for (int i = 0; i < n0; i++)
    {
        DTYPE *mat_irow = mat + i * ldm;
        const DTYPE x0_i = x0[i];

        for (int j = 0; j < n1; j++)
        {
            DTYPE dx = x0_i - x1[j];
            DTYPE r2 = dx * dx;
            mat_irow[j] = pow(1.0 + c * r2, a);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif
