#ifndef __H2PACK_CEXT_H__
#define __H2PACK_CEXT_H__

/*
 * @file h2pack_cext.h
 * @brief Header file for H2Pack Python C extension module
 * @details This file defines the Python C extension interface for H2Pack,
 *          providing hierarchical matrix functionality to Python.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include H2Pack headers
#include "h2pack/H2Pack.h"
#include "h2pack/H2Pack_config.h"
#include "h2pack/H2Pack_typedef.h"
#include "h2pack/H2Pack_kernels.h"

/*------------------------------------H2Matrix Object------------------------------------*/

/**
 * @brief Python object for H2 matrix
 * @details Wraps H2Pack_p structure for use in Python
 */
typedef struct {
    PyObject_HEAD

    // H2Pack structure
    H2Pack_p h2pack;

    // Matrix parameters
    int n_points;           // Number of points
    int dim;                // Point dimension
    int krnl_dim;           // Kernel output dimension

    // Build status
    int is_built;           // Whether H2 matrix has been built

    // Kernel information
    char kernel_name[64];   // Kernel function name
    double *kernel_params;  // Kernel parameters (Python)
    int n_kernel_params;    // Number of kernel parameters
    double *h2pack_kernel_params;  // Converted parameters for H2Pack

    // Point coordinates (owned by this object)
    double *points;         // Point coordinates (n_points * dim)

    // H2 parameters
    double rel_tol;         // Relative tolerance
    int BD_JIT;             // JIT mode flag
    int max_leaf_points;    // Max points per leaf
    double max_leaf_size;   // Max leaf box size

} H2MatrixObject;

/**
 * @brief Initialize H2Matrix object
 */
static int H2Matrix_init(H2MatrixObject *self, PyObject *args, PyObject *kwds);

/**
 * @brief Deallocate H2Matrix object
 */
static void H2Matrix_dealloc(H2MatrixObject *self);

/**
 * @brief Build H2 matrix representation
 */
static PyObject* H2Matrix_build(H2MatrixObject *self, PyObject *args);

/**
 * @brief Matrix-vector multiplication
 */
static PyObject* H2Matrix_matvec(H2MatrixObject *self, PyObject *args, PyObject *kwds);

/**
 * @brief Matrix-matrix multiplication
 */
static PyObject* H2Matrix_matmul(H2MatrixObject *self, PyObject *args, PyObject *kwds);

/**
 * @brief Get H2 matrix statistics
 */
static PyObject* H2Matrix_get_stats(H2MatrixObject *self, PyObject *args);

/**
 * @brief String representation
 */
static PyObject* H2Matrix_repr(H2MatrixObject *self);

// Method definitions for H2Matrix
static PyMethodDef H2Matrix_methods[] = {
    {"build", (PyCFunction) H2Matrix_build, METH_NOARGS,
     "Build the H2 matrix representation"},
    {"matvec", (PyCFunction) H2Matrix_matvec, METH_VARARGS | METH_KEYWORDS,
     "Matrix-vector multiplication: y = H * x"},
    {"matmul", (PyCFunction) H2Matrix_matmul, METH_VARARGS | METH_KEYWORDS,
     "Matrix-matrix multiplication: Y = H * X"},
    {"get_stats", (PyCFunction) H2Matrix_get_stats, METH_NOARGS,
     "Get H2 matrix statistics"},
    {NULL}
};

// H2Matrix type object
static PyTypeObject H2MatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "h2pack._h2pack_cext.H2Matrix",
    .tp_doc = "H2 matrix object",
    .tp_basicsize = sizeof(H2MatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) H2Matrix_init,
    .tp_dealloc = (destructor) H2Matrix_dealloc,
    .tp_repr = (reprfunc) H2Matrix_repr,
    .tp_methods = H2Matrix_methods,
};

/*------------------------------------HSSMatrix Object------------------------------------*/

/**
 * @brief Python object for HSS matrix
 */
typedef struct {
    PyObject_HEAD

    H2Pack_p h2pack;

    int n_points;
    int dim;
    int is_built;
    int is_factorized;

    char kernel_name[64];
    double *kernel_params;
    int n_kernel_params;

    double *points;
    double rel_tol;
    int max_leaf_points;

} HSSMatrixObject;

/**
 * @brief Initialize HSSMatrix object
 */
static int HSSMatrix_init(HSSMatrixObject *self, PyObject *args, PyObject *kwds);

/**
 * @brief Deallocate HSSMatrix object
 */
static void HSSMatrix_dealloc(HSSMatrixObject *self);

/**
 * @brief Build HSS matrix representation
 */
static PyObject* HSSMatrix_build(HSSMatrixObject *self, PyObject *args);

/**
 * @brief Perform ULV factorization
 */
static PyObject* HSSMatrix_factorize(HSSMatrixObject *self, PyObject *args, PyObject *kwds);

/**
 * @brief Solve linear system
 */
static PyObject* HSSMatrix_solve(HSSMatrixObject *self, PyObject *args, PyObject *kwds);

// Method definitions for HSSMatrix
static PyMethodDef HSSMatrix_methods[] = {
    {"build", (PyCFunction) HSSMatrix_build, METH_NOARGS,
     "Build the HSS matrix representation"},
    {"factorize", (PyCFunction) HSSMatrix_factorize, METH_VARARGS | METH_KEYWORDS,
     "Perform ULV factorization"},
    {"solve", (PyCFunction) HSSMatrix_solve, METH_VARARGS | METH_KEYWORDS,
     "Solve linear system H * x = b"},
    {NULL}
};

// HSSMatrix type object
static PyTypeObject HSSMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "h2pack._h2pack_cext.HSSMatrix",
    .tp_doc = "HSS matrix object",
    .tp_basicsize = sizeof(HSSMatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) HSSMatrix_init,
    .tp_dealloc = (destructor) HSSMatrix_dealloc,
    .tp_methods = HSSMatrix_methods,
};

/*------------------------------------Module Definition------------------------------------*/

// Module methods (if any module-level functions needed)
static PyMethodDef module_methods[] = {
    {NULL}
};

// Module definition
static struct PyModuleDef h2pack_cext_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_h2pack_cext",
    .m_doc = "H2Pack C extension module for hierarchical matrices",
    .m_size = -1,
    .m_methods = module_methods,
};

/*------------------------------------Helper Functions------------------------------------*/

/**
 * @brief Get kernel function pointer from kernel name
 */
static kernel_eval_fptr get_kernel_function(const char *kernel_name);

/**
 * @brief Setup kernel parameters for H2Pack
 */
static int setup_kernel_params(H2Pack_p h2pack, const char *kernel_name,
                                double *params, int n_params);

/**
 * @brief Convert NumPy array to C array
 */
static int numpy_to_c_array(PyArrayObject *np_array, double **c_array,
                             int *nrows, int *ncols);

/**
 * @brief Convert C array to NumPy array
 */
static PyArrayObject* c_to_numpy_array(double *c_array, int nrows, int ncols);

#endif // __H2PACK_CEXT_H__
