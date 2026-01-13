/*
 * @file h2pack_cext.c
 * @brief Python C extension for H2Pack
 * @details Provides Python interface to H2Pack hierarchical matrix library
 */

#include "h2pack_cext.h"
#include <omp.h>

/*------------------------------------Helper Functions------------------------------------*/

/**
 * @brief Get kernel function pointer from kernel name
 */
static kernel_eval_fptr get_kernel_function(const char *kernel_name) {
    if (strcmp(kernel_name, "Gaussian") == 0 || strcmp(kernel_name, "gaussian") == 0) {
        return Gaussian_3D_eval_intrin_t;
    }
    else if (strcmp(kernel_name, "Matern32") == 0 || strcmp(kernel_name, "matern32") == 0) {
        return Matern32_3D_eval_intrin_t;
    }
    else if (strcmp(kernel_name, "Matern52") == 0 || strcmp(kernel_name, "matern52") == 0) {
        return Matern52_3D_eval_intrin_t;
    }
    else if (strcmp(kernel_name, "Coulomb") == 0 || strcmp(kernel_name, "coulomb") == 0) {
        return Coulomb_3D_eval_intrin_t;
    }
    else if (strcmp(kernel_name, "Quadratic") == 0 || strcmp(kernel_name, "quadratic") == 0) {
        return Quadratic_3D_eval_intrin_t;
    }
    // Add more kernels as needed
    return NULL;
}

/**
 * @brief Convert NumPy array to C array (row-major)
 */
static int numpy_to_c_array(PyArrayObject *np_array, double **c_array,
                             int *nrows, int *ncols) {
    if (np_array == NULL) {
        return -1;
    }

    int ndim = PyArray_NDIM(np_array);
    npy_intp *dims = PyArray_DIMS(np_array);

    if (ndim == 1) {
        *nrows = (int)dims[0];
        *ncols = 1;
    } else if (ndim == 2) {
        *nrows = (int)dims[0];
        *ncols = (int)dims[1];
    } else {
        PyErr_SetString(PyExc_ValueError, "Array must be 1D or 2D");
        return -1;
    }

    // Ensure array is contiguous and in C order
    if (!PyArray_IS_C_CONTIGUOUS(np_array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous");
        return -1;
    }

    // Ensure double precision
    if (PyArray_TYPE(np_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Array must be float64/double");
        return -1;
    }

    *c_array = (double*)PyArray_DATA(np_array);
    return 0;
}

/**
 * @brief Convert C array to NumPy array
 */
static PyArrayObject* c_to_numpy_array(double *c_array, int nrows, int ncols) {
    npy_intp dims[2];

    if (ncols == 1) {
        // Return 1D array
        dims[0] = nrows;
        PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (result == NULL) return NULL;

        double *data = (double*)PyArray_DATA(result);
        memcpy(data, c_array, nrows * sizeof(double));
        return result;
    } else {
        // Return 2D array
        dims[0] = nrows;
        dims[1] = ncols;
        PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (result == NULL) return NULL;

        double *data = (double*)PyArray_DATA(result);
        memcpy(data, c_array, nrows * ncols * sizeof(double));
        return result;
    }
}

/*------------------------------------H2Matrix Implementation------------------------------------*/

/**
 * @brief Initialize H2Matrix object
 */
static int H2Matrix_init(H2MatrixObject *self, PyObject *args, PyObject *kwds) {
    PyArrayObject *points_array = NULL;
    const char *kernel_name = "Gaussian";
    PyObject *kernel_params_obj = NULL;
    double rel_tol = 1e-6;
    int jit_mode = 1;
    int max_leaf_points = 400;
    double max_leaf_size = 0.0;
    int n_threads = -1;

    static char *kwlist[] = {"points", "kernel", "kernel_params", "rel_tol",
                              "jit_mode", "max_leaf_points", "max_leaf_size",
                              "n_threads", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|sOdiidi", kwlist,
                                      &PyArray_Type, &points_array,
                                      &kernel_name,
                                      &kernel_params_obj,
                                      &rel_tol,
                                      &jit_mode,
                                      &max_leaf_points,
                                      &max_leaf_size,
                                      &n_threads)) {
        return -1;
    }

    // Validate and copy points
    int nrows, ncols;
    double *points_c;
    if (numpy_to_c_array(points_array, &points_c, &nrows, &ncols) != 0) {
        return -1;
    }

    if (ncols > 3) {
        PyErr_SetString(PyExc_ValueError, "Point dimension must be <= 3");
        return -1;
    }

    // Store basic information
    self->n_points = nrows;
    self->dim = ncols;
    self->is_built = 0;
    self->rel_tol = rel_tol;
    self->BD_JIT = jit_mode;
    self->max_leaf_points = max_leaf_points;
    self->max_leaf_size = max_leaf_size;

    // Copy kernel name
    strncpy(self->kernel_name, kernel_name, sizeof(self->kernel_name) - 1);
    self->kernel_name[sizeof(self->kernel_name) - 1] = '\0';

    // Allocate and copy point coordinates (transpose to column-major for H2Pack)
    self->points = (double*)malloc(sizeof(double) * nrows * ncols);
    if (self->points == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for points");
        return -1;
    }

    // Transpose from row-major (Python/NumPy) to column-major (H2Pack)
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            self->points[j * nrows + i] = points_c[i * ncols + j];
        }
    }

    // Handle kernel parameters
    self->kernel_params = NULL;
    self->n_kernel_params = 0;
    self->h2pack_kernel_params = NULL;  // Will be set during build()

    if (kernel_params_obj != NULL && PyDict_Check(kernel_params_obj)) {
        // Extract lengthscale if present
        PyObject *lengthscale_obj = PyDict_GetItemString(kernel_params_obj, "lengthscale");
        if (lengthscale_obj != NULL) {
            double lengthscale = PyFloat_AsDouble(lengthscale_obj);
            if (PyErr_Occurred()) {
                free(self->points);
                return -1;
            }

            self->n_kernel_params = 1;
            self->kernel_params = (double*)malloc(sizeof(double) * self->n_kernel_params);
            if (self->kernel_params == NULL) {
                free(self->points);
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel params");
                return -1;
            }
            self->kernel_params[0] = lengthscale;
        }
    }

    // Set number of threads
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    // Initialize H2Pack structure (will be fully set up in build())
    self->h2pack = NULL;

    return 0;
}

/**
 * @brief Deallocate H2Matrix object
 */
static void H2Matrix_dealloc(H2MatrixObject *self) {
    if (self->points != NULL) {
        free(self->points);
        self->points = NULL;
    }

    if (self->kernel_params != NULL) {
        free(self->kernel_params);
        self->kernel_params = NULL;
    }

    if (self->h2pack_kernel_params != NULL) {
        free(self->h2pack_kernel_params);
        self->h2pack_kernel_params = NULL;
    }

    if (self->h2pack != NULL) {
        H2P_destroy(&self->h2pack);
        self->h2pack = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
 * @brief Build H2 matrix representation
 */
static PyObject* H2Matrix_build(H2MatrixObject *self, PyObject *args) {
    if (self->is_built) {
        PyErr_SetString(PyExc_RuntimeError, "H2 matrix already built");
        return NULL;
    }

    // Get kernel function
    kernel_eval_fptr krnl_eval = get_kernel_function(self->kernel_name);
    if (krnl_eval == NULL) {
        PyErr_Format(PyExc_ValueError, "Unknown kernel: %s", self->kernel_name);
        return NULL;
    }

    // Set kernel parameters
    double *krnl_param = self->kernel_params;
    int param_len = self->n_kernel_params;

    // Convert from Python lengthscale to H2Pack parameter
    // H2Pack Gaussian: exp(-param * r^2)
    // Standard Gaussian: exp(-r^2 / (2 * lengthscale^2))
    // So: H2Pack param = 1 / (2 * lengthscale^2)
    double default_param = 0.5;  // lengthscale = 1.0 -> param = 0.5
    double h2pack_param;

    if (krnl_param == NULL) {
        h2pack_param = default_param;
    } else {
        // krnl_param[0] is lengthscale from Python
        double lengthscale = krnl_param[0];
        h2pack_param = 1.0 / (2.0 * lengthscale * lengthscale);
    }

    // Allocate persistent storage for converted parameter
    if (self->h2pack_kernel_params != NULL) {
        free(self->h2pack_kernel_params);
    }
    self->h2pack_kernel_params = (double*)malloc(sizeof(double));
    if (self->h2pack_kernel_params == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel parameters");
        return NULL;
    }
    self->h2pack_kernel_params[0] = h2pack_param;

    // Set kernel dimension based on kernel type
    // For now, assume krnl_dim = 1 for scalar kernels
    self->krnl_dim = 1;

    // Create H2Pack structure
    // H2P_init(h2pack, pt_dim, krnl_dim, QR_stop_type, QR_stop_param)
    H2P_init(&self->h2pack, self->dim, self->krnl_dim, QR_REL_NRM, &self->rel_tol);

    // Build H2 representation
    H2P_calc_enclosing_box(self->dim, self->n_points, self->points, NULL, &self->h2pack->root_enbox);

    int max_leaf_points = self->max_leaf_points;
    DTYPE max_leaf_size = (DTYPE)self->max_leaf_size;
    H2P_partition_points(self->h2pack, self->n_points, self->points, max_leaf_points, max_leaf_size);

    // Generate proxy points
    H2P_dense_mat_p *pp = NULL;
    H2P_generate_proxy_point_ID_file(
        self->h2pack, self->h2pack_kernel_params, krnl_eval, NULL, &pp
    );

    // Build H2 representation
    int BD_JIT = self->BD_JIT;

    // We don't have kernel_bimv_fptr, pass NULL and 0
    H2P_build(
        self->h2pack, pp, BD_JIT, self->h2pack_kernel_params, krnl_eval, NULL, 0
    );

    self->is_built = 1;

    Py_RETURN_NONE;
}

/**
 * @brief Matrix-vector multiplication
 */
static PyObject* H2Matrix_matvec(H2MatrixObject *self, PyObject *args, PyObject *kwds) {
    PyArrayObject *x_array = NULL;
    static char *kwlist[] = {"x", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &x_array)) {
        return NULL;
    }

    if (!self->is_built) {
        PyErr_SetString(PyExc_RuntimeError, "Must call build() before matvec()");
        return NULL;
    }

    // Convert input array
    int nrows, ncols;
    double *x_data;
    if (numpy_to_c_array(x_array, &x_data, &nrows, &ncols) != 0) {
        return NULL;
    }

    if (nrows != self->n_points) {
        PyErr_Format(PyExc_ValueError,
                     "Input vector size mismatch: expected %d, got %d",
                     self->n_points, nrows);
        return NULL;
    }

    // Allocate output array
    double *y_data = (double*)malloc(sizeof(double) * self->n_points * ncols);
    if (y_data == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }

    // Perform matrix-vector multiplication
    // Note: H2P_matvec handles only single vectors, for multiple columns we need to loop
    if (ncols == 1) {
        H2P_matvec(self->h2pack, x_data, y_data);
    } else {
        // For multiple columns, perform matvec for each column
        for (int col = 0; col < ncols; col++) {
            H2P_matvec(self->h2pack,
                      x_data + col * self->n_points,
                      y_data + col * self->n_points);
        }
    }

    // Convert result to NumPy array
    PyArrayObject *result = c_to_numpy_array(y_data, self->n_points, ncols);
    free(y_data);

    return (PyObject*)result;
}

/**
 * @brief Matrix-matrix multiplication
 */
static PyObject* H2Matrix_matmul(H2MatrixObject *self, PyObject *args, PyObject *kwds) {
    // Similar to matvec but handles multiple columns
    return H2Matrix_matvec(self, args, kwds);
}

/**
 * @brief Get H2 matrix statistics
 */
static PyObject* H2Matrix_get_stats(H2MatrixObject *self, PyObject *args) {
    if (!self->is_built) {
        PyErr_SetString(PyExc_RuntimeError, "Must call build() before get_stats()");
        return NULL;
    }

    // Create dictionary for statistics
    PyObject *stats = PyDict_New();
    if (stats == NULL) return NULL;

    // Add basic information
    PyDict_SetItemString(stats, "n_points", PyLong_FromLong(self->n_points));
    PyDict_SetItemString(stats, "dim", PyLong_FromLong(self->dim));
    PyDict_SetItemString(stats, "is_built", PyBool_FromLong(self->is_built));

    // Add H2 structure information
    H2Pack_p h2pack = self->h2pack;
    if (h2pack != NULL) {
        PyDict_SetItemString(stats, "n_levels", PyLong_FromLong(h2pack->max_level + 1));
        PyDict_SetItemString(stats, "n_nodes", PyLong_FromLong(h2pack->n_node));

        // Calculate max rank from U matrices
        int max_rank = 0;
        if (h2pack->U != NULL) {
            for (int i = 0; i < h2pack->n_UJ; i++) {
                if (h2pack->U[i] != NULL && h2pack->U[i]->ncol > max_rank) {
                    max_rank = h2pack->U[i]->ncol;
                }
            }
        }
        PyDict_SetItemString(stats, "max_rank", PyLong_FromLong(max_rank));

        // Calculate average rank
        if (h2pack->n_UJ > 0 && h2pack->U != NULL) {
            int total_rank = 0;
            int count = 0;
            for (int i = 0; i < h2pack->n_UJ; i++) {
                if (h2pack->U[i] != NULL) {
                    total_rank += h2pack->U[i]->ncol;
                    count++;
                }
            }
            double avg_rank = (count > 0) ? (double)total_rank / count : 0.0;
            PyDict_SetItemString(stats, "avg_rank", PyFloat_FromDouble(avg_rank));
        }

        // Memory usage (rough estimate in MB)
        double storage_mb = 0.0;
        if (h2pack->mat_size[0] > 0) {
            // Sum up U, B, D matrix sizes
            storage_mb = (h2pack->mat_size[U_SIZE_IDX] +
                         h2pack->mat_size[B_SIZE_IDX] +
                         h2pack->mat_size[D_SIZE_IDX]) * sizeof(DTYPE) / (1024.0 * 1024.0);
        }
        PyDict_SetItemString(stats, "storage_mb", PyFloat_FromDouble(storage_mb));

        // Compression ratio
        double dense_size = (double)self->n_points * self->n_points * sizeof(double) / (1024.0 * 1024.0);
        double compression_ratio = (storage_mb > 0) ? dense_size / storage_mb : 0.0;
        PyDict_SetItemString(stats, "compression_ratio", PyFloat_FromDouble(compression_ratio));
    }

    return stats;
}

/**
 * @brief String representation
 */
static PyObject* H2Matrix_repr(H2MatrixObject *self) {
    const char *status = self->is_built ? "built" : "not built";
    return PyUnicode_FromFormat(
        "H2Matrix(n_points=%d, dim=%d, kernel=%s, status=%s)",
        self->n_points, self->dim, self->kernel_name, status
    );
}

/*------------------------------------Module Initialization------------------------------------*/

PyMODINIT_FUNC PyInit__h2pack_cext(void) {
    // Import NumPy array API
    import_array();

    // Create module
    PyObject *module = PyModule_Create(&h2pack_cext_module);
    if (module == NULL) return NULL;

    // Prepare H2Matrix type
    if (PyType_Ready(&H2MatrixType) < 0) return NULL;
    Py_INCREF(&H2MatrixType);
    if (PyModule_AddObject(module, "H2Matrix", (PyObject *) &H2MatrixType) < 0) {
        Py_DECREF(&H2MatrixType);
        Py_DECREF(module);
        return NULL;
    }

    // TODO: Add HSSMatrix type when implemented
    // if (PyType_Ready(&HSSMatrixType) < 0) return NULL;
    // ...

    return module;
}
