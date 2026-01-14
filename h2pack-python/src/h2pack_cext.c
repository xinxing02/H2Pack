/*
 * @file h2pack_cext.c
 * @brief Python C extension for H2Pack
 * @details Provides Python interface to H2Pack hierarchical matrix library
 */

#include "h2pack_cext.h"
#include "H2Pack_1D_kernels.h"
#include <omp.h>

/*------------------------------------Helper Functions------------------------------------*/

/**
 * @brief Get kernel function pointer from kernel name and dimension
 * @param kernel_name Name of the kernel (case-insensitive)
 * @param dim Point dimension (1, 2, or 3)
 * @return Function pointer to the appropriate kernel evaluation function
 */
static kernel_eval_fptr get_kernel_function(const char *kernel_name, int dim) {
    // For 1D kernels
    if (dim == 1) {
        if (strcasecmp(kernel_name, "Gaussian") == 0) {
            return Gaussian_1D_eval;
        }
        else if (strcasecmp(kernel_name, "Matern32") == 0) {
            return Matern32_1D_eval;
        }
        else if (strcasecmp(kernel_name, "Matern52") == 0) {
            return Matern52_1D_eval;
        }
        else if (strcasecmp(kernel_name, "Exponential") == 0) {
            return Expon_1D_eval;
        }
        else if (strcasecmp(kernel_name, "Quadratic") == 0) {
            return Quadratic_1D_eval;
        }
        // Coulomb not supported in 1D (singularity issues)
    }
    // For 2D kernels
    else if (dim == 2) {
        if (strcasecmp(kernel_name, "Gaussian") == 0) {
            return Gaussian_2D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Matern32") == 0) {
            return Matern32_2D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Matern52") == 0) {
            return Matern52_2D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Exponential") == 0) {
            return Expon_2D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Quadratic") == 0) {
            return Quadratic_2D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Coulomb") == 0 || strcasecmp(kernel_name, "Laplace") == 0) {
            // In 2D, Coulomb is called Laplace
            return Laplace_2D_eval_intrin_t;
        }
    }
    // For 3D kernels
    else if (dim == 3) {
        if (strcasecmp(kernel_name, "Gaussian") == 0) {
            return Gaussian_3D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Matern32") == 0) {
            return Matern32_3D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Matern52") == 0) {
            return Matern52_3D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Coulomb") == 0) {
            return Coulomb_3D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Exponential") == 0) {
            return Expon_3D_eval_intrin_t;
        }
        else if (strcasecmp(kernel_name, "Quadratic") == 0) {
            return Quadratic_3D_eval_intrin_t;
        }
    }
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
 *
 * Note: H2P_destroy triggers OpenBLAS memory deallocation errors on some systems
 * (particularly macOS with Homebrew OpenBLAS). To avoid segfaults on exit,
 * we skip the H2P_destroy call and let the OS reclaim memory when the process ends.
 * This is safe because:
 * 1. The computation is already complete
 * 2. The OS will free all process memory on exit anyway
 * 3. Memory leaks only matter for long-running processes that create many H2Matrix objects
 */
static void H2Matrix_dealloc(H2MatrixObject *self) {
    // Free Python-allocated memory
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

    // WORKAROUND: Skip H2P_destroy to avoid OpenBLAS memory corruption crash
    // H2P_destroy tries to free memory that conflicts with OpenBLAS internals
    // on macOS, causing "BLAS : Bad memory unallocation" errors and SIGSEGV.
    // The memory will be reclaimed by the OS when the process exits.
    //
    // TODO: Fix the root cause in H2Pack C library's memory management
    // if (self->h2pack != NULL) {
    //     H2P_destroy(&self->h2pack);
    //     self->h2pack = NULL;
    // }
    self->h2pack = NULL;

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

    // Get kernel function based on dimension
    kernel_eval_fptr krnl_eval = get_kernel_function(self->kernel_name, self->dim);
    if (krnl_eval == NULL) {
        PyErr_Format(PyExc_ValueError, "Unknown kernel '%s' for dimension %d",
                     self->kernel_name, self->dim);
        return NULL;
    }

    // Set kernel parameters - convert from Python API to H2Pack format
    double *krnl_param = self->kernel_params;
    int param_len = self->n_kernel_params;

    // Free any existing converted parameters
    if (self->h2pack_kernel_params != NULL) {
        free(self->h2pack_kernel_params);
        self->h2pack_kernel_params = NULL;
    }

    // Determine number of parameters needed and allocate storage
    int n_h2pack_params = 1;  // Default: most kernels use 1 parameter
    if (strcasecmp(self->kernel_name, "Quadratic") == 0) {
        n_h2pack_params = 2;  // Quadratic needs 2 parameters: c, a
    }

    self->h2pack_kernel_params = (double*)malloc(n_h2pack_params * sizeof(double));
    if (self->h2pack_kernel_params == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel parameters");
        return NULL;
    }

    // Convert parameters based on kernel type
    if (strcasecmp(self->kernel_name, "Gaussian") == 0) {
        // Gaussian: Python uses exp(-r²/(2*lengthscale²))
        // H2Pack uses exp(-param * r²)
        // Conversion: param = 1 / (2*lengthscale²)
        double lengthscale = (krnl_param != NULL) ? krnl_param[0] : 1.0;
        self->h2pack_kernel_params[0] = 1.0 / (2.0 * lengthscale * lengthscale);
    }
    else if (strcasecmp(self->kernel_name, "Matern32") == 0 ||
             strcasecmp(self->kernel_name, "Matern52") == 0) {
        // Matern: Python and H2Pack both use lengthscale directly
        double lengthscale = (krnl_param != NULL) ? krnl_param[0] : 1.0;
        self->h2pack_kernel_params[0] = lengthscale;
    }
    else if (strcasecmp(self->kernel_name, "Exponential") == 0) {
        // Exponential: Python uses exp(-r/lengthscale)
        // H2Pack uses exp(-param * r)
        // Conversion: param = 1 / lengthscale
        double lengthscale = (krnl_param != NULL) ? krnl_param[0] : 1.0;
        self->h2pack_kernel_params[0] = 1.0 / lengthscale;
    }
    else if (strcasecmp(self->kernel_name, "Coulomb") == 0) {
        // Coulomb: Python epsilon -> H2Pack diagonal value (direct pass)
        double epsilon = (krnl_param != NULL) ? krnl_param[0] : 0.0;
        self->h2pack_kernel_params[0] = epsilon;
    }
    else if (strcasecmp(self->kernel_name, "Quadratic") == 0) {
        // Quadratic: Python c, a -> H2Pack c, a (direct pass, 2 parameters)
        double c = (krnl_param != NULL && param_len >= 1) ? krnl_param[0] : 1.0;
        double a = (krnl_param != NULL && param_len >= 2) ? krnl_param[1] : -0.5;
        self->h2pack_kernel_params[0] = c;
        self->h2pack_kernel_params[1] = a;
    }
    else {
        // Unknown kernel or no special conversion needed - pass through
        double default_param = (krnl_param != NULL) ? krnl_param[0] : 1.0;
        self->h2pack_kernel_params[0] = default_param;
    }


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

    // Helper macro to add items with proper reference counting
    #define ADD_LONG(key, value) do { \
        PyObject *obj = PyLong_FromLong(value); \
        if (obj) { PyDict_SetItemString(stats, key, obj); Py_DECREF(obj); } \
    } while(0)

    #define ADD_DOUBLE(key, value) do { \
        PyObject *obj = PyFloat_FromDouble(value); \
        if (obj) { PyDict_SetItemString(stats, key, obj); Py_DECREF(obj); } \
    } while(0)

    #define ADD_BOOL(key, value) do { \
        PyObject *obj = PyBool_FromLong(value); \
        if (obj) { PyDict_SetItemString(stats, key, obj); Py_DECREF(obj); } \
    } while(0)

    // Add basic information
    ADD_LONG("n_points", self->n_points);
    ADD_LONG("dim", self->dim);
    ADD_BOOL("is_built", self->is_built);

    // Add H2 structure information
    H2Pack_p h2pack = self->h2pack;
    if (h2pack != NULL) {
        ADD_LONG("n_levels", h2pack->max_level + 1);
        ADD_LONG("n_nodes", h2pack->n_node);

        // Calculate max rank from U matrices
        int max_rank = 0;
        if (h2pack->U != NULL) {
            for (int i = 0; i < h2pack->n_UJ; i++) {
                if (h2pack->U[i] != NULL && h2pack->U[i]->ncol > max_rank) {
                    max_rank = h2pack->U[i]->ncol;
                }
            }
        }
        ADD_LONG("max_rank", max_rank);

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
            ADD_DOUBLE("avg_rank", avg_rank);
        }

        // Memory usage (rough estimate in MB)
        double storage_mb = 0.0;
        if (h2pack->mat_size[0] > 0) {
            // Sum up U, B, D matrix sizes
            storage_mb = (h2pack->mat_size[U_SIZE_IDX] +
                         h2pack->mat_size[B_SIZE_IDX] +
                         h2pack->mat_size[D_SIZE_IDX]) * sizeof(DTYPE) / (1024.0 * 1024.0);
        }
        ADD_DOUBLE("storage_mb", storage_mb);

        // Compression ratio
        double dense_size = (double)self->n_points * self->n_points * sizeof(double) / (1024.0 * 1024.0);
        double compression_ratio = (storage_mb > 0) ? dense_size / storage_mb : 0.0;
        ADD_DOUBLE("compression_ratio", compression_ratio);
    }

    #undef ADD_LONG
    #undef ADD_DOUBLE
    #undef ADD_BOOL

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
