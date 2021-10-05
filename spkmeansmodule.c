#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>  
#include "spkmeans.h"


static double** matrix_py_to_c(PyObject *matrix, int rows, int cols)
/*
    Input: a python object representing a flat matrix and its dimensions.
    Output: a 2D-matrix of doubles with similar values.
*/
{
    int i;
    double *c_flat_matrix, **c_matrix;
    PyObject *item;

    c_flat_matrix = malloc(sizeof(double) * rows * cols);
    c_matrix = malloc(sizeof(double*) * rows);

    if(c_flat_matrix == NULL || c_matrix == NULL)
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    for (i = 0; i < rows * cols; i++) 
    {
        item = PyList_GetItem(matrix, i);
        if (!PyFloat_Check(item))
            continue;
        c_flat_matrix[i] = PyFloat_AsDouble(item);
    }
    for (i = 0; i < rows; i++) 
    {
		c_matrix[i] = c_flat_matrix + i * cols;
	}

    return c_matrix;
}

static PyObject* matrix_c_to_py(double **matrix, int rows, int cols)
/*
    Input: a matrix of doubles and its dimensions.
    Output: a python object representing a flat matrix with similar values.
*/
{
    PyObject *py_matrix;
    int i, j;

    py_matrix = PyList_New(rows * cols);

    if (!py_matrix)
    {
        printf("An Error Has Occured");
        exit(-1);
    }

    for (i = 0; i < rows; i++) 
    {
        for(j = 0; j < cols; j++)
        {
            PyObject *num = PyFloat_FromDouble(matrix[i][j]);
            if (!num) 
            {
                printf("An Error Has Occured");
                exit(-1);
            }
            PyList_SET_ITEM(py_matrix, i*cols + j, num);
        }
    }
    return py_matrix;
}

static PyObject* wam(PyObject *self, PyObject *args)
/*
    A wrapper C-API function for WAM.
*/
{
    int N, dim;
    double **points, **WAM;
    PyObject *points_object, *py_WAM;

    if(!PyArg_ParseTuple(args, "Oii", &points_object, &N, &dim))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    
    points = matrix_py_to_c(points_object, N, dim);
    WAM = Build_Weighted_Adjacency_Matrix(points, N, dim);

    delete_matrix(points);

    py_WAM = matrix_c_to_py(WAM, N, N);
    delete_matrix(WAM);

    return py_WAM;
}

static PyObject* ddm(PyObject *self, PyObject *args)
/*
    A wrapper C-API function for DDG.
*/
{
    int N, dim;
    double **points, **WAM, **DDM;
    PyObject *points_object, *py_DDM;

    if(!PyArg_ParseTuple(args, "Oii", &points_object, &N, &dim))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
         
    points = matrix_py_to_c(points_object, N, dim);
    WAM = Build_Weighted_Adjacency_Matrix(points, N, dim);
    delete_matrix(points);
    DDM = Build_Diagonal_Degree_Matrix(WAM, N);
    delete_matrix(WAM);
    py_DDM = matrix_c_to_py(DDM, N, N);
    delete_matrix(DDM);

    return py_DDM;
}

static PyObject* lnorm(PyObject *self, PyObject *args)
/*
    A wrapper C-API function for L_norm (NGLM).
*/
{
    int N, dim;
    double **points, **WAM, **DDM, **NGLM;
    PyObject *points_object, *py_NGLM;

    if(!PyArg_ParseTuple(args, "Oii", &points_object, &N, &dim))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    
    points = matrix_py_to_c(points_object, N, dim);
    WAM = Build_Weighted_Adjacency_Matrix(points, N, dim);
    delete_matrix(points);
    DDM = Build_Diagonal_Degree_Matrix(WAM, N);
    NGLM = Normalized_Graph_Laplacian_Matrix(DDM, WAM, N);
    delete_matrix(WAM); 
    delete_matrix(DDM);
    py_NGLM = matrix_c_to_py(NGLM, N, N);
    delete_matrix(NGLM);

    return py_NGLM;
}

static PyObject* jacobi(PyObject *self, PyObject *args)
/*
    A wrapper C-API function for the Jacobi algorithm.
    input: a real symmetic matrix of size N x N.
    output: returns eigen matrix of size (N + 1) x N, 
    where the first row contains the eigenvalues and
    the following rows contain their corresponding eigenvectors.
*/
{
    int N, i, j;
    double **matrix, **eigen, **eigenvectors, *eigen_row;
    PyObject *matrix_object, *py_eigen;

    if(!PyArg_ParseTuple(args, "Oi", &matrix_object, &N))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    
    matrix = matrix_py_to_c(matrix_object, N, N);
    eigenvectors = Finding_Eigenvalues_and_Eigenvectors(matrix ,N);

    /* Creating the eigen matrix */
    eigen = malloc(sizeof(double*) * (N+1));
    eigen_row = malloc(sizeof(double) * N);
    if((eigen == NULL) || (eigen_row == NULL))
    {
        printf("An Error Has Occured");
        exit(-1);
    }

    for(i = 0; i < N; i++)
    {
        eigen_row[i] = matrix[i][i];
    }
    delete_matrix(matrix);
    eigen[0] = eigen_row;

    for(i = 0; i < N; i++)
    {
        eigen_row = malloc(sizeof(double) * N);
        if(eigen_row == NULL)
        {
            printf("An Error Has Occured");
            exit(-1);
        }
        for(j = 0; j < N; j++)
        {
            eigen_row[j] = eigenvectors[i][j];
        }
        eigen[i+1] = eigen_row;
    }
    delete_matrix(eigenvectors);
    py_eigen = matrix_c_to_py(eigen, N + 1, N);
    for(i = 0; i < N+1; i++)
    {
        free(eigen[i]);
    }
    free(eigen); 

    return py_eigen;
}

static PyObject* eigengapHeuristic(PyObject *self, PyObject *args)
/*
    A wrapper C-API function for Eigengap_Heuristic function.
*/
{
    int N, K;
    double **points;
    PyObject *points_object;

    if(!PyArg_ParseTuple(args, "Oi", &points_object, &N))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    
    points = matrix_py_to_c(points_object, N, N);

    K = Eigengap_Heuristic(points, N);
    delete_matrix(points);

    return Py_BuildValue("i", K);
}

static PyObject* get_T(PyObject *self, PyObject *args)
/*
    A wrapper C-API function which calculates and returns T,
    the normallized U matrix.
*/
{
    int N, K;
    double **eigenvalues, **eigenvectors, **T;

    PyObject *eigenvalues_object, *eigenvectors_object, *py_T;

    if(!PyArg_ParseTuple(args, "OOii", &eigenvectors_object, 
        &eigenvalues_object, &N, &K))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    
    eigenvectors = matrix_py_to_c(eigenvectors_object, N, N);
    eigenvalues = matrix_py_to_c(eigenvalues_object, N, N);
    T = create_U(eigenvalues, eigenvectors, N, K);
    delete_matrix(eigenvectors);
    delete_matrix(eigenvalues);
    renormalizing_matrix(T, N, K);
    py_T = matrix_c_to_py(T, N, K);
    delete_matrix(T);

    return py_T;
}

static PyObject* kmeanspp(PyObject *self, PyObject *args)
/*
    A wrapper C-API function implementing kmeans algoeithm
    using kmeans++.
*/
{
    int N, K;
    double **centroids, **T;
    PyObject *T_object, *centroids_object, *py_clusters;

    if(!PyArg_ParseTuple(args, "OOii", &T_object, &centroids_object, &N, &K))
    {
        printf("An Error Has Occured");
        exit(-1);
    }

    T = matrix_py_to_c(T_object, N, K);
    centroids = matrix_py_to_c(centroids_object, K, K);
    kmeans(T, N, K, centroids);
    delete_matrix(T);
    py_clusters = matrix_c_to_py(centroids, K, K);
    delete_matrix(centroids);

    return py_clusters;
}

//  ============================ BOILERPLATE CODE ============================

static PyMethodDef capiMethods[] = 
{
    {"WAM_c", (PyCFunction) wam, METH_VARARGS, 
        PyDoc_STR("Weighted Adjecncy Matrix")},
    {"DDM_c", (PyCFunction) ddm, METH_VARARGS, 
        PyDoc_STR("Diagonal Degree Matrix")},
    {"LNORM_c", (PyCFunction) lnorm, METH_VARARGS, 
        PyDoc_STR("Normalized Graph Laplacian Matrix")},
    {"JACOBI_c", (PyCFunction) jacobi, METH_VARARGS, 
        PyDoc_STR("Normalized Graph Laplacian Matrix")},
    {"Eigengap_Heuristic_c", (PyCFunction) eigengapHeuristic, METH_VARARGS, 
        PyDoc_STR("Eigengap Heuristic_c")},
    {"get_T_c", (PyCFunction) get_T, METH_VARARGS, 
        PyDoc_STR("Calculate normallized T")},
    {"KMEANS_c", (PyCFunction) kmeanspp, METH_VARARGS, 
        PyDoc_STR("Kmeans++ Algorithm")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans", 
    NULL, 
    -1,  
    capiMethods 
};

PyMODINIT_FUNC
PyInit_spkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) 
        return NULL;
    return m;
}