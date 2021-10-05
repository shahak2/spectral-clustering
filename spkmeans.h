#ifndef spkmeans_h
#define spkmeans_h

/* ========================== Constants  ========================== */

#define epsilon pow(10,-15)
#define max_iter_jacobi 100
#define max_iter_kmeans 300

/* ==================== Functions Declerations  ==================== */

double** Build_Weighted_Adjacency_Matrix(double **point_matrix, int N, int dim);
double** Build_Diagonal_Degree_Matrix(double **WAM, int N);
double** Normalized_Graph_Laplacian_Matrix(double **DDM, double **WAM, int N);
double** Finding_Eigenvalues_and_Eigenvectors(double **NGLM, int N);
void kmeans(double **matrix, int N, int K, double** centroids);
double **create_U(double **EVaM, double **EVM, int N, int K);
void renormalizing_matrix(double **matrix, int row_num, int column_num);
int Eigengap_Heuristic(double **NGLM, int N);
void delete_matrix(double **matrix);

#endif