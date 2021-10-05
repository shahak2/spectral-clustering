#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "spkmeans.h"

/*  Legend: 
    WAM  - Weighted_Adjacency_Matrix
    DDM  - Diagonal_Degree_Matrix
    NGLM - Normalized_Graph_Laplacian_Matrix
    EVM  - Eigenvectors_Matrix
    EVaM - Eigenvalues_Matrix                       
*/

/* ======================= Auxiliary Functions ======================= */

void get_size(FILE *fd, int *N, int *dim)
/*
    This function receives a file descriptor to a file containing
    the data points and addresses for both N - the number of points 
    and dim - the dimension of a point.
    It updates N and dim.
*/
{
    char ch;
    double temp;
    int total_coordinates;
    rewind(fd);
    *dim = 1;
    while (1) /* Getting the dimension from the first data point. */
    {
        if ((ch = getc(fd)) == ',')
            *dim = *dim + 1;
        if (ch == EOF || ch == '\n')
            break;
    }
    total_coordinates = 1;
    while (fscanf(fd, "%lf%c", &temp, &ch) == 2)
    /* Counting all coordinates, but last. Hence, counting from 1. */
    {
        total_coordinates++;
    }
    /* Adding the point from which we got the dimension. */
    *N = 1 + (total_coordinates / (*dim));
}

double **init_matrix(int row_num, int column_num)
/*
    input: number of rows and columns, accordingly.
    output: returns a newly allocated matrix of doubles.
*/
{
    double **matrix, *data;
    int i;
    matrix = (double **)malloc(sizeof(double *) * row_num);
    data = (double *)malloc(sizeof(double) * row_num * column_num);
    if ((matrix == NULL) || (data == NULL))
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    for (i = 0; i < row_num; i++)
    {
        matrix[i] = data + i * column_num;
    }
    return matrix;
}

void delete_matrix(double **matrix)
/* 
    deletes a matrix initiated by init_matrix.
*/
{
    free(matrix[0]);
    free(matrix);
}

double **create_point_matrix(FILE *data, int N, int dim)
/*
    input: file descriptor to a file containing the data points, the number
    of points - N and their dimension - dim.
    output: returns a matrix containing the data points.
*/
{
    double** point_matrix;
    double coordinate;
    int row, column;
    point_matrix = init_matrix(N, dim);
    rewind(data);
    for (row = 0; row < N; row++)
    {
        for (column = 0; column < dim; column++)
        {
            if(fscanf(data, "%lf", &coordinate) < 0)
            {
                printf("An Error Has Occured");
                exit(-1);
            }
            point_matrix[row][column] = coordinate;
            getc(data);
        }
    }
    return point_matrix;
}

double** input_handler(int argc, char* argv[], int *N, int *K, int *dim)
/*
    This function validates the input, updates K - the number of clusters,
    N - the number of data points and dim - the dimension of a point.
    Lastly, it returns the data points matrix.
*/
{
    double **point_matrix;
    char *goal, *file_name, *k_str;
    FILE *fd;
    if (argc != 4)
    {
        printf("Invalid Input!");
        exit(-1);
    }
    goal = argv[2];
    if (strcmp(goal, "spk") != 0 && strcmp(goal, "wam") != 0 &&
        strcmp(goal, "ddg") != 0 && strcmp(goal, "lnorm") != 0 &&
        strcmp(goal, "jacobi") != 0)
    {
        printf("Invalid Input!");
        exit(-1);
    }
    file_name = argv[3];
    if ((fd = fopen(file_name, "r")) == NULL)
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    get_size(fd, N, dim); /* Setting N and dim */
    if (strcmp(goal, "spk") == 0)
    {
        k_str = argv[1];
        if ((atoi(k_str) < 0) || (*N <= atoi(k_str)))
        {
            printf("Invalid Input!");
            fclose(fd);
            exit(-1);
        }
        *K = atoi(k_str);
    }
    point_matrix = create_point_matrix(fd, *N, *dim);
    fclose(fd);
    return point_matrix;
}

double distance_squared(double *a, double *b, int dim)
/*  
    calculates the distance(squared) between two points of dimension dim. 
*/
{
    double dis = 0;
    int i;
    for (i = 0; i < dim; i++)
    {
        dis += pow((b[i] - a[i]), 2);
    }
    return dis;
}

double fix_zero(double d)
/* 
    prevents representation of a zero as a negative zero.
*/
{
    if(d==0)
        return 0;
    else
        return d;
}

void print_matrix(double **matrix, int row_num, int column_num)
/* 
    Given a matrix and its dimensions, prints it. 
*/
{
    int row, column;
    for (row = 0; row < row_num; row++)
    {
        for (column = 0; column < column_num - 1; column++)
        {
            printf("%.4f,", fix_zero(matrix[row][column]));
        }
        if(row==row_num-1){
            printf("%.4f", fix_zero(matrix[row][column_num - 1]));
        }
        else
        {
            printf("%.4f\n", fix_zero(matrix[row][column_num - 1]));
        }
    }
}

void print_jacobi(double** EVM, double** EVaM, int N)
/*
    prints the eigenvalues and eigenvectors.
*/
{
    int row, column;
    for(row = 0; row < N-1; row++)
    {
        printf("%.4f,", fix_zero(EVaM[row][row]));
    }
    printf("%.4f\n", fix_zero(EVaM[N-1][N-1]));
    /* prints the eigenvectors as rows - transpose of EVM matrix. */
    for(column = 0; column < N; column++)
    {
        for(row = 0; row < N-1; row++){
            printf("%.4f,", fix_zero(EVM[row][column]));
        }
        if(column==N-1){
            printf("%.4f", fix_zero(EVM[N-1][N-1]));
        }
        else{
            printf("%.4f\n", fix_zero(EVM[N-1][column]));
        }
    }
}

/* ======================= WAM Implementation ======================= */

double get_weight(double *pointA, double *pointB, int dim)
/*  
    input: two points.
    output: returns the weight between two points. 
*/
{
    double dis, weight;
    dis = sqrt(distance_squared(pointA, pointB, dim));
    weight = exp(-0.5 * dis);
    return weight;
}

double **Build_Weighted_Adjacency_Matrix(double **point_matrix, int N, int dim)
/*  
    input: a matrix of data points.
    output: returns its weighted adjacency matrix. 
*/
{
    double** WAM;
    int row, column;
    WAM = init_matrix(N, N);
    for (row = 0; row < N; row++)
    {
        WAM[row][row] = 0;
        for (column = row + 1; column < N; column++)
        {
            double weight = get_weight(point_matrix[row], 
                point_matrix[column], dim);
            WAM[row][column] = weight;
            WAM[column][row] = weight;
        }
    }
    return WAM;
}

/* ======================= DDM Implementation ======================= */

double** Build_Diagonal_Degree_Matrix(double **WAM, int N)
/*  
    input: a weighted adjacency matrix, and its number or rows/columns N.
    output: returns its diagonal degree matrix.
*/
{
    double** DDM, sum;
    int row, column;
    DDM = init_matrix(N, N);
    for (row = 0; row < N; row++)
    {
        sum = 0;
        for (column = 0; column < N; column++)
        {
            sum = sum + WAM[row][column];
            DDM[row][column] = 0;
        }
        DDM[row][row] = sum;
    }
    return DDM;
}

/* ======================= NGLM Implementation ======================= */

double* DDM_to_minus_half(double **DDM, int N)
/*  
    input: a diagonal degree matrix and its number of row/columns N.
    output: returns an array where for each i, array[i] = 1/sqrt(DDM[i][i]) 
*/
{
    double* arr;
    int i;
    arr = (double *)malloc(sizeof(double) * N);
    if (arr == NULL)
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    for (i = 0; i < N; i++)
    {
        arr[i] = 1 / (sqrt(DDM[i][i]));
    }
    return arr;
}

double** Normalized_Graph_Laplacian_Matrix(double **DDM, double **WAM, int N)
/*
    input: weighted adjacency matrix, its diagonal degree matrix
    and N, the number of its rows/columns.
    output: NGLM - return their normalized graph laplacian matrix.

    Mathematical Background:
    Denote DDM^(-1/2) as D'. Denote DWD = D' * WAM * D'. Hence, NGLM = I - DWD.
    DDM is symmetic, thus so is D'. We get the following formula: 
    DWD_i,j = (D'_i,i) * (WAM_i,j) * (D'_j,j)

    The Normalized Graph Laplacian Matrix is also symetric because DDM 
    and WAM are symetric (deduced from the mentioned formula).

    In addition, since the diagonal of WAM contains zeros, the diagonal 
    of DWD also contains zeros. Hence, the diagonal of NGLM is 1.
*/
{
    double **NGLM, *DDM_arr;
    int row, column;
    NGLM = init_matrix(N, N);
    DDM_arr = DDM_to_minus_half(DDM, N);
    
    for (row = 0; row < N; row++)
    {
        for (column = row + 1; column < N; column++)
        {
            NGLM[row][column] = -1 * DDM_arr[row] * 
                WAM[row][column] * DDM_arr[column];
            NGLM[column][row] = NGLM[row][column]; 
        }
        NGLM[row][row] = 1;
    }
    free(DDM_arr);
    return NGLM;
}

/* =================  Jacobi Algorithm Implementation ================= */

int sign(double num)
/*  
    retruns 1 if num >= 0, or -1 otherwise 
*/
{
    if(num < 0)
        return -1;
    return 1;
}

void calculate_sc(double **matrix, int row, int column, 
    double *s_address, double *c_address)
/*
    input: a matrix. row, column: the indices of element with the 
    biggest absolute that is out of the diagonal.
    s_address, c_address: 
    output: calculates and assigns s and c.
 */
{
    double theta, t, c, s;
    theta = (matrix[column][column] - matrix[row][row]) /
        (2 * matrix[row][column]);
    t = (sign(theta) / (fabs(theta) + sqrt(theta * theta + 1)));
    c = 1 / (sqrt(t * t + 1));
    s = t * c;
    *c_address = c;
    *s_address = s;
}

double calculate_offset(double** matrix, int N)
/*  
    input: a symetric matrix of size N x N. 
    output: returns the offset (to the power of 2) of the matrix.  

    As the matrix is symmetric, it is enough to consider elements above
    the diagonal and multiply by 2.
*/
{
    double sum;
    int row, column;
    sum = 0;
    for(row = 0; row < N; row++)
    {
        for(column = row + 1; column < N; column++)
        {
            sum = sum + 2*pow(matrix[row][column],2);
        }
    }
    return sum;
}

void update_EVaM(double **matrix, int N, int row_bav, int column_bav, double s,
    double c, double *new_off, double old_off)
/*
    input: matrix - the EVaM matrix of dimensions N x N.
    row_bav, column_bv - the indices for the biggest abolute value number, 
    (outside of the diagonal).
    s,c - the parameters as describe in the assigment.
    new_off - the address for the new offset.
    old_off - the current offset of matrix.

    output: updates Eigenvalues matrix, calculates the offset of the
    new matrix and assigns it to new_off.

    Notice: This function updates only the elements of the matrix
    above the diagonal, as the matrix is symmetric (for efficiency).
    Furthermore, it only updates the relevant indices (as explained 
    in step 1.2.1.6 of the assigment).
*/
{
    int r;
    double temp;
    /* the formula below is for the offset of the new matrix. 
        We dedeuce it by writing the square sum of the
        values has and the fact that (s^2 + c^2) = 1 */
    *new_off = old_off - 2 * pow(matrix[row_bav][column_bav], 2);
    r = 0;
    /*  for r < row_bav update the relevant indices above the diagonal. */
    while (r < row_bav)
    {
        temp = matrix[r][row_bav];
        matrix[r][row_bav] = c * temp - s * matrix[r][column_bav];
        matrix[r][column_bav] = c * matrix[r][column_bav] + s * temp;
        r++;
    }
    r = row_bav + 1;
    /*for row_bav + 1 < r < column_bav update the indices
     changed above the diagonal*/
    while (r < column_bav)
    {
        temp = matrix[row_bav][r];
        matrix[row_bav][r] = c * temp - s * matrix[r][column_bav];
        matrix[r][column_bav] = c * matrix[r][column_bav] + s * temp;
        r++;
    }
    r = column_bav + 1;
    /* for column_bav < r update the indexes changed above the diagonal */
    while (r < N)
    {
        temp = matrix[row_bav][r];
        matrix[row_bav][r] = c * temp - s * matrix[column_bav][r];
        matrix[column_bav][r] = c * matrix[column_bav][r] + s * temp;
        r++;
    }
    /* updates matrix in cells (i,j), (i,i), (j,j) */
    temp = matrix[row_bav][row_bav];
    matrix[row_bav][row_bav] = pow(c,2) * temp + 
        pow(s,2)* matrix[column_bav][column_bav] - 
            2 * s * c * matrix[row_bav][column_bav];
    matrix[column_bav][column_bav] = pow(s,2)* temp + 
        pow(c,2) * matrix[column_bav][column_bav] + 
            2 * s * c * matrix[row_bav][column_bav];
    matrix[row_bav][column_bav] = 0;
}

void find_biggest_abs_value(double **matrix, int N,
     int *row_bav, int *column_bav)
/*
    input: a symmetric matrix of size N x N, row_bav and column_bav.
    output: finds the biggest value outside the diagonal and saves its indices
    in row_bav and column_bav.
*/
{
    double biggest_abs;
    int biggest_abs_row, biggest_abs_column, row, column;
    /*the indexes of the current biggest value outside the diagonal*/
    biggest_abs_row = 0;
    biggest_abs_column = 1;
    biggest_abs = fabs(matrix[0][1]);
    /*  Since the matrix is symmetric it is enough to consider elements
        above the diagonal*/
    for (row = 0; row < N; row++)
    {
        for (column = row + 1; column < N; column++)
        {
            if (fabs(matrix[row][column]) > biggest_abs)
            {
                biggest_abs_row = row;
                biggest_abs_column = column;
                biggest_abs = fabs(matrix[row][column]);
            }
        }
    }
    *row_bav = biggest_abs_row;
    *column_bav = biggest_abs_column;
}

void update_EVM(double **EVM, int N, int s_row, 
    int s_column, double s, double c)
/*
    input: EVM: the current eigenvectors matrix. s_row, s_column: 
    row and column indices of s in P, respectively. s,c: the parameters of P.
    output: updates EVM by multiplying EVM (from right) by matrix P.

    Mathematical Background:
    In matrix P the diagonal contains 1's except P_s_row,s_row = 
    P_s_column,s_column = c. All the other values are zero, besides
    P_s_row,s_column = -(P_s_column,s_row) = s.

    Due to those properties, and as explained in the assignment,
    the only affected elements are the rows and columns with indices of
    s_row, s_column.
 */
{
    int i;
    double temp;
    for (i = 0; i < N; i++)
    {
        temp = EVM[i][s_row];
        EVM[i][s_row] = temp * c - EVM[i][s_column] * s;
        EVM[i][s_column] = temp * s + EVM[i][s_column] * c;
    }
}

/* ================== Spectral Clustering Implementation ================== */

double** Finding_Eigenvalues_and_Eigenvectors(double **NGLM, int N)
/*
    Implementing Jacobi Algorithm.

    input: a normalized graph laplacian matrix of size N x N.
    output: returns EVM - the eigenvalues matrix and updates NGLM
    so it will contain the eigenvectors.
*/
{
    int row, column, iter, row_bav, column_bav;
    double **EVM, old_off, new_off, s, c;
    /* Initializing EVM matrix as the identity matrix (I) */
    EVM = init_matrix(N, N);
    for (row = 0; row < N; row++)
    {
        for (column = 0; column < N; column++)
        {
            EVM[row][column] = 0;
        }
        EVM[row][row] = 1;
    }
    /*  
        old_off and new_off are the offsets of the previous A' and the new A',
        respectively. row_bav and column_bav are the indices of the
        row and column of the biggest absolute value outside the diagonal 
    */
    find_biggest_abs_value(NGLM, N, &row_bav, &column_bav);
    calculate_sc(NGLM, row_bav, column_bav, &s, &c);
    /* the first update/iteration of EVM is simple. */
    EVM[row_bav][row_bav] = c;
    EVM[row_bav][column_bav] = s;
    EVM[column_bav][row_bav] = -s;
    EVM[column_bav][column_bav] = c;
    old_off = calculate_offset(NGLM, N);
    update_EVaM(NGLM, N, row_bav, column_bav, s, c, &new_off, old_off);
    /* Notice new_off was also updated by update_EVaM. */
    iter = 1; 
    while (iter < max_iter_jacobi)
    {
        if (old_off - new_off <= epsilon)
            break;
        old_off = new_off;
        iter++;
        find_biggest_abs_value(NGLM, N, &row_bav, &column_bav);
        calculate_sc(NGLM, row_bav, column_bav, &s, &c);
        update_EVM(EVM, N, row_bav, column_bav, s, c);
        update_EVaM(NGLM, N, row_bav, column_bav, s, c, &new_off, old_off);
        /* Notice new_off was updated by update_EVaM. */
    }
    return EVM;
}

void renormalizing_matrix(double **matrix, int row_num, int column_num)
/*  
    input: a matrix to normalize and its dimensions.
    output: renormalizing the given matrix    
*/
{
    int row, column;
    double sum, norm;
    for (row = 0; row < row_num; row++)
    {
        sum = 0;
        for (column = 0; column < column_num; column++)
        {
            sum = sum + pow(matrix[row][column],2);
        }
        norm = sqrt(sum);
        for (column = 0; column < column_num; column++)
        {
            matrix[row][column] = matrix[row][column] / norm;
        }
    }
}

int comprator(const void *x, const void *y)
/*
    Compare function for qsort.  
    input: Two elements, each comprised of two doubles. 
    output: the function compares by the first value and then by second.
    It returns 0 if x = y. -1 if x < y. and 1 if x > y.
*/
{
    double *X, *Y;
    X = (double *)x;
    Y = (double *)y;
    if (X[0] == Y[0] && X[1] == Y[1])
        return 0;
    if (X[0] < Y[0] || (X[0] == Y[0] && X[1] < Y[1]))
        return -1;
    return 1;
}

double* eigenvalues_array(double **EVaM, int N)
/*
    input: EVaM, the eigenvalues matrix and N, the number of rows/columns.
    output: creates an array of size 2N, where an element
    can be viewed as a tuple: (eigenvalue, its column in the original EVaM).
    Sorts it first by eigenvalues then by column. Returns the sorted array.
*/
{
    double *eigenvalues;
    int i;
    eigenvalues = (double *)malloc(sizeof(double) * N * 2);
    if (eigenvalues == NULL)
    {
        printf("An Error Has Occured");
        exit(-1);
    }
    for (i = 0; i < N; i++)
    {
        eigenvalues[2 * i] = EVaM[i][i];
        eigenvalues[2 * i + 1] = i;
    }
    qsort(eigenvalues, N, 2 * sizeof(double), comprator);
    return eigenvalues;
}

double** create_U(double **EVaM, double **EVM, int N, int K)
/*
    input: EVaM - the eigenvalues matrix, EVM - the eigenvectors matrix, 
    N - the number of rows/columns and K - the number of clusters.
    output: returns the matrix U, as described in spk algorithm, step 4.
*/
{
    double *eigenvalues, **U;
    int column, column_EVaM, row;
    eigenvalues = eigenvalues_array(EVaM, N);
    U = init_matrix(N, K);
    for (column = 0; column < K; column++)
    {
        column_EVaM = (int)eigenvalues[2 * column + 1];
        for (row = 0; row < N; row++)
        {
            U[row][column] = EVM[row][column_EVaM];
        }
    }
    free(eigenvalues);
    return U;
}

int Eigengap_Heuristic(double **EVaM, int N)
/*
    input: Eigenvalues matrix, N - its number of rows/columns.
    output: return K, the optimal number of clusters.
*/
{
    double* eigenvalues, max_gap, gap;
    int i, opt_cluster_num;
    eigenvalues = eigenvalues_array(EVaM, N);
    opt_cluster_num = 1;
    max_gap = 0;
    for (i = 1; i <= floor(N / 2); i++)
    {
        gap = fabs(eigenvalues[2 * (i - 1)] - 
            eigenvalues[2 * (i - 1) + 2]);
        if (gap > max_gap)
        {
            max_gap = gap;
            opt_cluster_num = i;
        }
    }
    free(eigenvalues);
    return opt_cluster_num;
}

void kmeans(double **matrix, int N, int K, double** centroids)
/*
    input: data points matrix of size N x K and a matrix of initial centroids
    of size K x K.
    output: updates centroids to a matrix where every row
    represents the center point of each cluster.
*/
{
    double **new_cluster, min_distance, distance;
    int row, column, iter, change_flag, point_num, min_index, i;
    new_cluster = init_matrix(K, K + 1);
    for (row = 0; row < K; row++)
    {
        for (column = 0; column < K + 1; column++)
        {
            new_cluster[row][column] = 0;
        }
    }
    for (iter = 0; iter < max_iter_kmeans; iter++)
    {
        change_flag = 0; /*flag for convergence*/
        for (point_num = 0; point_num < N; point_num++)
        {
            min_distance = distance_squared(centroids[0], matrix[point_num], K);
            min_index = 0;
            for (i = 1; i < K; i++)
            {
                distance = distance_squared(centroids[i], matrix[point_num], K);
                if (min_distance < distance) 
                {
                    min_distance = distance;
                    min_index = i;
                }
            }
            for(i = 0; i < K; i++)/*add the current point to its new cluster*/
            {
                new_cluster[min_index][i] += matrix[point_num][i];
            }
            new_cluster[min_index][K]++;
        }
        for(row = 0; row < K; row++) /*calculate the new cluster.*/
        {
            if (new_cluster[row][K] != 0)
            {
                for(i = 0; i < K; i++)
                {
                    new_cluster[row][i] = (new_cluster[row][i] / 
                        new_cluster[row][K]);
                }
            }
        }
        for(row = 0; row < K; row++)
        { /* checking if clusters changed and reset the matrices*/
            for(column = 0; column < K; column++)
            {
                if (centroids[row][column] != new_cluster[row][column])
                {
                    centroids[row][column] = new_cluster[row][column];
                    change_flag = 1;
                }
                new_cluster[row][column] = 0;
            }
            new_cluster[row][K] = 0;
        }
        if(!change_flag)
            break;
    }
    delete_matrix(new_cluster);
}

double** kmeans_wrapper(double **matrix, int N, int K)
/*
    A wrapper function for the kmeans algorithm, which sets the
    initial centroids as the first K rows of the matrix.
*/
{   
    double **centroids;
    int row, column;

    centroids = init_matrix(K, K);
    for (row = 0; row < K; row++)
    {
        for (column = 0; column < K; column++)
        {
            centroids[row][column] = matrix[row][column];
        }
    }
    kmeans(matrix, N, K, centroids);
    return centroids;
}

/* =========================== Main Function =========================== */

int main(int argc, char *argv[])
{
    double** point_matrix, **WAM, **DDM, **NGLM, **eigenvalues,
    **eigenvectors, **ClusterM, **U;
    int N, K, dim;
    char* goal;

    point_matrix = input_handler(argc, argv, &N, &K, &dim);
    goal = argv[2];

    if(strcmp(goal,"jacobi") == 0)
    {
        eigenvectors = Finding_Eigenvalues_and_Eigenvectors(point_matrix, N);
        eigenvalues = point_matrix;
        print_jacobi(eigenvectors, eigenvalues, N);
        delete_matrix(eigenvectors);
        delete_matrix(eigenvalues);
        return 1;
    }
    WAM = Build_Weighted_Adjacency_Matrix(point_matrix, N, dim);
    delete_matrix(point_matrix);
    if(strcmp(goal, "wam") == 0)
    {
        print_matrix(WAM, N, N);
        delete_matrix(WAM);
        return 1;
    }
    DDM = Build_Diagonal_Degree_Matrix(WAM, N);
    if(strcmp(goal, "ddg") == 0)
    {
        print_matrix(DDM, N, N);
        delete_matrix(WAM);
        delete_matrix(DDM);
        return 1;
    }
    NGLM = Normalized_Graph_Laplacian_Matrix(DDM, WAM, N);
    delete_matrix(WAM);
    delete_matrix(DDM);
    if(strcmp(goal, "lnorm") == 0)
    {
        print_matrix(NGLM, N, N);
        delete_matrix(NGLM);
        return 1;
    }
    /* goal = "spk" */
    eigenvectors = Finding_Eigenvalues_and_Eigenvectors(NGLM, N);
    /* Notice NGLM from now on contains the eigenvalues */
    if(K == 0)
        K = Eigengap_Heuristic(NGLM, N);
    U = create_U(NGLM, eigenvectors, N, K); 
    /* U - the matrix described in step 4 */
    delete_matrix(eigenvectors);
    delete_matrix(NGLM);
    renormalizing_matrix(U, N, K); /* Getting T */
    ClusterM = kmeans_wrapper(U, N, K); /* ClusterM - the cluster matrix */
    delete_matrix(U);
    print_matrix(ClusterM, K, K);
    delete_matrix(ClusterM);
    return 1;
}