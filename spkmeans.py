import sys
import numpy as np
import pandas as pd
import spkmeans as spkm

np.random.seed(0)  # Setting randomness seed according to the instructions.


def distance(x, y):
    # Input: two points of unknown dimensions.
    # Output: The distance between them(squared).
    dist = 0
    for i in range(len(x)):
        dist = dist + (x[i] - y[i]) ** 2
    return dist


def kmeans_pp(data_points_list, K):
    # Input: N datapoints and K < N
    # Output: returns K initial centroids and their inidices.
    Z = 1
    indices_list = [x for x in range(0, len(data_points_list))]
    rand_centroid_index = np.random.choice(indices_list)
    rand_indices_list = [rand_centroid_index]
    centroids = [data_points_list[rand_centroid_index]]
    while Z < K:
        D_sum = 0
        D_i = [0 for x in range(len(data_points_list))]
        for point_index in range(len(data_points_list)):
            D_i[point_index] = distance(data_points_list[point_index],
                                        centroids[0])
            for centroid_index in range(1, Z):
                D_i[point_index] = min(D_i[point_index],
                                       distance(data_points_list[point_index],
                                                centroids[centroid_index]))
            D_sum += D_i[point_index]
        Z += 1
        # Appends randomly according to the weighted probability.
        rand_centroid_index = np.random.choice(indices_list,
                                               p=[di / D_sum for di in D_i])
        rand_indices_list.append(rand_centroid_index)
        centroids.append(data_points_list[rand_centroid_index])
    return rand_indices_list, centroids


def input_handler():
    # This function handles and validates the input.
    goals_set = {"spk", "wam", "ddg", "lnorm", "jacobi"}
    args = sys.argv
    assert (len(args) == 4), "Invalid Input!"
    try:
        K = int(args[1])
        goal = str(args[2])
        file = str(args[3])
        assert (goal in goals_set), "Invalid Input!"
    except ValueError:
        print("Invalid Input!")
        raise SystemExit(0)
    df = pd.read_csv(file, header=None, delimiter=",", dtype=np.float64)
    assert (0 <= K < len(df)), "Invalid Input!"
    return K, goal, df


def print_matrix(matrix):
    row_num = len(matrix)
    col_num = len(matrix[0])
    for row in range(row_num):
        line = ""
        for col in range(col_num - 1):
            if matrix[row][col] == 0:
                matrix[row][col] = 0
            line += "{x:.4f}".format(x=matrix[row][col]) + ','
        if matrix[row][col_num - 1] == 0:
            matrix[row][len(matrix[row]) - 1] = 0
        line += "{x:.4f}".format(x=matrix[row][col_num - 1])
        if row == row_num - 1:
            print(line, end="")
        else:
            print(line)


def main():
    K, goal, data_points = input_handler()
    N = len(data_points)
    dim = len(data_points.columns)
    flat_data_points = data_points.to_numpy().flatten().tolist()

    if goal == "jacobi":
        jacobi_matrix = spkm.JACOBI_c(flat_data_points, N)
        eigenvalues = np.array([jacobi_matrix[0:N]])
        eigenvectors = np.array(jacobi_matrix[N:]).reshape(N, N).T
        np_JACOBI = np.concatenate((eigenvalues, eigenvectors))
        np_JACOBI = np_JACOBI.round(decimals=4)
        print_matrix(np_JACOBI)
        return
    if goal == "wam":
        wam_matrix = spkm.WAM_c(flat_data_points, N, dim)
        np_WAM = np.array(wam_matrix).reshape(N, N)
        print_matrix(np_WAM)
    else:
        if goal == "ddg":
            ddg_matrix = spkm.DDM_c(flat_data_points, N, dim)
            np_DDG = np.array(ddg_matrix).reshape(N, N)
            print_matrix(np_DDG)
        else:
            NGLM_matrix = spkm.LNORM_c(flat_data_points, N, dim)
            if goal == "lnorm":
                np_NGLM = np.array(NGLM_matrix).reshape(N, N)
                print_matrix(np_NGLM)
            else:  # ~~~~ goal = spk ~~~~
                jacobi_matrix = spkm.JACOBI_c(NGLM_matrix, N)
                eigenvalues = np.zeros((N, N))
                np.fill_diagonal(eigenvalues, jacobi_matrix[0:N])
                eigenvalues = eigenvalues.flatten().tolist()
                eigenvectors = np.array(jacobi_matrix[N:]).reshape(N, N)
                eigenvectors = eigenvectors.flatten().tolist()
                if K == 0:
                    K = spkm.Eigengap_Heuristic_c(eigenvalues, N)
                T = spkm.get_T_c(eigenvectors, eigenvalues, N, K)
                centroids_indices, initial_centroids = \
                    kmeans_pp(np.array(T).reshape(N, K), K)
                flat_centroids = np.array(initial_centroids).flatten().tolist()
                centroids = spkm.KMEANS_c(T, flat_centroids, N, K)
                print(",".join([str(x) for x in centroids_indices]))
                print_matrix(np.array(centroids).reshape(K, K))


if __name__ == "__main__":
    main()
