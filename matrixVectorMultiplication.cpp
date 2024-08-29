#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 4;
    vector<int> matrix_A(N * N); // flattened matrix
    vector<int> vector_X(N);

    if (rank == 0)
    {
        // initialize the matrix
        int init_matrix[N][N] = {
            {1, 2, 3, 4}, {7, 2, 3, 4}, {9, 1, 3, 5}, {1, 2, 9, 5}};

        // flatten the matrix
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix_A[i * N + j] = init_matrix[i][j];
            }
        }
        // initialize the vector
        vector_X = {1, 3, 4, 5};

        // print if you want
    }

    // scatter matrix A to all the processors
    vector<int> local_matrix(N);
    MPI_Scatter(&matrix_A, N, MPI_INT, &local_matrix, N, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter vector X to all the processors
    vector<int> local_vector(N / size);
    MPI_Scatter(&vector_X, N / size, MPI_INT, &local_vector, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather the full vector on all the processors
    vector<int> full_vector(N);
    MPI_Allgather(&local_vector, N / size, MPI_INT, &full_vector, N / size, MPI_INT, MPI_COMM_WORLD);

    // computation on each processor
    vector<int> local_result_vec(N);
    int local_result = 0;
    // local_result[rank] = 0;
    for (int i = 0; i < N; i++)
    {
        local_result += local_matrix[i] * full_vector[i];
    }
    local_result_vec[rank] = local_result;
    // Gather results at the rank
    vector<int> result(N);
    MPI_Gather(&local_result, 1, MPI_INT, &result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // print the result
    if (rank == 0)
    {
        cout << "Output :: " << endl;
        for (auto ele : result)
        {
            cout << ele << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}