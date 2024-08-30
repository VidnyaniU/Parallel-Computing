#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

void matrix_vec_multiplication(int n, vector<double> &flattened_A, vector<double> &X, vector<double> &B, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int num_of_rows_per_proc = n / size;

    vector<double> local_matrix(num_of_rows_per_proc * n);
    MPI_Scatter(flattened_A.data(), num_of_rows_per_proc * n, MPI_DOUBLE, local_matrix.data(), num_of_rows_per_proc * n, MPI_DOUBLE, 0, comm);

    vector<double> local_vector(num_of_rows_per_proc);
    MPI_Scatter(X.data(), num_of_rows_per_proc, MPI_DOUBLE, local_vector.data(), num_of_rows_per_proc, MPI_DOUBLE, 0, comm);

    vector<double> gathered_vector(n);
    MPI_Allgather(local_vector.data(), num_of_rows_per_proc, MPI_DOUBLE, gathered_vector.data(), num_of_rows_per_proc, MPI_DOUBLE, comm);

    vector<double> local_result(num_of_rows_per_proc, 0.0);
    for (int i = 0; i < num_of_rows_per_proc; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local_result[i] += local_matrix[i * n + j] * gathered_vector[j];
        }
    }

    MPI_Gather(local_result.data(), num_of_rows_per_proc, MPI_DOUBLE, B.data(), num_of_rows_per_proc, MPI_DOUBLE, 0, comm);
    if (rank == 0)
    {
        cout << "OUTPUT :: ";
        for (int i = 0; i < n; i++)
        {

            cout << B[i] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // input
    int n = 4;
    vector<vector<double>> A = {
        {1, 2, 3, 4}, {7, 2, 3, 4}, {9, 1, 3, 5}, {1, 2, 9, 5}};
    vector<double> X = {1, 3, 4, 5};
    // flatten the matrix A

 //   vector<double> flattened_A(n * n);
 //   for (int i = 0; i < n; i++)
//    {
 //       for (int j = 0; j < n; j++)
   //     {
    //        flattened_A[i * n + j] = A[i][j];
  //      }
   // }
    vector<double> B(n);
    matrix_vec_multiplication(n, flattened_A, X, B, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
