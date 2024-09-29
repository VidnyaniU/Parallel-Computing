#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

void matrixMultiply(vector<double> A_sub, vector<double> B_sub, vector<double> C_sub, int sub_size)
{
    for (int i = 0; i < sub_size; i++)
    {
        for (int j = 0; j < sub_size; j++)
        {
            C_sub[i * sub_size + j] = 0;
            for (int k = 0; k < sub_size; k++)
            {
                C_sub[i * sub_size + j] += A_sub[i * sub_size + k] * B_sub[k * sub_size + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4;

    vector<double> A(n * n), B(n * n);
    A = {3, 4, 5, 6, 7, 8, 9, 1, 3, 5, 6, 7, 2, 6, 4, 2};
    B = {8, 3, 1, 9, 6, 7, 2, 6, 4, 2, 7, 8, 9, 1, 3, 5};

    int sub_size = n / sqrt(size);
    // cout << "sub_size :: " << sub_size << endl;

    vector<double> A_sub(sub_size * sub_size), B_sub(sub_size * sub_size);

    MPI_Scatter(A.data(), sub_size * sub_size, MPI_DOUBLE, A_sub.data(), sub_size * sub_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), sub_size * sub_size, MPI_DOUBLE, B_sub.data(), sub_size * sub_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "From processor :: " << rank << "\nSubmatrix A :: " << endl;
        for (int i = 0; i < sub_size * sub_size; i++)
        {
            // for (int j = 0; j < sub_size; j++)
            // {
            cout << A_sub[i] << " ";
            // }
        }
        cout << endl;

        cout << "From processor :: " << rank << "\nSubmatrix B :: " << endl;
        for (int i = 0; i < sub_size * sub_size; i++)
        {
            // for (int j = 0; j < ; j++)
            // {
            cout << B_sub[i] << " ";
            // }
        }
        cout << endl;
    }
    MPI_Finalize();

    return 0;
}