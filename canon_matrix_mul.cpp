#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

vector<double> flattenMatrix(vector<vector<double>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<double> flatMatrix(rows * cols); // 1D vector to store the flattened matrix

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            flatMatrix[i * cols + j] = matrix[i][j]; // Flatten row-wise
        }
    }

    return flatMatrix;
}

void multiplyMatrices(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = 0; // Initialize C[i][j] to 0
            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
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

    int n = 8; // size of matrix

    vector<vector<double>> A(n, vector<double>(n)), B(n, vector<double>(n));
    vector<vector<double>> final_C(n, vector<double>(n, 0)); // Initialize final result matrix
sssss
    MPI_Finalize();
    return 0;
}