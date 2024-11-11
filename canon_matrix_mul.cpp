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
    if (rank == 0)
    {
        ifstream fin("matrix_matrix_mul_input.txt");
        if (!fin)
        {
            cerr << "Error opening file." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Read matrices A and B from file
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                fin >> A[i][j];

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                fin >> B[i][j];

        fin.close();

        if (size != 4)
        {
            cout << "Error: Number of processors should be 4." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // Broadcast matrix B to all processes
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int block_size = n / sqrt(size);
    vector<double> local_A(block_size * n);    // Each process will receive block_size rows
    vector<double> local_C(block_size * n, 0); // Initialize local C
    // Scatter rows of A to all processes
    MPI_Scatter(A.data(), block_size * n, MPI_DOUBLE, local_A.data(), block_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Convert local_A to a 2D vector for easier multiplication
    vector<vector<double>> local_A_2D(block_size, vector<double>(n));
    for (int i = 0; i < block_size; ++i)
        for (int j = 0; j < n; ++j)
            local_A_2D[i][j] = local_A[i * n + j];
    MPI_Finalize();
    return 0;
}