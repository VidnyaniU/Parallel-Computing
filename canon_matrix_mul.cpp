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