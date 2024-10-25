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

    ifstream fin;
    fin.open("matrix_matrix_mul_input.txt");

    vector<vector<double>> A(n, vector<double>(n)), B(n, vector<double>(n));

    int block_size = int(n / sqrt(size));

    if (rank == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                fin >> A[i][j];
            }
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                fin >> B[i][j];
            }
        }
        fin.close();

        if (size != 4)
        {
            cout << "Error: Number of processors should be 4." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    vector<double> flattened_A = flattenMatrix(A);
    vector<double> flattened_B = flattenMatrix(B);

    //================================C00 & C11===================================================================================================================================
    vector<vector<double>> local_A(block_size, vector<double>(block_size));
    vector<vector<double>> local_B(block_size, vector<double>(block_size));
    vector<vector<double>> Aij(block_size, vector<double>(block_size));

    int send_count[size], displacement_arr[size];
    displacement_arr[0] = 0;
    for (int i = 0; i < size; i++)
        send_count[i] = size;

    // displacement array for Scatterv A
    for (int i = 0; i < size; i++)
    {
        int row = i / (int)sqrt(size);
        int col = i % (int)sqrt(size);
        displacement_arr[i] = row * block_size * n + col * block_size;
    }

    for (int j = 0; j < size; j++)
    {
        MPI_Scatterv(flattened_A.data(), send_count, displacement_arr, MPI_DOUBLE, local_A[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++)
        {
            displacement_arr[i] += n;
        }
    }

    // for matrix B

    int send_count_B[size], displacement_arr_B[size];
    displacement_arr_B[0] = 0;
    for (int i = 0; i < size; i++)
        send_count_B[i] = size;

    // displacement array for Scatterv B
    for (int i = 0; i < size; i++)
    {
        int row = i / (int)sqrt(size);
        int col = i % (int)sqrt(size);
        displacement_arr_B[i] = col * block_size * n + row * block_size;
    }

    for (int j = 0; j < size; j++)
    {
        MPI_Scatterv(flattened_B.data(), send_count_B, displacement_arr_B, MPI_DOUBLE, local_B[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++)
        {
            displacement_arr_B[i] += n;
        }
    }

    multiplyMatrices(local_A, local_B, Aij, block_size);

    vector<double> C(block_size * block_size);

    vector<double> flattened_Aij = flattenMatrix(Aij);

    if (rank == 1)
    {
        MPI_Send(flattened_Aij.data(), block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    vector<double> C11(block_size * block_size);
    vector<double> C00(block_size * block_size);
    if (rank == 0)
    {
        MPI_Recv(C.data(), block_size * block_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "C00 :: " << endl;
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                C00[i * block_size + j] = flattened_Aij[i * block_size + j] + C[i * block_size + j];
                cout << C00[i * block_size + j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        // receive C11

        MPI_Recv(C11.data(), block_size * block_size, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "C11 :: " << endl;
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                cout << C11[i * block_size + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    if (rank == 3)
    {
        MPI_Send(flattened_Aij.data(), block_size * block_size, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD);
    }
    if (rank == 2)
    {
        MPI_Recv(C.data(), block_size * block_size, MPI_DOUBLE, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                C11[i * block_size + j] = flattened_Aij[i * block_size + j] + C[i * block_size + j];
            }
        }
        MPI_Send(C11.data(), block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    //================================================================================================================================

    //==============================C01 & C10==================================================================================================

    // scattering matrix B differently than previous one to get C01 and C10
    //  for matrix B
    vector<vector<double>> local_B2(block_size, vector<double>(block_size));

    int send_count_B2[size], displacement_arr_B2[size] = {4, 36, 0, 32};
    for (int i = 0; i < size; i++)
        send_count_B2[i] = block_size;

    for (int j = 0; j < size; j++)
    {
        MPI_Scatterv(flattened_B.data(), send_count_B2, displacement_arr_B2, MPI_DOUBLE, local_B2[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++)
        {
            displacement_arr_B2[i] += n;
        }
    }
    vector<vector<double>> Aij2(block_size, vector<double>(block_size));

    multiplyMatrices(local_A, local_B2, Aij2, block_size);

    vector<double> C2(block_size * block_size);

    vector<double> flattened_Aij2 = flattenMatrix(Aij2);

    if (rank == 1)
    {
        MPI_Send(flattened_Aij2.data(), block_size * block_size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
    vector<double> C10(block_size * block_size);
    vector<double> C01(block_size * block_size);
    if (rank == 0)
    {
        MPI_Recv(C2.data(), block_size * block_size, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "C01 :: " << endl;

        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                C01[i * block_size + j] = flattened_Aij2[i * block_size + j] + C2[i * block_size + j];
                cout << C01[i * block_size + j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        // receive C10

        MPI_Recv(C10.data(), block_size * block_size, MPI_DOUBLE, 2, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "C10:: " << endl;
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                cout << C10[i * block_size + j] << " ";
                /* code */
            }
            cout << endl;
        }
        cout << endl;
    }
    if (rank == 3)
    {
        MPI_Send(flattened_Aij2.data(), block_size * block_size, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD);
    }
    if (rank == 2)
    {
        MPI_Recv(C2.data(), block_size * block_size, MPI_DOUBLE, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                C10[i * block_size + j] = flattened_Aij2[i * block_size + j] + C2[i * block_size + j];
            }
        }
        MPI_Send(C10.data(), block_size * block_size, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
    }
    //==============================================================================================================================

    // final matrix C
    if (rank == 0)
    {
        vector<vector<double>> final_C(n, vector<double>(n));
        // C00
        for (int j = 0; j < block_size; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                final_C[j][k] = C00[j * block_size + k];
            }
        }

        // C01
        for (int j = 0; j < block_size; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                final_C[j][k + block_size] = C01[j * block_size + k];
            }
        }

        // C10
        for (int j = 0; j < block_size; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                final_C[j + block_size][k] = C10[j * block_size + k];
            }
        }

        // C11
        for (int j = 0; j < block_size; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                final_C[j + block_size][k + block_size] = C11[j * block_size + k];
            }
        }

        cout << "Final matrix C :: " << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                cout << final_C[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}