

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

void addMatrices(vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
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

        // cout << "Matrix A :: " << endl;

        // for (int i = 0; i < n; ++i)
        // {
        //     for (int j = 0; j < n; ++j)
        //     {
        //         cout << A[i][j] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << "Matrix B :: " << endl;
        // for (int i = 0; i < n; ++i)
        // {
        //     for (int j = 0; j < n; ++j)
        //     {
        //         cout << B[i][j] << " ";
        //     }
        //     cout << endl;
        // }
    }
    vector<double> flattened_A = flattenMatrix(A);
    vector<double> flattened_B = flattenMatrix(B);

    vector<vector<double>> local_A(block_size, vector<double>(block_size));
    vector<vector<double>> local_B(block_size, vector<double>(block_size));
    vector<vector<double>> local_C(block_size, vector<double>(block_size));

    int send_count[size], displacement_arr[size];
    displacement_arr[0] = 0;
    for (int i = 0; i < size; i++)
        send_count[i] = size;

    // displacements for Scatterv
    for (int i = 0; i < size; i++)
    {
        int row = i / (int)sqrt(size);
        int col = i % (int)sqrt(size);
        displacement_arr[i] = row * block_size * n + col * block_size;
    }

    for (int j = 0; j < size; j++)
    {
        MPI_Scatterv(flattened_A.data(), send_count, displacement_arr, MPI_DOUBLE, local_A[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Scatterv(flattened_B.data(), send_count, displacement_arr, MPI_DOUBLE, local_B[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    // displacements for Scatterv
    for (int i = 0; i < size; i++)
    {
        int row = i / (int)sqrt(size);
        int col = i % (int)sqrt(size);
        displacement_arr_B[i] = col * block_size * n + row * block_size;
    }

    for (int j = 0; j < size; j++)
    {
        // MPI_Scatterv(flattened_A.data(), send_count, displacement_arr, MPI_DOUBLE, local_A[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(flattened_B.data(), send_count_B, displacement_arr_B, MPI_DOUBLE, local_B[j].data(), block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++)
        {
            displacement_arr_B[i] += n;
        }
    }

    multiplyMatrices(local_A, local_B, local_C, block_size);

    // vector<double> flattened_local_A = flattenMatrix(local_A);
    // vector<double> flattened_local_B = flattenMatrix(local_B);
    if (rank == 2)
    {
        cout << "Matrix A :: " << endl;
        cout << "\nProcess " << rank << " received: " << endl;
        for (int i = 0; i < block_size; ++i)
        {
            for (int j = 0; j < block_size; ++j)
            {
                cout << local_A[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        cout << "Matrix B :: " << endl;
        cout << "\nProcess " << rank << " received: " << endl;
        for (int i = 0; i < block_size; ++i)
        {
            for (int j = 0; j < block_size; ++j)
            {
                cout << local_B[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;

        cout << "Matrix C :: " << endl;
        cout << "\nProcess " << rank << " received: " << endl;
        for (int i = 0; i < block_size; ++i)
        {
            for (int j = 0; j < block_size; ++j)
            {
                cout << local_C[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    /*
    // Assuming n is the dimension of the matrices and size is the number of processes
    vector<double> gathered_A_G(n * n); // Allocate space for the final gathered result
    vector<double> gathered_B_G(n * n); // Allocate space for the final gathered result

    // Prepare the send_count and displacement arrays
    int send_count_g[size];
    int displacement_arr_g[size];

    for (int i = 0; i < size; i++)
    {
        send_count_g[i] = block_size * block_size; // Each process sends block_size * block_size elements
    }

    // Set the displacements for gathering
    for (int i = 0; i < size; i++)
    {
        int row = i / (int)sqrt(size);
        int col = i % (int)sqrt(size);
        displacement_arr_g[i] = (row * block_size * n) + (col * block_size);
    }

    // Gather the resulting matrix C from all processes
    MPI_Gatherv(local_A[0].data(), block_size * block_size, MPI_DOUBLE,
                gathered_A_G.data(), send_count_g, displacement_arr_g, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_B[0].data(), block_size * block_size, MPI_DOUBLE,
                gathered_B_G.data(), send_count_g, displacement_arr_g, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        cout << "Gathered_A_G :: " << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                cout << gathered_A_G[i * n + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    // Add matrices A and B on the root process
    // if (rank == 0)
    // {
    //     vector<vector<double>> final_C(n, vector<double>(n, 0.0));

    //     // addMatrices(gathered_A_G, gathered_B_G, final_C, n);

    //     cout << "Final Matrix C (A + B):" << endl;
    //     for (int i = 0; i < n; ++i)
    //     {
    //         for (int j = 0; j < n; ++j)
    //         {
    //             cout << final_C[i][j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }
    */
    MPI_Finalize();
    return 0;
}