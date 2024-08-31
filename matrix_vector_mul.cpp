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
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int n;
    vector<vector<double>> A;
    vector<double> X;

    // Read input from file
    ifstream infile("matrix_vec_input.txt");
    if (infile.is_open())
    {
        infile >> n;
        A.resize(n, vector<double>(n));
        X.resize(n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                infile >> A[i][j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            infile >> X[i];
        }

        infile.close();
    }
    else
    {
        cerr << "Unable to open input file" << endl;
        
    }

    
    vector<double> flattened_A(n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            flattened_A[i * n + j] = A[i][j];
        }
    }

    vector<double> B(n);
    matrix_vec_multiplication(n, flattened_A, X, B, MPI_COMM_WORLD);

    
    if (rank == 0)
    {
        ofstream outfile("matrix_vec_output.txt");
        if (outfile.is_open())
        {
            for (int i = 0; i < n; i++)
            {
                outfile << B[i] << " ";
            }
            outfile << endl;
            outfile.close();
        }
        else
        {
            cerr << "Unable to open output file" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
