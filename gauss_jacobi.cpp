#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// Function to calculate the Euclidean norm of a vector
double norm(vector<double> vec)
{
    double temp = 0.0;
    for (int i = 0; i < vec.size(); i++)
        temp += vec[i] * vec[i];
    return sqrt(temp);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Defining necessary variables

    vector<double> A;
    vector<double> b;
    int n;
    // Read the matrix A and vector b from a file (done by rank 0)
    if (rank == 0)
    {
        ifstream input_file("input_gauss_jacobi.txt");
        if (input_file.is_open())
        {
            input_file >> n; // Read the dimension of the matrix/vector
            A.resize(n * n);
            b.resize(n);

            for (int i = 0; i < n * n; i++)
            {
                input_file >> A[i];
            }

            for (int i = 0; i < n; i++)
            {
                input_file >> b[i];
            }

            input_file.close();
        }
        else
        {
            cerr << "Unable to open input file!" << endl;
            // MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // // Broadcast the size of the matrix to all processes
    // MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int k = n / size;

    vector<double> A_local(k * n);
    vector<double> b_local(k);

    // Scatter the matrix A and vector b
    MPI_Scatter(A.data(), k * n, MPI_DOUBLE, A_local.data(), k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b.data(), k, MPI_DOUBLE, b_local.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> X_old(n, 0); // Initial guess for X
    MPI_Bcast(X_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double tolerence = 0.000000001;
    int iterations = 1000;
    vector<double> X_new_local(k);

    for (int it = 0; it < iterations; it++)
    {
        for (int i = 0; i < k; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (rank * k + i != j)
                {
                    sum += A_local[i * n + j] * X_old[j];
                }
            }
            X_new_local[i] = (b_local[i] - sum) / A_local[i * n + (rank * k + i)];
        }

        vector<double> X_new(n);
        MPI_Allgather(X_new_local.data(), k, MPI_DOUBLE, X_new.data(), k, MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == 0)
        {
            if (fabs(norm(X_new) - norm(X_old)) < tolerence)
            {
                break;
            }

            X_old = X_new;
        }

        MPI_Bcast(X_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Output
    if (rank == 0)
    {
        ofstream output_file("output_gauss_jacobi.txt");
        if (output_file.is_open())
        {
            output_file << "Final solution :: ";
            for (int i = 0; i < n; i++)
            {
                output_file << X_old[i] << " ";
            }
            output_file << endl;
            output_file.close();
            cout << "Output sent to the file!" << endl;
        }
        else
        {
            cerr << "Unable to open output file!" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
