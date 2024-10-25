#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

#define N 4 // Size of the matrix

void gaussian_elimination(double A[N][N], double b[N], int rank, int size)
{
    for (int k = 0; k < N; ++k)
    {
        // Broadcasting the pivot row
        if (rank == k % size)
        {
            for (int j = k; j < N; ++j)
            {
                MPI_Bcast(&A[k][j], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
            }
            MPI_Bcast(&b[k], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = k; j < N; ++j)
            {
                MPI_Bcast(&A[k][j], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
            }
            MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        }

        for (int i = k + 1; i < N; ++i)
        {
            if (i % size == rank)
            {
                double factor = A[i][k] / A[k][k];
                for (int j = k; j < N; ++j)
                {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }
}

void back_substitution(double A[N][N], double b[N], double x[N], int rank, int size)
{
    for (int i = N - 1; i >= 0; --i)
    {
        if (rank == i % size)
        {
            x[i] = b[i] / A[i][i];
        }
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);

        for (int j = i - 1; j >= 0; --j)
        {
            if (rank == j % size)
            {
                b[j] -= A[j][i] * x[i];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A[N][N];
    double b[N];
    double x[N] = {0};

    // Read the matrix and vector
    if (rank == 0)
    {
        ifstream fin("gaussian_elimination_input.txt");
        if (!fin)
        {
            cerr << "Error opening file!" << endl;
            exit(EXIT_FAILURE);
        }

        // matrix A
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                fin >> A[i][j];
            }
        }

        // vector b
        for (int i = 0; i < N; ++i)
        {
            fin >> b[i];
        }

        fin.close();
    }

    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gaussian elimination and back substitution
    gaussian_elimination(A, b, rank, size);
    back_substitution(A, b, x, rank, size);

    // solution vector x
    if (rank == 0)
    {
        cout << "Solution vector x using " << size << " processors : ";
        for (int i = 0; i < N; ++i)
        {
            cout << x[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
