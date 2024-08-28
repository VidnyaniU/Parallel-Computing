#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 4;
    vector<int> matrix_A(N * N); // flattened matrix
    vector<int> vector_X(N);

    if (rank == 0)
    {
        // initialize the matrix

        // flatten the matrix

        // initialize the vector

        // print if you want
    }

    // scatter matrix A to all the processors
    MPI_Scatter();

    // scatter vector X to all the processors
    MPI_Scatter();

    // Gather the full vector on all the processors
    MPI_Allgather();

    // computation on each processor

    // Gather results at the rank 0
    MPI_Gather();

    // print the result
    if (rank == 0)
    {
    }

    MPI_Finalize();
    return 0;
}