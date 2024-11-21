#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int dim = 2;
    int dimv[2] = {2, 2};
    int period[2] = {1, 1};
    int reorder = 1;
    MPI_Comm Cart_Topology;

    MPI_Cart_create(MPI_COMM_WORLD, dim, dimv, period, reorder, &Cart_Topology);

    int rank;
    MPI_Comm_rank(Cart_Topology, &rank);

    int A[4][4], B[4][4], C[4][4] = {0};
    int A_local[2][2], B_local[2][2], C_local[2][2] = {0};

    if (rank == 0)
    {
        // Input matrices from file
        FILE *file = fopen("canon_matrix_mul_input.txt", "r");
        if (file == NULL)
        {
            printf("Error opening input file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read matrix A
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                fscanf(file, "%d", &A[i][j]);
            }
        }

        // Read matrix B
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                fscanf(file, "%d", &B[i][j]);
            }
        }

        fclose(file);
    }

    // Scatter matrices A and B to processes
    int sendcounts1[4] = {2, 2, 2, 2};
    int displs1[4] = {0, 2, 8, 10};
    MPI_Scatterv(&A, sendcounts1, displs1, MPI_INT, &A_local[0], 2, MPI_INT, 0, Cart_Topology);
    MPI_Scatterv(&B, sendcounts1, displs1, MPI_INT, &B_local[0], 2, MPI_INT, 0, Cart_Topology);

    int sendcounts2[4] = {2, 2, 2, 2};
    int displs2[4] = {4, 6, 12, 14};
    MPI_Scatterv(&A, sendcounts2, displs2, MPI_INT, &A_local[1], 2, MPI_INT, 0, Cart_Topology);
    MPI_Scatterv(&B, sendcounts2, displs2, MPI_INT, &B_local[1], 2, MPI_INT, 0, Cart_Topology);

    // Perform matrix multiplication
    for (int step = 0; step < 2; step++)
    {
        if (step == 0)
        {
            if (rank == 2 || rank == 3)
            {
                MPI_Request send_request, recv_request;
                MPI_Status status;
                int left, right;
                MPI_Cart_shift(Cart_Topology, 1, -1, &left, &right);
                MPI_Isend(&A_local, 4, MPI_INT, left, 0, Cart_Topology, &send_request);
                MPI_Irecv(&A_local, 4, MPI_INT, right, 0, Cart_Topology, &recv_request);
                MPI_Wait(&send_request, &status);
                MPI_Wait(&recv_request, &status);
            }

            if (rank == 1 || rank == 3)
            {
                MPI_Request send_request1, recv_request1;
                MPI_Status status1;
                int up, down;
                MPI_Cart_shift(Cart_Topology, 0, 1, &up, &down);
                MPI_Isend(&B_local, 4, MPI_INT, up, 1, Cart_Topology, &send_request1);
                MPI_Irecv(&B_local, 4, MPI_INT, down, 1, Cart_Topology, &recv_request1);
                MPI_Wait(&send_request1, &status1);
                MPI_Wait(&recv_request1, &status1);
            }
        }

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    C_local[i][j] += A_local[i][k] * B_local[k][j];
                }
            }
        }
    }

    // Gather results into matrix C
    MPI_Gatherv(&C_local[0], 2, MPI_INT, &C, sendcounts1, displs1, MPI_INT, 0, Cart_Topology);
    MPI_Gatherv(&C_local[1], 2, MPI_INT, &C, sendcounts2, displs2, MPI_INT, 0, Cart_Topology);

    // Print result on rank 0
    if (rank == 0)
    {
        printf("Resultant Matrix C:\n");
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
