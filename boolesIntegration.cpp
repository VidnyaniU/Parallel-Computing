#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

// Function to be integrated
double f_x(double x)
{
    return sin(x);
}

// Function to partition the domain
double *partition(int a, int b, int n)
{
    double h = (double)(b - a) / n;
    double *A = new double[n + 1];
    A[0] = a;
    for (int i = 1; i <= n; i++)
    {
        A[i] = A[i - 1] + h;
    }
    return A;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // domain of the function
    int a = 0, b = 1;
    int n = 1000; // Number of partitions
    double h = (double)(b - a) / n;

    // Number of points each processor handles
    int points_per_proc = n / size;
    int remainder = n % size;

    // local start and end indices for each processor
    int local_start = rank * points_per_proc + min(rank, remainder);
    int local_end = local_start + points_per_proc + (rank < remainder ? 1 : 0);

    double local_ans = 0.0;

    double *A = nullptr;

    if (rank == 0)
    {
        A = partition(a, b, n);

        // Distributing the data to each processor
        for (int i = 1; i < size; i++)
        {
            int start_index = i * points_per_proc + min(i, remainder);
            int end_index = start_index + points_per_proc + (i < remainder ? 1 : 0);
            int data_size = end_index - start_index + 1;
            MPI_Send(&A[start_index], data_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        // booles integration for points on rank 0

        // storing the values of the f_x on each point
        vector<double> f_xs(local_end - local_start + 1);
        for (int i = 0; i < f_xs.size(); i++)
        {
            f_xs[i] = f_x(A[i]);
        }

        for (int i = 0; i <= local_end - local_start - 4; i += 4)
        {
            local_ans += (7 * f_xs[i] + 32 * f_xs[i + 1] + 12 * f_xs[i + 2] + 32 * f_xs[i + 3] + 7 * f_xs[i + 4]);
        }

        double global_ans = ((2 * h) / 45) * local_ans;

        // receiving local_ans from each processor
        for (int i = 1; i < size; i++)
        {
            double temp_ans;
            MPI_Recv(&temp_ans, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_ans += ((2 * h) / 45) * temp_ans;
        }

        // printing the result
        // cout << "Final result of Booles Integration ::  " << global_ans << endl;

        ofstream outputFile("output_booles_integration.txt");
        if (outputFile.is_open())
        {
            outputFile << "Final result of Booles Integration :: " << global_ans << endl;
            outputFile.close();
        }
        else
        {
            cerr << "Unable to open file output_booles_integration.txt" << endl;
        }

        delete[] A;
    }
    else
    {
        int data_size = local_end - local_start + 1;
        A = new double[data_size];
        MPI_Recv(A, data_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // booles integration for points on other processors

        // storing the values of the f_x on each point
        vector<double> f_xs(local_end - local_start + 1);
        for (int i = 0; i < f_xs.size(); i++)
        {
            f_xs[i] = f_x(A[i]);
        }

        for (int i = 0; i <= local_end - local_start - 4; i += 4)
        {
            local_ans += (7 * f_xs[i] + 32 * f_xs[i + 1] + 12 * f_xs[i + 2] + 32 * f_xs[i + 3] + 7 * f_xs[i + 4]);
        }

        MPI_Send(&local_ans, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        delete[] A;
    }

    MPI_Finalize();
    return 0;
}
