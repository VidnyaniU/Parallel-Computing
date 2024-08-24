#include <bits/stdc++.h>
#include <mpi.h>

// sum of all the numbers
using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    int sum = 0, global_sum = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream inputFile("input_parallel_add.txt");
    ofstream outputFile("output_parallel_add.txt");

    if (!inputFile.is_open())
    {
        cerr << "Error opening input file!" << endl;
        return 1;
    }

    if (!outputFile.is_open())
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    vector<int> arr;
    int number;

    while (inputFile >> number)
    {
        arr.push_back(number);
    }

    // vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int n = arr.size() / size;
    int B[n];
    int offset = 0;

    if (rank == 0)
    {
        // sending n elements to each processor
        for (int i = 1; i < size; i++)
        {
            offset = n * i;
            MPI_Send(&arr[offset], n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // calculating sum of first n elements on rank 0
        for (int i = 0; i < n; i++)
        {
            sum += arr[i];
        }

        global_sum = sum;
        // adding all the local_sum received from each processor to the global_sum
        for (int i = 1; i < size; i++)
        {
            int temp = 0;
            MPI_Recv(&temp, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum += temp;
        }

        // calculating sum of the remaining elements in the array
        int remaining_elements = arr.size() % size;
        for (int i = arr.size() - remaining_elements; i < arr.size(); i++)
        {
            global_sum += arr[i];
        }

        // cout << "Sum of all the numbers :: " << global_sum << endl;
        outputFile << "Sum of all the elements: " << global_sum << endl;
        inputFile.close();
        outputFile.close();
        cout << "Sum has been written to output_parallel_add.txt successfully!" << endl;
    }

    // calculating local_sum on each processor other than 0
    else
    {
        int B[n];
        int local_sum = 0;

        // receiving n elements in B
        MPI_Recv(&B, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < n; i++)
        {
            local_sum += B[i];
        }
        MPI_Send(&local_sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    // cout<<"Sum from process " <<rank << " is "<<sum<<endl;

    MPI_Finalize();

    return 0;
}
