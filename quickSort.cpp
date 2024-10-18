#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// Swap function
void swap(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}

// Partition function for quicksort
int partition(vector<int> &nums, int low, int high)
{
    int i = low, j = high;
    int pivot = low; // We are considering pivot to be the first element

    while (i < j) // Until they cross each other
    {
        // i will keep moving from the left until it finds element greater than pivot
        while (nums[i] <= nums[pivot] && i < high)
        {
            i++;
        }

        // j will keep moving from the right until it finds element smaller than pivot
        while (nums[j] > nums[pivot] && j > low)
        {
            j--;
        }

        // Swapping the smaller and greater elements
        if (i < j)
        {
            swap(nums[i], nums[j]);
        }
    }
    swap(nums[pivot], nums[j]); // Placing the pivot element at the correct position
    return j;
}

// Quicksort function
void quickSort(vector<int> &nums, int low, int high)
{
    if (low < high) // Array has more than two elements
    {
        int p = partition(nums, low, high);
        quickSort(nums, low, p - 1);
        quickSort(nums, p + 1, high);
    }
}

// Parallel Quicksort function using MPI
void parallelQuickSort(vector<int> &nums, int rank, int size, MPI_Comm comm)
{
    int n = nums.size();
    int k = n / size;
    int remainder = n % size;

    vector<int> local_nums(k + (rank < remainder ? 1 : 0)); // To distribute leftover elements

    vector<int> sendcounts(size), displs(size);
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = k + (i < remainder ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    MPI_Scatterv(nums.data(), sendcounts.data(), displs.data(), MPI_INT, local_nums.data(), local_nums.size(), MPI_INT, 0, comm);

    quickSort(local_nums, 0, local_nums.size() - 1);

    MPI_Gatherv(local_nums.data(), local_nums.size(), MPI_INT, nums.data(), sendcounts.data(), displs.data(), MPI_INT, 0, comm);

    if (rank == 0)
    {
        for (int i = 1; i < size; ++i)
        {
            inplace_merge(nums.begin(), nums.begin() + displs[i], nums.begin() + displs[i] + sendcounts[i]);
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<int> nums;
    int n;

    // Input from file (only on rank 0)
    if (rank == 0)
    {
        ifstream inputFile("input_quick_sort.txt");
        if (!inputFile)
        {
            cerr << "Error opening input file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inputFile >> n; // First line contains the number of elements
        nums.resize(n);
        for (int i = 0; i < n; i++)
        {
            inputFile >> nums[i];
        }

        inputFile.close();
    }

    // Broadcast the size of the array to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        nums.resize(n); // Resize the nums vector in other processes
    }

    // Perform parallel quicksort
    parallelQuickSort(nums, rank, size, MPI_COMM_WORLD);

    // Output to file (only on rank 0)
    if (rank == 0)
    {
        ofstream outputFile("output_quick_sort.txt");
        if (!outputFile)
        {
            cerr << "Error opening output file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outputFile << "Sorted Array: ";
        for (int i = 0; i < n; i++)
        {
            outputFile << nums[i] << " ";
        }
        outputFile << endl;

        outputFile.close();
    }

    MPI_Finalize();

    return 0;
}
