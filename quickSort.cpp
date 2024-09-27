#include <bits/stdc++.h>
#include <mpi.h>

// sum of all the numbers
using namespace std;
// swap
void swap(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}

int partition(vector<int> &nums, int low, int high)
{

    int i = low, j = high;
    int pivot = low; // we are considering pivot to be the first element

    while (i < j) // until they cross each other
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

        // swapping the smaller and greater elements
        if (i < j)
        {
            swap(nums[i], nums[j]);
        }
    }
    swap(nums[pivot], nums[j]); // placing the pivot element at the correct position
    return j;
}
void quickSort(vector<int> &nums, int low, int high)
{
    if (low < high) // array has more than two elements
    {

        int p = partition(nums, low, high);
        quickSort(nums, low, p - 1);
        quickSort(nums, p + 1, high);
    }
}

void parallelQuickSort(vector<int> &nums, int rank, int size, MPI_Comm comm)
{
    // int rank, size;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = nums.size();
    int k = n / size;
    int remainder = n % size;

    vector<int> local_nums(k + (rank < remainder ? 1 : 0)); // to distribute leftover elements

    vector<int> sendcounts(size), displs(size);
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = k + (i < remainder ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];

        /*Suppose n = 10 and size = 3 (three processes), so:

k = 10 / 3 = 3 (each process gets at least 3 elements).
remainder = 10 % 3 = 1 (1 leftover element).
For each process:

Process 0 (rank = 0):
Gets k + 1 = 3 + 1 = 4 elements (because rank < remainder).
Starts at displacement 0, so displs[0] = 0.
Process 1 (rank = 1):
Gets k = 3 elements (since rank >= remainder).
Starts at displacement 4, so displs[1] = 4.
Process 2 (rank = 2):
Gets k = 3 elements.
Starts at displacement 7, so displs[2] = 7.


sendcounts = {4, 3, 3}: Process 0 gets 4 elements, and processes 1 and 2 get 3 elements each.
displs = {0, 4, 7}: Process 0’s chunk starts at index 0, process 1’s chunk starts at index 4, and process 2’s chunk starts at index 7 in the nums array.*/
    }

    MPI_Scatterv(nums.data(), sendcounts.data(), displs.data(), MPI_INT, local_nums.data(), local_nums.size(), MPI_INT, 0, comm);

    quickSort(local_nums, 0, local_nums.size() - 1);

    MPI_Gatherv(local_nums.data(), local_nums.size(), MPI_INT, nums.data(), sendcounts.data(), displs.data(), MPI_INT, 0, comm);

    if (rank == 0)
    {
        vector<int> temp(nums); // temporary array for merging
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

    vector<int> nums = {10, 7, 8, 9, 1, 6, 5};
    int n = nums.size();

    parallelQuickSort(nums, rank, size, MPI_COMM_WORLD);

    // Gather results and print the sorted array
    if (rank == 0)
    {
        cout << "Sorted Array :: ";
        for (int i = 0; i < n; i++)
        {
            cout << nums[i] << " ";
        }
        cout << endl;
    }
    MPI_Finalize();

    return 0;
}