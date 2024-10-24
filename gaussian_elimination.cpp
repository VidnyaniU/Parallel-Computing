
/*#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

vector<double> gaussian_elimination(int rows, int cols, vector<vector<double>> mat, int rank, int size)
{
    // 1D row-wise partitioning
    vector<double> local_mat(cols);
    MPI_Scatter(mat.data(), cols, MPI_DOUBLE, local_mat.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gaussian elimination
    for (int r = 0; r < rows; r++)
    {
        // Normalizing the pivot row
        double pivot = mat[r][r];
        if (pivot != 0)
        {
            for (int i = 0; i < cols; i++)
            {
                mat[r][i] /= pivot;
            }

            // Eliminating below the pivot
            for (int i = r + 1; i < rows; i++)
            {
                double temp = mat[i][r];
                for (int j = r; j < cols; j++)
                {
                    mat[i][j] -= temp * mat[r][j];
                }
            }
        }
    }

    MPI_Gather(local_mat.data(), cols, MPI_DOUBLE, mat.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Back substitution
    vector<double> ans(rows);
    ans[rows - 1] = mat[rows - 1][cols - 1] / mat[rows - 1][rows - 1];

    for (int r = rows - 2; r >= 0; r--)
    {
        double lhs = 0;
        for (int c = r + 1; c < rows; c++)
        {
            lhs += mat[r][c] * ans[c];
        }
        ans[r] = (mat[r][cols - 1] - lhs) / mat[r][r];
    }

    return ans;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "size :: " << size << endl;
    // int rows = 4, cols = 5;
    int rows, cols;
    vector<vector<double>> mat;
    // mat = {{1, 2, 1, 4, 13}, {2, 0, 4, 3, 28}, {4, 2, 2, 1, 20}, {-3, 1, 3, 2, 6}};

    if (rank == 0)
    {
        ifstream input_file("gaussian_elimination_input.txt");
        if (input_file.is_open())
        {
            input_file >> rows;
            input_file >> cols;

            mat.resize(rows, vector<double>(cols));

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {

                    input_file >> mat[i][j];
                }
            }

            input_file.close();
        }
        else
        {
            cerr << "Unable to open input file!" << endl;
            // MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Output
    if (rank == 0)
    {
        vector<double> ans = gaussian_elimination(rows, cols, mat, rank, size);
        string filename = "gaussian_elimination_output" + to_string(size) + ".txt";
        ofstream output_file(filename);
        if (output_file.is_open())
        {
            output_file << "Final solution :: ";
            for (int i = 0; i < rows; i++)
            {
                output_file << " " << ans[i] << " ";
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
    // if (rank == 0)
    // {
    //     vector<double> ans = gaussian_elimination(rows, cols, mat, rank, size);
    //     cout << "Answer :: ";
    //     for (int i = 0; i < rows; i++)
    //     {
    //         cout << " " << ans[i] << " ";
    //     }
    //     cout << endl;
    // }
    else
    {
        gaussian_elimination(rows, cols, mat, rank, size);
    }

    MPI_Finalize();
    return 0;
}

*/
// MPI parallel gaussian elimination
// By: Nick from CoffeeBeforeArch

#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

void gaussian_elimination(int rows, int cols, float *mat, int rank, int size)
{
    int n_rows = rows / size; // Assuming rows divide evenly among processes
    unique_ptr<float[]> local_mat(new float[n_rows * cols]);
    unique_ptr<float[]> pivot_row(new float[cols]);

    // Scatter the matrix to all processes
    MPI_Scatter(mat, n_rows * cols, MPI_FLOAT, local_mat.get(), n_rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int r = 0; r < rows; r++)
    {
        int mapped_rank = r / n_rows;

        if (rank == mapped_rank)
        {
            int local_row = r % n_rows;
            float pivot = local_mat[local_row * cols + r];

            // Normalize the pivot row
            for (int j = r; j < cols; j++)
            {
                local_mat[local_row * cols + j] /= pivot;
            }

            // Send pivot row to other processes
            for (int i = mapped_rank + 1; i < size; i++)
            {
                MPI_Request request; // Declare a request variable
                MPI_Isend(local_mat.get() + local_row * cols, cols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
            }

            // Eliminate rows in the local matrix
            for (int i = local_row + 1; i < n_rows; i++)
            {
                float scale = local_mat[i * cols + r];
                for (int j = r; j < cols; j++)
                {
                    local_mat[i * cols + j] -= scale * local_mat[local_row * cols + j];
                }
            }
        }
        else
        {
            // Receive pivot row
            MPI_Recv(pivot_row.get(), cols, MPI_FLOAT, mapped_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Eliminate rows in the local matrix
            for (int i = 0; i < n_rows; i++)
            {
                float scale = local_mat[i * cols + r];
                for (int j = r; j < cols; j++)
                {
                    local_mat[i * cols + j] -= pivot_row[j] * scale;
                }
            }
        }
    }

    // Gather the final results into rank 0
    MPI_Gather(local_mat.get(), n_rows * cols, MPI_FLOAT, mat, n_rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;
    float *mat = nullptr;

    if (rank == 0)
    {
        // Reading the matrix from input file
        ifstream input_file("gaussian_elimination_input.txt");
        if (input_file.is_open())
        {
            input_file >> rows >> cols;
            mat = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    input_file >> mat[i * cols + j];
                }
            }
            input_file.close();
        }
        else
        {
            cerr << "Unable to open input file!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast rows and columns to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        mat = new float[rows * cols]; // Allocate memory for non-root ranks
    }

    // Call the Gaussian elimination function
    gaussian_elimination(rows, cols, mat, rank, size);

    // Only the root process will write the output
    if (rank == 0)
    {
        ofstream output_file("gaussian_elimination_output.txt");
        if (output_file.is_open())
        {
            output_file << "Final matrix:\n";
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output_file << mat[i * cols + j] << " ";
                }
                output_file << endl;
            }
            output_file.close();
            cout << "Output sent to the file!" << endl;
        }
        else
        {
            cerr << "Unable to open output file!" << endl;
        }
        delete[] mat; // Free memory allocated for the matrix
    }
    else
    {
        delete[] mat; // Free memory allocated for non-root ranks
    }

    MPI_Finalize();
    return 0;
}
