#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);
    if (argc < 2)
    {
        printf("please give the numebr of iterations you want to test!\n");
        abort();
    }
    int N = atoi(argv[1]);
    int p;//number of nodes/processes we have.
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    int* intearr = new int[500000];
    MPI_Barrier(comm);
    double tt = MPI_Wtime();
    for (int k = 0;k < N;k++) {
        MPI_Status status;
        if (rank == 0) {//first node just send and then receive
            MPI_Send(intearr, 500000, MPI_INT, 1, k, comm);
            MPI_Recv(intearr, 500000, MPI_INT, p - 1, k, comm, &status);
        }
        else if (rank == p - 1) {//last node receive and then send to first
            MPI_Recv(intearr, 500000, MPI_INT, p - 2, k, comm, &status);
            MPI_Send(intearr, 500000, MPI_INT, 0, k, comm);
        }
        else {//interior node receive and then send to next
            MPI_Recv(intearr, 500000, MPI_INT, rank - 1, k, comm, &status);
            MPI_Send(intearr, 500000, MPI_INT, rank + 1, k, comm);
        }

    }
    delete[] intearr;
    MPI_Barrier(comm);
    tt = MPI_Wtime() - tt;
    if (!rank) {
        printf("estimated bandwidth: %e GB/s\n", (500000 * N * p) / tt / 1e9);
    }
    MPI_Finalize();
}