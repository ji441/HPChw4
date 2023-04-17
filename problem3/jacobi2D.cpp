#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include "jacobi.h"
#include "compute_residual.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int iternum;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;
    int p;//number of nodes/processes we have.
    MPI_Comm_size(comm, &p);
    if (p != 4) {
        printf("number of processes must be 4");
        abort();
    }
    int N;//nodes per dimension
    if (argc == 3)
    {
        N = atoi(argv[1]);
        if (N % 2 == 0)
        {
            printf("number of nodes must be odd !");
            abort();
        }
        iternum = atoi(argv[2]);
    }
    else
    {
        printf("please give me one input for the number of nodes which must be odd! and one int for number of iterations");
        abort();
    }
    int n = (N - 1) / 2 + 1; //get the number of nodes in subpieces
    //allocate arrays
    double** phi = new double* [n];
    double** aux = new double* [n];
    double** f = new double* [n];
    for (int i = 0;i < n;i++)
    {
        phi[i] = new double[n];
        aux[i] = new double[n];
        f[i] = new double[n];
    }
    // set boundary condition and initial guess for phi and f
    //here since we have boundary value and initial guess all 0
    //and the f === 1 on all interior points, we do not need to assign spcifically according to rank.
    MPI_Barrier(comm);
    double tt = MPI_Wtime();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == 0 || j == 0 || i == n - 1 || j == n - 1)
            {
                // boundarycondition goes here
                phi[i][j] = 0;
            }
            else
            {
                // initial guess goes here
                phi[i][j] = 0;
            }
            f[i][j] = 1;
        }
    }
    if (rank == 0)
    {
        printf("Solution initialized.\n");
    }
    //calculate the initial error
    double norm0 = compute_residual(phi, f, n, comm);
    //now we begin jacobi
    jacobi(phi, aux, f, n, iternum, comm);
    if (rank == 0)
    {
        printf("jacobi iterations finished.\n");
    }
    //after fixed number of iterations, we compute the residual
    double norm = compute_residual(phi, f, n, comm);
    //clean up
    for (int i = 0;i < n;i++)
    {
        delete[] phi[i];
        delete[] aux[i];
        delete[] f[i];
    }
    delete[] phi;
    delete[] aux;
    delete[] f;
    MPI_Barrier(comm);
    tt = MPI_Wtime() - tt;
    if (rank == 0)
    {
        printf("Initial error is %5f,after %d iterations, the error is: %5f,time spent is: %e ms\n",
            norm0, iternum, norm, tt * 1000);
    }
    MPI_Finalize();


}