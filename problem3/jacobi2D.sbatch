#!/bin/bash

#SBATCH --job-name=jacobi2D
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00 
#SBATCH --output=jacobi2D.out
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=qj2030@nyu.edu

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./jacobi2D 1001 2000