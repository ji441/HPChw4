#!/bin/bash

#SBATCH --job-name=int_ring
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=00:30:00 
#SBATCH --output=int_ring.out
#SBATCH --verbose

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./int_ring 1000