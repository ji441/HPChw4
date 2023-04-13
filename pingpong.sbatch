#!/bin/bash

#SBATCH --job-name=pingpong
#SBATCH --nodes=2 
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=00:20:00 
#SBATCH --verbose

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./pingpong 0 1