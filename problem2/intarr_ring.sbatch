#!/bin/bash

#SBATCH --job-name=intarr_ring
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=00:40:00 
#SBATCH --output=intarr_ring.out
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=qj2030@nyu.edu

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./intarr_ring 1000