#!/bin/bash

#SBATCH --job-name=pingpong
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --mem=1GB 
#SBATCH --time=01:00:00 
#SBATCH --output=pingpong.out
#SBATCH --verbose
#SBATCH --mail-type=END
#SBATCH --mail-user=qj2030@nyu.edu

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./pingpong 0 1