#!/bin/sh

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=32   # number of threads
## SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "QG"   # job name
#SBATCH --no-requeue
#SBATCH --output=indirect_NN.out

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



python Indirect_NN.py
