#!/bin/sh

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=5   # number of threads
## SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "QG-const"   # job name
#SBATCH --no-requeue
#SBATCH --output=indirect_const.out

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



python Indirect_const.py
