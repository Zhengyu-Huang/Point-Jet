#!/bin/bash

#SBATCH --time=01:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH -J "beta_0p5"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2

################################
# Point jet DNS data generation #
################################
# submit with: sbatch job.sh

set -euo pipefail

# simulation parameters
beta=1.0
Gamma=1.0
# relax_values=(0.32 0.16 0.08 0.04 0.02 0.01 0.005 0.002 0.001)  
relax_values=(0.2 0.06 0.005)
relax=${relax_values[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/beta_${beta}_Gamma_${Gamma}_relax_${relax}"
stdout="${output_path}/stdout_data"
stderr="${output_path}/stderr_data"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

module load matlab/r2021a
matlab -nodisplay -nodesktop -r "load_data ${beta} ${Gamma} ${relax} ${output_path}" \
    >${stdout} 2>${stderr}
