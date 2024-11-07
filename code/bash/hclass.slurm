#!/bin/bash
#SBATCH --job-name=radar_processing
#SBATCH --output=hclass_%j.out
#SBATCH --error=hclass_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=braut@anl.gov

# Load modules
module load python/3.7-anaconda3

# Activate conda environment
source /ccsopen/home/braut/analysis-env2/

# Run script
python /ccsopen/home/braut/hclass/code/Python/hclass_processing.py
# Deactivate the conda environment
conda deactivate
