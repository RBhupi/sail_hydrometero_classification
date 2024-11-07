#!/bin/bash

year=2022  
months=(1 8) #(1 2 3 4 5 6)  # months to process
data_dir="/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradarcmacS2.c1/ppi/"
output_dir="/gpfs/wolf2/arm/atm124/proj-shared/HydroPhase/"

job_name="hp_processing"
output_log_dir="${output_dir}/logs"
mkdir -p "$output_log_dir"

# Loop jobs
for month in "${months[@]}"; do
    # add zero if needed
    month_padded=$(printf "%02d" "$month")

    # use season based on month
    if [[ $month -ge 4 && $month -le 9 ]]; then
        season="summer"
    else
        season="winter"
    fi

    # job submission command
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_${year}_${month_padded}
#SBATCH --output=${output_log_dir}/${job_name}_${year}_${month_padded}_%j.out
#SBATCH --error=${output_log_dir}/${job_name}_${year}_${month_padded}_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --account=ATM124
#SBATCH --mail-type=ALL
#SBATCH --mail-user=braut@anl.gov

# Load modules and activate environment
module load python/3.7-anaconda3
source /ccsopen/home/braut/analysis-env2/bin/activate

# Run the script 
/ccsopen/home/braut/analysis-env2/bin/python /ccsopen/home/braut/hclass/code/Python/hp_processing.py "$year" "$month_padded" --data_dir "$data_dir" --output_dir "$output_dir" --season "$season"

# Deactivate the conda environment
conda deactivate
EOF

done