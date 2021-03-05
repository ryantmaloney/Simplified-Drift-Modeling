#!/bin/bash
#SBATCH -n 24 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 15 # Runtime in minutes
#SBATCH -p shared # Partition to submit to
#SBATCH --mem=1024 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH --output=hostname_%j_array_%A-%a.out    # Standard output and error log
#SBATCH --test-only
#SBATCH --array=1-5                 # Array range
#SBATCH 
module load Anaconda3/2019.10
source activate drift
python clusterscript.py