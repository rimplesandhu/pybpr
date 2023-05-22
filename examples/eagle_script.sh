#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --time=7:00:00             # Job should run for up to 6 hours
#SBATCH --account=zazzle  # Where to charge NREL Hours
#SBATCH --output=log.%j.out
##SBATCH --partition=debug
#SBATCH --job-name=wals_zazzle

module load conda
conda activate bpr_env
conda env list
python $1
#jupyter nbconvert --to notebook --inplace --execute $1
