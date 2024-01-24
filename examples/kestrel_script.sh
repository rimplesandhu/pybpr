#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --time=1-0:00:00             # Job should run for up to 6 hours
#SBATCH --account=zazzle  # Where to charge NREL Hours
#SBATCH --output=log.%j.out
##SBATCH --partition=debug
#SBATCH --job-name=bpr_zazzle

module load anaconda3
source activate bpr_env
python zazzle_run.py
