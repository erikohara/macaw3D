#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=300GB
##SBATCH --partition=bigmem


####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python /home/erik.ohara/macaw/utils/scripts/3D/PCA_sklearn/testPCAinvert.py
