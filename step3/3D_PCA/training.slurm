#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=1550GB
#SBATCH --partition=bigmem
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL


####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python -u /home/erik.ohara/macaw/utils/scripts/3D/PCA3D/training.py
