#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100


####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python /home/erik.ohara/macaw/utils/scripts/3D/PCA_sklearn/training-script.py $SLURM_ARRAY_TASK_ID
