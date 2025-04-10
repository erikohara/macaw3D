#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL

####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python -u /home/erik.ohara/macaw/utils/scripts/3D/vqvae2D/MACAWtraining.py $SLURM_ARRAY_TASK_ID
