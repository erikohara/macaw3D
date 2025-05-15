#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=9GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100



####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python /home/erik.ohara/macaw/utils/scripts/3D/2_5D/axisX/training-script-x.py $SLURM_ARRAY_TASK_ID
