#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --mem=16GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python -u /home/erik.ohara/macaw/utils/scripts/3D/vqvae2D/VQVAE2Dtraining.py $SLURM_ARRAY_TASK_ID
