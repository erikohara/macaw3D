#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=300GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
##SBATCH --mail-user=erik.ohara@ucalgary.ca
##SBATCH --mail-type=ALL

####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python /home/erik.ohara/macaw/utils/scripts/3D/2_5D/axisX/cf-generation_all_same_age_reverse.py $1
