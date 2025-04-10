#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL
##SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100



####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate macaw    
python -u /home/erik.ohara/macaw/utils/scripts/3D/AE3D/cf-generation_all_same_sex_reverse.py
