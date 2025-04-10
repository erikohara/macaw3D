#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --mem=5GB

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate macaw
python /home/erik.ohara/macaw/utils/scripts/3D/2_5D/2_5D-crop-y.py $SLURM_ARRAY_TASK_ID

