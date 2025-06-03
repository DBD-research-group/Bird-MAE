#!/usr/bin/zsh
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --partition=main
#SBATCH --job-name=birdAVES-hsn
#SBATCH --output=/mnt/work/bird2vec/logs_mw/%x_%j_%t.log
#SBATCH --time=96:00:00
#SBATCH --exclude=gpu-a100-5,gpu-v100-[1-4],gpu-l40s-1
####SBATCH --array=3-3%3

date;hostname;pwd
source /mnt/stud/home/anonymous/.zshrc

conda activate GADME
echo $PYTHONPATH

cd /mnt/stud/home/anonymous/projects/birdMAE

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

srun python finetune.py \
        experiment=BirdAVES/finetune_hsn.yaml