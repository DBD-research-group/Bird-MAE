#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --partition=main
#SBATCH --job-name=bigshot_ppnet_large
#SBATCH --output=/mnt/work/bird2vec/logs/bigshot/bigshot_ppnet_hsntest_%N.log
######SBATCH --time=01:00:00
#SBATCH --nodelist=gpu-l40s-1

date; hostname; pwd
source /mnt/home/anonymous/.zshrc
echo "Activate conda"
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/anonymous/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Fixed dataset and seed values.
DATASET="hsn"
SEED=1

# Construct the configuration path.
CONFIG_PATH="experiment=paper/bigshot/birdMAE/${DATASET}_large_ppnet.yaml"

echo "Running experiment: ${CONFIG_PATH} with seed: ${SEED}"
scontrol update job=$SLURM_JOB_ID name="bigshot_ppnet_${DATASET}_seed${SEED}"

srun python finetune.py \
    ${CONFIG_PATH} \
    seed=${SEED} \
    module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-13_213828/callback_checkpoints/AudioMAE_XCL_epoch=149.ckpt'"