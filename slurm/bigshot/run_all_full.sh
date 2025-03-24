#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --partition=main
#SBATCH --job-name=bigshot_ppnet_large_full
#SBATCH --output=/mnt/work/bird2vec/logs/bigshot/full_bigshot_ppnet_%N_%a.log
######SBATCH --time=01:00:00
#SBATCH --nodelist=gpu-l40s-1
#SBATCH --array=0-14%8

date; hostname; pwd
source /mnt/home/lrauch/.zshrc
echo "Activate conda"
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/lrauch/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Define array for datasets and seeds.
#DATASETS=("hsn" "nbp" "nes" "per" "pow" "sne" "ssw" "uhh")
DATASETS=("per" "sne" "ssw" "uhh" "uhh")

SEEDS=(1 2 3)

# Fixed shot configuration.
SHOT="allshot"

# Total experiments = 5 datasets * 3 seeds = 15 tasks.
# Compute dataset index and seed index (zsh arrays are 1-indexed).
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 3 + 1 ))   # Using 3 seeds per dataset
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))            # Remainder: 0, 1, or 2
SEED=${SEEDS[$(( SEED_IDX + 1 ))]}

# Construct the configuration path.
CONFIG_PATH="experiment=paper/bigshot/birdMAE/${DATASETS[$DATASET_IDX]}_large_ppnet.yaml"

echo "Running experiment: ${CONFIG_PATH} with seed: ${SEED}"
DATASET_NAME=${DATASETS[$DATASET_IDX]}
scontrol update job=$SLURM_JOB_ID name="bigshot_ppnet_full_${DATASET_NAME}_seed${SEED}_${SLURM_ARRAY_TASK_ID}"

srun python finetune.py \
    ${CONFIG_PATH} \
    seed=${SEED} \
    module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-13_213828/callback_checkpoints/AudioMAE_XCL_epoch=149.ckpt'"