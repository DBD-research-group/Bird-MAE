#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --partition=main
#SBATCH --job-name=fewshot_linear_2
#SBATCH --output=/mnt/work/bird2vec/logs/fewshot/fewshot_linear_%N_%a.log
######SBATCH --time=01:00:00
#SBATCH --nodelist=gpu-l40s-1
#SBATCH --array=0-71%8

date; hostname; pwd
source /mnt/home/anonymous/.zshrc
echo "Activate conda"
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/anonymous/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Define arrays for datasets, shots, and seeds.
DATASETS=("hsn" "nbp" "nes" "per" "pow" "sne" "ssw" "uhh")
SHOTS=("1shot" "5shot" "10shot")
SEEDS=(1 2 3)

# Total experiments = 8 datasets * 3 shots = 24. Each config runs with 3 seeds → 72 tasks.
# Compute configuration index and seed index (zsh arrays are 1-indexed).
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))   # 0 to 23 (0-indexed)
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))       # 0 to 2 (0-indexed)

# Convert to 1-indexed positions for zsh arrays.
DATASET_IDX=$(( CONFIG_IDX / 3 + 1 ))
SHOT_IDX=$(( CONFIG_IDX % 3 + 1 ))
SEED=${SEEDS[$(( SEED_IDX + 1 ))]}

# Construct the configuration path.
CONFIG_PATH="experiment=paper/fewshot/linear/${DATASETS[$DATASET_IDX]}${SHOTS[$SHOT_IDX]}.yaml"

echo "Running experiment: ${CONFIG_PATH} with seed: ${SEED}"

DATASET_NAME=${DATASETS[$DATASET_IDX]}
scontrol update job=$SLURM_JOB_ID name="fewshot_linear_${DATASET_NAME}_seed${SEED}_${SLURM_ARRAY_TASK_ID}_3"

srun python finetune.py \
    ${CONFIG_PATH} \
    seed=${SEED} \
    module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-13_213828/callback_checkpoints/AudioMAE_XCL_epoch=149.ckpt'"