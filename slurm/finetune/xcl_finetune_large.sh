#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:4
#SBATCH --mem=500gb
#SBATCH --partition=main
#SBATCH --job-name=finetune_XCL_large
#SBATCH --output=/mnt/work/bird2vec/logs/finetune/finetune_XCL_large_%N_%t.log
#SBATCH --time=96:00:00
#SBATCH --nodelist=gpu-l40s-1

####SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4,gpu-a100-4

######,gpu-a100-1,gpu-a100-2
####SBATCH --array=3-3%3

date;hostname;pwd
source /mnt/home/anonymous/.zshrc
#source ~/envs/gadme_v1/bin/activate
echo Activate conda
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/anonymous/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

hostname
srun python finetune.py \
        experiment=finetune_xcl.yaml \
        trainer.devices=4 \
        trainer.precision=bf16 \
        module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-12-20_143556/callback_checkpoints/AudioMAE_XCL_epoch=99.ckpt'" \
        #+trainer.num_nodes=1 \
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"

# Capture the exit status of the Python script
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Python script encountered an error (exit status: $EXIT_STATUS). Waiting for 60 seconds before exiting..."
    sleep 60m  # Adjust the wait time as needed
else
    echo "Python script completed successfully."
fi

echo "Finished script."
