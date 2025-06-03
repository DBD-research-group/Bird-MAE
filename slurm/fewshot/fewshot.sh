#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --partition=main
#SBATCH --job-name=fewshot_ppnet_hsn
#SBATCH --output=/mnt/work/bird2vec/logs/fewshot/fewshot_ppnet_hsn_%N_%t.log
#SBATCH --time=01:00:00
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
        experiment=paper/fewshot/ppnet/hsn5shot.yaml \
        module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-13_213828/callback_checkpoints/AudioMAE_XCL_epoch=149.ckpt'" \

        #trainer.precision=bf16 \
        #+trainer.num_nodes=1 \
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"

