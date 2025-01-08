#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:4
#SBATCH --mem=450gb
#SBATCH --partition=main
#SBATCH --job-name=100ep_birdMAE_XCL_large_0.5mix_reproduce
#SBATCH --output=/mnt/work/bird2vec/logs/without_siwn/birdMAE_XCL_large_%N_%t_0.5mix_100ep_reproduce.log
#SBATCH --time=100:00:00
#SBATCH --nodelist=gpu-l40s-1

###SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4
######,gpu-a100-1,gpu-a100-2
#####SBATCH --nodelist=gpu-a100-5
####SBATCH --array=3-3%3

date;hostname;pwd
source /mnt/home/lrauch/.zshrc
#source ~/envs/gadme_v1/bin/activate
echo Activate conda
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/lrauch/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

hostname
srun python train_ssl.py \
        experiment=pretrain_xcl_wave_large.yaml \
        trainer.devices=4 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16 \
        data.transform.waveform_augmentations.mixup_wave.p=0.5 \
        trainer.max_epochs=100\
        data.loaders.train.batch_size=256 \
        module.network.mask_ratio=0.75 \
        trainer.gradient_clip_val=2.0

        #data.dataset.save_to_disk="/scratch/birdset/XCL/XCL_processd_500_2events_ogg_addsoundscapes-hsn" \
        #trainer.strategy=ddp_find_unused_parameters_true \
        ##ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_large_swin/runs/XCL/AudioMAE/2024-12-12_162203/callback_checkpoints/last.ckpt"
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"

if [ $? -ne 0 ]; then
    echo "Error: srun failed. Sleeping for 2 hours..."
    sleep 6h
    # Optionally, exit non-zero to signal failure
    exit 1
fi


echo "Finished script."
