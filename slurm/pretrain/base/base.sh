#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:4
#SBATCH --mem=500gb
#SBATCH --partition=main
#SBATCH --job-name=100ep_birdMAE_XCL_base_0.3mix
#SBATCH --output=/mnt/work/bird2vec/logs/without_siwn/birdMAE_XCL_base_%N_%t_0.3mix_100ep_reproduce.log
#SBATCH --time=120:00:00
#SBATCH --nodelist=gpu-l40s-1


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
srun python pretrain.py \
        experiment=paper/pretrain/pretrain_xcl_wave_base.yaml \
        trainer.devices=4 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16 \
        data.transform.waveform_augmentations.mixup_wave.p=0.3 \
        trainer.max_epochs=150\
        data.loaders.train.batch_size=256 \
        module.network.mask_ratio=0.75 \
        trainer.gradient_clip_val=1.0

        #data.dataset.save_to_disk="/scratch/birdset/XCL/XCL_processd_500_2events_ogg_addsoundscapes-hsn" \
        #trainer.strategy=ddp_find_unused_parameters_true \
        ##ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_large_swin/runs/XCL/AudioMAE/2024-12-12_162203/callback_checkpoints/last.ckpt"
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"



echo "Finished script."
