#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=150gb
#SBATCH --partition=main
#SBATCH --job-name=eat_base_xcl_var_patched_multiview_8_informed_clone8
#SBATCH --output=/mnt/work/bird2vec/logs/eat/eat_base_var_patched_multiview_8_informed_clone8.log
#SBATCH --time=6-15:00:00  
###SBATCH --exclude=gpu-v100-3
#SBATCH --nodelist=gpu-l40s-1

date;hostname;pwd
source /mnt/home/lrauch/.zshrc
#source ~/envs/gadme_v1/bin/activate
echo Activate conda
conda activate gadme_v1_lightningup
echo $PYTHONPATH

cd /mnt/home/lrauch/projects/birdMAE/


# export CUDA_LAUNCH_BLOCKING=1
# export HYDRA_FULL_ERROR=1

hostname
srun python pretrain.py \
        experiment=eat/pretrain_xcl_eat_base.yaml \
        task_name="eat_base_xcl_var_patch_multiview" \
        trainer.devices=8 \
        +trainer.num_nodes=1 \
        trainer.precision=16-mixed \
        trainer.strategy=ddp_find_unused_parameters_true \
        data.transform.waveform_augmentations.mixup_wave.p=0.0 \
        trainer.max_epochs=60 \
        data.loaders.train.batch_size=32 \
        data.loaders.train.num_workers=16 \
        data.loaders.train.pin_memory=true \
        +data.loaders.train.prefetch_factor=2 \
        trainer.gradient_clip_val=1.0 \
        module.optimizer.target.lr=5e-4 \
        module.network.task.cls_task="regression" \
        module.network.task.feature_regularizer="var" \
        module.network.task.clustering_regularizer=null \
        module.network.task.regularize_patch_tokens=true \
        module.network.task.use_teacher_assistant=false \
        module.network.compile_mode=null \
        module.network.task.multi_view=true \
        module.network.encoder.drop_masked_tokens=true \
        module.network.encoder.informed_masking=true \
        module.network.encoder.clone_size=8

        #data.dataset.save_to_disk="/scratch/birdset/XCL/XCL_processed_500_2events_ogg_addsoundscapes-hsn" \
        #trainer.strategy=ddp_find_unused_parameters_true \
        ##ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_large_swin/runs/XCL/AudioMAE/2024-12-12_162203/callback_checkpoints/last.ckpt"
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"


exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Training failed with exit code $exit_code, sleeping for 15 hours..."
    sleep 54000  # 15h on failure
else
    echo "Training completed successfully! Sleeping for 3 hours..."
    sleep 54000  # 3h on success
fi

echo "Finished script."


