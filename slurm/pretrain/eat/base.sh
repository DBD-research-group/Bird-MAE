#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 2 ranks / Python processes
####SBATCH --gpus-per-task=1            # 1 GPU per rank
#SBATCH --gres=gpu:2                 # total GPUs 
#SBATCH --cpus-per-task=8
#SBATCH --mem=280G
#SBATCH --partition=main
#SBATCH --nodelist=gpu-l40s-1
#SBATCH --time=6-00:00:00            # 6 days
#SBATCH --job-name=eat_base_xcl_red
#SBATCH --output=/mnt/work/bird2vec/logs/eat/eat_base_%j.log

date; hostname; pwd

source /mnt/home/lrauch/.zshrc
conda activate gadme_v1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK       # Torch/OpenMP
# export NCCL_IB_DISABLE=1      # skip Infiniband/RoCE
# export NCCL_NET=socket        # force TCP for any NIC traffic

cd /mnt/home/lrauch/projects/birdMAE/

srun python pretrain.py \
      experiment=eat/pretrain_xcl_eat_base.yaml \
      trainer.strategy=auto \
      trainer.accelerator=gpu \
      trainer.devices=2 \
      +trainer.num_nodes=1 \
      trainer.precision=16-mixed \
      trainer.max_epochs=30 \
      trainer.gradient_clip_val=1.0 \
      data.transform.waveform_augmentations.mixup_wave.p=0.0 \
      data.loaders.train.batch_size=64 \
      data.loaders.train.num_workers=${SLURM_CPUS_PER_TASK} \
      data.loaders.train.pin_memory=true \
      +data.loaders.train.prefetch_factor=2 \
      module.optimizer.target.lr=5e-4

echo "Finished script."