#!/usr/bin/zsh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_pretrain_XCL_scratch_mgpu_ogg12
#SBATCH --output=/mnt/work/bird2vec/logs/mgpu_ogg12_%N_%t.log
#SBATCH --time=48:00:00  # Specify a 2-hour time limit
#SBATCH --exclude=gpu-v100-3

########SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4
######,gpu-a100-1,gpu-a100-2
#####SBATCH --nodelist=gpu-a100-5
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
export NCCL_DEBUG=INFO

NUM_GPUS=$SLURM_GPUS_ON_NODE

hostname
srun python pretrain.py experiment=pretrain_xcl_wave.yaml \
        trainer.devices=1 \
        +trainer.num_nodes=2 \
        trainer.precision=16

echo "Finished script."
