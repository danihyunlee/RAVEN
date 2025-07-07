#!/bin/bash
#SBATCH -p vision-pulkitag-a6000
#SBATCH --qos=vision-pulkitag-free-cycles
#SBATCH --account=vision-pulkitag
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/raven_train/raven_train_%j.out
#SBATCH --error=logs/raven_train/raven_train_%j.err

# create logs
mkdir -p logs/raven_train
model="CNN_MLP_MIN"
dset="/data/scratch/danicmhlee/data/Artificial2Intelligence/datasets/rpm/10000-samples/160x160/multi-rules"
# launch scripts
python src/model/main.py --model $model --path $dset --img_size 80 --wandb --wandb_name "$model/IRAVEN"
