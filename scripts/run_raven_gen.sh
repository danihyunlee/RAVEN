#!/bin/bash
#SBATCH -p vision-pulkitag-a6000
#SBATCH --qos=vision-pulkitag-free-cycles
#SBATCH --account=vision-pulkitag
#SBATCH -t 8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/raven_gen_%j.out
#SBATCH --error=logs/raven_gen_%j.err

# create logs
mkdir -p logs

# launch scripts
python src/dataset/main.py --num-samples 10000 --save-dir RAVEN_data/10000
