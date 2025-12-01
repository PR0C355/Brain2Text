#!/bin/bash
#SBATCH --job-name=b2t-train
#SBATCH --output=/mnt/arc/oguntola/Brain2Text/logs/%x-%j.log
#SBATCH --error=/mnt/arc/oguntola/Brain2Text/logs/%x-%j.log
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1

cd /mnt/arc/oguntola/Brain2Text/
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Check for "node" in hostname
if [[ $(hostname) == *"node"* ]]; then
    echo "Running on sunflower node, TMPDIR set to $TMPDIR"
else
    export TMPDIR=/playpen-storage/oguntola/.tmp
    echo "Not running on sunflower, set TMPDIR to $TMPDIR"
    mkdir -p $TMPDIR
fi

echo "Running on $(hostname) at $(date)"

echo $CUDA_VISIBLE_DEVICES

python 04-Train.py --gpu $CUDA_VISIBLE_DEVICES --logdir $SLURM_JOB_ID