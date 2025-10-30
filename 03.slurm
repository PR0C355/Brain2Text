#!/bin/bash
#SBATCH --job-name=synthesis
#SBATCH --output=/mnt/arc/oguntola/Brain2Text/logs/%x-%j.log
#SBATCH --error=/mnt/arc/oguntola/Brain2Text/logs/%x-%j.log
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=100
#SBATCH --partition=defq
#SBATCH --gres=gpu:8

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

python 03-Synthesis.py
# python qa.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# python qa.py --model "Qwen/Qwen2.5-72B-Instruct"