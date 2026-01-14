#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=gen_corgan_10k
#SBATCH --output=slurm-%j.out
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Generating 10k synthetic patients using CorGAN 45k checkpoint"

source /u/jalenj4/pehr_scratch/venv/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth:$PYTHONPATH

python examples/generate_synthetic_mimic3_corgan.py \
    --checkpoint /scratch/jalenj4/corgan_45k_results/corgan_best.pth \
    --vocab /scratch/jalenj4/corgan_45k_results/icd9_vocabulary.txt \
    --binary_matrix /scratch/jalenj4/corgan_45k_results/icd9_binary_matrix.npy \
    --output /scratch/jalenj4/synthetic_corgan_10k.csv \
    --n_samples 10000 \
    --mean_k 13 \
    --std_k 5

echo "Generation complete!"
