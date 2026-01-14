#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2@gmail.com

cd /u/jalenj4/PyHealth
source /u/jalenj4/pehr_scratch/venv/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth:$PYTHONPATH

echo "Training CorGAN on 45k MIMIC-III train split"
echo "Using V2 hyperparameters: batch_size=128, lr=0.0002, frozen decoder"
echo "Train data: /u/jalenj4/pehr_scratch/data_files_train"
echo "Output: /scratch/jalenj4/corgan_45k_results"

python train_corgan_45k.py \
    --train_path /u/jalenj4/pehr_scratch/data_files_train \
    --output_path /scratch/jalenj4/corgan_45k_results \
    --ae_epochs 50 \
    --gan_epochs 500 \
    --batch_size 128 \
    --lr 0.0002 \
    --latent_dim 128 \
    --weight_decay 0.0001 \
    --n_iter_D 5
