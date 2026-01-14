#!/usr/bin/env python3
"""
Train MedGAN on 45k MIMIC-III train split with 1k holdout.

Uses standard MedGAN hyperparameters from synthEHRella.
"""

import os
import sys
sys.path.insert(0, '/u/jalenj4/PyHealth')
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict

from pyhealth.models.generators.medgan import MedGAN


def load_mimic3_data(data_path):
    """Load MIMIC-III diagnosis data from train split."""
    print(f"Loading MIMIC-III data from: {data_path}")

    diagnoses_df = pd.read_csv(os.path.join(data_path, "DIAGNOSES_ICD.csv"))
    diagnoses_df = diagnoses_df.dropna(subset=['ICD9_CODE'])
    diagnoses_df['ICD9_CODE'] = diagnoses_df['ICD9_CODE'].astype(str)

    patient_codes = defaultdict(set)
    for _, row in diagnoses_df.iterrows():
        patient_codes[row['SUBJECT_ID']].add(row['ICD9_CODE'])

    all_codes = sorted(set().union(*patient_codes.values()))
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}

    binary_matrix = np.zeros((len(patient_codes), len(all_codes)))
    for i, (patient_id, codes) in enumerate(sorted(patient_codes.items())):
        for code in codes:
            binary_matrix[i, code_to_idx[code]] = 1

    print(f"Loaded {len(patient_codes)} patients with {len(all_codes)} unique ICD-9 codes")
    print(f"Data sparsity: {(binary_matrix == 0).mean():.4f}")
    print(f"Avg codes per patient: {binary_matrix.sum(axis=1).mean():.2f}")

    return binary_matrix, all_codes


def train_medgan(model, dataloader, n_epochs, device, save_dir, lr=0.001, weight_decay=0.0001, b1=0.5, b2=0.9):
    """Train MedGAN model."""

    def generator_loss(y_fake):
        return -torch.mean(torch.log(y_fake + 1e-12))

    def discriminator_loss(outputs, labels):
        loss = -torch.mean(labels * torch.log(outputs + 1e-12)) - torch.mean((1 - labels) * torch.log(1. - outputs + 1e-12))
        return loss

    optimizer_g = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.autoencoder.decoder.parameters(), 'lr': lr * 0.1}
    ], lr=lr, betas=(b1, b2), weight_decay=weight_decay)

    optimizer_d = torch.optim.Adam(model.discriminator.parameters(),
                                  lr=lr * 0.1, betas=(b1, b2), weight_decay=weight_decay)

    g_losses = []
    d_losses = []

    print("="*60)
    print("Epoch | D_loss | G_loss | Progress")
    print("="*60)

    for epoch in range(n_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            real_data = batch[0].to(device)
            batch_size = real_data.size(0)

            valid = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)

            z = torch.randn(batch_size, model.latent_dim).to(device)

            # Train generator
            for p in model.discriminator.parameters():
                p.requires_grad = False

            fake_samples = model.generator(z)
            fake_samples = model.autoencoder.decode(fake_samples)

            fake_output = model.discriminator(fake_samples).view(-1)
            g_loss = generator_loss(fake_output)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Train discriminator
            for p in model.discriminator.parameters():
                p.requires_grad = True

            fake_samples = model.generator(z)
            fake_samples = model.autoencoder.decode(fake_samples)

            real_output = model.discriminator(real_data).view(-1)
            fake_output = model.discriminator(fake_samples.detach()).view(-1)

            d_loss_real = discriminator_loss(real_output, valid)
            d_loss_fake = discriminator_loss(fake_output, fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"{epoch+1:5d} | {avg_d_loss:6.4f} | {avg_g_loss:6.4f} | {(epoch+1)/n_epochs*100:.1f}%")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"medgan_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'autoencoder_state_dict': model.autoencoder.state_dict(),
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, checkpoint_path)

    return {'g_losses': g_losses, 'd_losses': d_losses}


def main():
    parser = argparse.ArgumentParser(description="Train MedGAN on 45k train split")
    parser.add_argument("--train_path", default="/u/jalenj4/pehr_scratch/data_files_train", help="Path to train data")
    parser.add_argument("--output_path", default="/scratch/jalenj4/medgan_45k_results", help="Output directory")
    parser.add_argument("--ae_epochs", type=int, default=100, help="Autoencoder pretraining epochs")
    parser.add_argument("--gan_epochs", type=int, default=1000, help="GAN training epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on: {args.train_path}")
    print(f"Output to: {args.output_path}")

    # Load train data
    binary_matrix, code_vocab = load_mimic3_data(args.train_path)
    n_patients, n_codes = binary_matrix.shape

    # Create dataset
    tensor_data = torch.FloatTensor(binary_matrix)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize MedGAN from binary matrix
    print(f"\nInitializing MedGAN (latent_dim={args.latent_dim})...")
    model = MedGAN.from_binary_matrix(
        binary_matrix=binary_matrix,
        latent_dim=args.latent_dim,
        autoencoder_hidden_dim=128,
        discriminator_hidden_dim=256,
        minibatch_averaging=True
    ).to(device)

    # Pretrain autoencoder
    print(f"\nPretraining autoencoder for {args.ae_epochs} epochs...")
    ae_optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.ae_epochs):
        total_loss = 0
        for batch_tuple in dataloader:
            batch = batch_tuple[0].to(device)
            reconstructed = model.autoencoder(batch)
            loss = torch.nn.functional.binary_cross_entropy(reconstructed, batch)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"AE Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Train GAN
    print(f"\nTraining GAN for {args.gan_epochs} epochs...")
    history = train_medgan(model, dataloader, args.gan_epochs, device, args.output_path,
                          lr=args.lr, weight_decay=args.weight_decay)

    # Save final checkpoint
    final_path = os.path.join(args.output_path, "medgan_final.pth")
    torch.save({
        'autoencoder_state_dict': model.autoencoder.state_dict(),
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'history': history,
    }, final_path)
    print(f"\nSaved final checkpoint to: {final_path}")

    # Save vocabulary
    vocab_path = os.path.join(args.output_path, "icd9_vocabulary.txt")
    with open(vocab_path, 'w') as f:
        for code in code_vocab:
            f.write(f"{code}\n")
    print(f"Saved vocabulary to: {vocab_path}")

    # Save binary matrix
    matrix_path = os.path.join(args.output_path, "icd9_binary_matrix.npy")
    np.save(matrix_path, binary_matrix)
    print(f"Saved binary matrix to: {matrix_path}")

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
