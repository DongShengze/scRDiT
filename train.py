import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from settings import args
from unet import Unet1d
import logging
from torch.utils.tensorboard import SummaryWriter
from loader import cell_dataloader
from diffusion import DiffusionGene
from transformer import DiT

# Run this file to train your model.
# Change training parameters in settings.py.


def train_ddpm(args):
    """
    Rewrite args parameters in settings.py
    :param args: Settings.
    :return: None
    """
    run_name = args.run_name
    device = args.device
    model = args.model.to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(f'ckpts/{args.run_name}/{run_name}_epoch{args.ckpt_epoch}.pt'))
    train_loss_list = list()
    dataloader = cell_dataloader

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)

    if args.ckpt:
        optimizer.load_state_dict(torch.load(f'optim/{args.run_name}_AdamW.pt'))

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss_list = list()

        for i, genes in enumerate(pbar):
            genes = genes.to(device)
            t = diffusion.sample_timesteps(genes.shape[0]).to(device)
            x_t, noise = diffusion.noise_genes(genes, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        train_loss_list.append(avg_epoch_loss)

        print("epoch: ", epoch, " avg_loss: ", avg_epoch_loss)

        if (epoch + 1) % args.save_frequency == 0:
            torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, f"{args.run_name}_epoch{epoch}.pt"))
            torch.save(optimizer.state_dict(), f'optim/{args.run_name}_AdamW.pt')


if __name__ == '__main__':
    train_ddpm(args)
