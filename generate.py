from modeling import Unet1d
from diffusion import DiffusionGene
from settings import args
import numpy as np
import torch


# Run this file to generate RNA-seq samples.
# Reset savepath and other parameters in this file (bottom).


def get_sub_time_seq(acc_rate: int):
    """
    We offer an example of generating timestep subsequences here.
    You can edit this function or write another function to generate different subsequences.
    :param acc_rate: accelerate rate. (int)
    :return: timestep subsequence. (list)
    """
    sub_time_seq = [i for i in range(0, 1001, acc_rate)]
    sub_time_seq.insert(1, 1)
    return sub_time_seq


def sample_batches(model, amount=1024, savepath: str = None, method='ddpm', sub_time_seq: list = None, eta: float = 1.,
                   clamp=False):
    """
    Generate RNA-seq samples with trained model.
    If you want to generate RNA-seq data with a minimum value of zero (no negative values), set clamp to True.
    :param model: model checkpoint's path.
    :param amount: sample size.
    :param savepath: where to save the samples.
    :param method: choose 'ddpm' or 'ddim' to accelerate sampling.
    :param sub_time_seq: send a sub-sequence of time step if you want to use DDIM. (should be a list)
    :param clamp: if true, erase all numbers less than zero from the generated sample to zero
    :return: None
    """
    device = args.device
    num = args.batch_size
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device)
    batches = None
    rest = amount % num
    for i in range(amount // num):
        print(f'generating batch {i + 1} ...')
        if method == 'ddpm':
            batch = diffusion.sample(model, n=num, clamp=clamp).to('cpu')
        else:
            batch = diffusion.sample_ddim(model, n=num, eta=eta, sub_time_seq=sub_time_seq, clamp=clamp).to('cpu')
        if batches is None:
            batches = batch
        else:
            batches = torch.cat((batches, batch), dim=0)
        print(batches.shape)
        if savepath:
            np.save(savepath, batches.numpy())

    if rest != 0:
        print(f'generating last {rest} samples ...')
        if method == 'ddpm':
            batch = diffusion.sample(model, n=rest, clamp=clamp).to('cpu')
        else:
            batch = diffusion.sample_ddim(model, n=rest, eta=eta, sub_time_seq=sub_time_seq, clamp=clamp).to('cpu')
        if batches is None:
            batches = batch
        else:
            batches = torch.cat((batches, batch), dim=0)
        print(batches.shape)
        if savepath:
            np.save(savepath, batches.numpy())


if __name__ == '__main__':
    # generating settings
    model_path = 'models/fibroblast_diffusion_ckpt.pt'
    # model_path = 'models/malignant_diffusion_ckpt.pt'
    sample_amount = 1024  # number of generated samples
    method = 'ddim'  # use 'ddpm' or 'ddim'
    save = 'results/fibroblast_samples'  # result savepath
    acc_rate = 10

    model = Unet1d().to(args.device)
    model.load_state_dict(torch.load(model_path))
    sub_seq = get_sub_time_seq(acc_rate)

    sample_batches(model, amount=sample_amount, savepath=save, method=method, sub_time_seq=sub_seq, clamp=True)
