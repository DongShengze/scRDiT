import torch
from tqdm import tqdm
from modeling import Unet1d
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DiffusionGene:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, gene_size=2000, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gene_size = gene_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)  # β of DDPM
        self.alpha = 1. - self.beta  # α
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # multiplicative α

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  # uniform β array

    def noise_genes(self, x, t):
        """add noise"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        epsilon = torch.randn_like(x)  # Gaussian noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """
        generate timesteps
        :param n:
        :return:
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, clamp=False):
        """
        DDPM sampling method.
        :param model:
        :param n:
        :param clamp:
        :return:
        """
        logging.info(f"Sampling {n} new genes....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.gene_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        if clamp:
            x = torch.clamp(x, min=0.0)
        return x

    def sample_ddim(self, model, n, eta=1.0, sub_time_seq=None, clamp=False):
        """
        DDIM sampling method.
        If you want to generate RNA-seq data with a minimum value of zero (no negative values), set clamp to True.
        :param model:
        :param n:
        :param eta:
        :param sub_time_seq:
        :param clamp:
        :return:
        """
        logging.info(f"Sampling {n} new genes with DDIM....")
        model.eval()
        if sub_time_seq:
            sub_time_seq = list(sub_time_seq)
        else:
            sub_time_seq = [k for k in range(self.noise_steps + 1)]
        with torch.no_grad():
            x = torch.randn((n, 1, self.gene_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                if i not in sub_time_seq:
                    x = x
                    continue

                time_index = sub_time_seq.index(i)
                i_pre = sub_time_seq[time_index - 1]
                # print(f"{i}:{i_pre}", end=' ')

                t = (torch.ones(n) * i).long().to(self.device)
                t_pre = (torch.ones(n) * i_pre).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None]
                # alpha_hat_pre = self.alpha_hat[t - 1][:, None, None]
                alpha_hat_pre = self.alpha_hat[t_pre][:, None, None]
                sigma = eta * torch.sqrt((1 - alpha_hat_pre) / (1 - alpha_hat)) * torch.sqrt(
                    1 - alpha_hat / alpha_hat_pre)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x0_predicted = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                mean_predicted = torch.sqrt(1 - alpha_hat_pre - sigma ** 2) * predicted_noise
                x = torch.sqrt(alpha_hat_pre) * x0_predicted + mean_predicted + sigma * noise
        model.train()
        if clamp:
            x = torch.clamp(x, min=0.0)
        return x


if __name__ == '__main__':
    # Test code.
    diffusion = DiffusionGene()
    model = Unet1d()
    model.to('cuda')
    X = diffusion.sample(model, 1)
    X = X.to('cpu')
    print(X.shape)
