import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from numpy.core.fromnumeric import prod

from .autoencoder import utils
from .autoencoder.moco import builder
from .autoencoder.model_ae_moco import AutoEncoder
from .diffusion.functions.denoising import compute_alpha, generalized_steps
from .diffusion.models.diffusion import Model as Diffusion


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class ModulationLayer(nn.Module):
    def __init__(self, *shape):
        super(ModulationLayer, self).__init__()

        self.register_buffer("mean", torch.zeros(*shape))
        self.register_buffer("std", torch.ones(*shape))

    def forward(self, z):
        return (z - self.mean) / self.std

    def backward(self, z_):
        return z_ * self.std + self.mean


class BoundaryInterpolationLayer(nn.Module):
    def __init__(self, *shape):
        super(BoundaryInterpolationLayer, self).__init__()
        self.register_buffer("boundary", torch.randn(*shape))

    def forward(self, z, steps):
        return z - steps * self.boundary


class D2C(object):
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config
        self.autoencoder_config = cae = config.autoencoder
        self.diffusion_config = cdif = config.diffusion

        arch_instance = utils.get_arch_cells(self.autoencoder_config.arch_instance)

        def autoencoder_fn():
            return AutoEncoder(self.autoencoder_config, None, arch_instance)

        latent_dim = cae.latent_dim
        dim_mlp_next = None
        latent_channels = cae.num_latent_per_group
        self.latent_size = (latent_channels, latent_dim, latent_dim)

        self.autoencoder = builder.MoCo(
            autoencoder_fn,
            cae.moco_dim,
            cae.moco_k,
            cae.moco_m,
            cae.moco_t,
            False,
            dim_mlp=prod(self.latent_size),
            dim_mlp_next=dim_mlp_next,
        ).cuda()

        self.diffusion = Diffusion(cdif).cuda()
        self.latent_mod = ModulationLayer((1, latent_channels, 1, 1)).cuda()

        self.betas = (
            torch.from_numpy(
                get_beta_schedule(
                    beta_schedule=cdif.diffusion.beta_schedule,
                    beta_start=cdif.diffusion.beta_start,
                    beta_end=cdif.diffusion.beta_end,
                    num_diffusion_timesteps=cdif.diffusion.num_diffusion_timesteps,
                )
            )
            .cuda()
            .float()
        )

    def load_state_dict(self, state_dict, inference=True):
        if inference:
            # MLP is not used in inference, so we do not load it.
            for k in list(state_dict["autoencoder"].keys()):
                if k.startswith("encoder_q.fc") or k.startswith("encoder_k.fc"):
                    del state_dict["autoencoder"][k]
        else:
            raise NotImplementedError("The class only supports inference.")
        self.autoencoder.load_state_dict(state_dict["autoencoder"])
        self.diffusion.load_state_dict(state_dict["diffusion"])
        self.latent_mod.load_state_dict(state_dict["latent_mod"])

    def image_to_latent(self, x):
        x = x.cuda(non_blocking=True)
        x = utils.pre_process(x.cuda(), self.autoencoder_config.num_x_bits)
        z = self.autoencoder.encoder_q(x, get_latent=True, reshape=False)
        return z

    def latent_to_image(self, z):
        logits = self.autoencoder.encoder_q(z, from_latent=True)
        xdist = self.autoencoder.encoder_q.decoder_output(logits)

        if isinstance(xdist, torch.distributions.Bernoulli):
            x = xdist.mean()
        else:
            x = xdist.sample()
        return x

    def transform_latent(self, z_, seq):
        z = generalized_steps(z_, seq, self.diffusion, self.betas, eta=0.0, last=True)
        return z

    def sample_latent(self, batch_size, skip=1):
        num_timesteps = len(self.betas)
        z_ = torch.randn(batch_size, *self.latent_size).cuda()
        seq = list(range(0, num_timesteps - 1, skip)) + [num_timesteps - 1]
        z = self.transform_latent(z_, seq)
        z = self.latent_mod.backward(z)
        return z

    def postprocess_latent(self, z, seq):
        z = self.latent_mod(z)
        a = compute_alpha(
            self.betas, seq[-1] * torch.ones(z.size(0)).long().to(z.device)
        ).to("cuda")
        z_ = a.sqrt() * z + (1 - a).sqrt() * torch.randn_like(z)

        z_post = self.transform_latent(z_, seq)

        return self.latent_mod.backward(z_post)

    def manipulate_latent(self, z, r_model, step):
        new_z = r_model(z, step)
        return new_z

    def eval(self):
        self.autoencoder.eval()
        self.diffusion.eval()
        self.latent_mod.eval()
