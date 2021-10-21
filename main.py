import argparse
import os

import torch
import torch.distributed as dist
import torchvision
import torchvision.datasets as tdatasets
import torchvision.transforms as transforms
import tqdm
import yaml

from d2c import D2C, BoundaryInterpolationLayer


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument(
        "action",
        type=str,
        help="Action to perform with D2C",
        choices=["manipulation", "sample_uncond"],
    )
    parser.add_argument("--d2c_path", type=str, required=True, help='D2C model location.')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init_method', type=str, default="tcp://localhost:10002")
    args, _ = parser.parse_known_args()

    if args.action == "manipulation":
        parser.add_argument(
            "--boundary_path",
            type=str,
            required=True,
            help="Path to save the interpolation model on the latents (attribute specific).",
        )
        parser.add_argument(
            "--image_dir",
            type=str,
            required=True,
            help="Image directory that stores images to manipulate (in PyTorch ImageFolder format).",
        )
        parser.add_argument(
            "--step",
            type=float,
            default=0.0,
            help="Step size taken in manipulation, depends on attribute.",
        )
        parser.add_argument(
            "--postprocess_steps",
            type=int,
            default=51,
            help="Noise added in postprocess with DDIM, default value is fine.",
        )
        parser.add_argument(
            "--postprocess_skip",
            type=int,
            default=50,
            help="Denoising skip size with DDIM for acceleration, default value is fine.",
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="Batch size for the D2C model."
        )
        parser.add_argument(
            "--save_location",
            type=str,
            default="results/default",
            help="Save edited and/or original images at this location.",
        )
        parser.add_argument(
            "--save_originals",
            action="store_true",
            help="If true, also save original images.",
        )
        args = parser.parse_args()

    if args.action == "sample_uncond":
        parser.add_argument(
            "--num_batches",
            type=int,
            default=1,
            help="Number of batches used in sampling, total number of images sampled is num_batches * batch_size.",
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="Batch size for the D2C model."
        )
        parser.add_argument(
            "--skip", type=int, default=1, help="Denoising skip step size with DDIM."
        )
        parser.add_argument(
            "--save_location",
            type=str,
            default="results/default",
            help="Save samples at this location.",
        )
        args = parser.parse_args()

    with open(os.path.join("configs", args.config + ".yml"), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def manipulation(args, config):
    print(f"Saving to {args.save_location} ...")
    if not os.path.isdir(os.path.join(args.save_location, "edited")):
        os.makedirs(os.path.join(args.save_location, "edited"))

    if args.save_originals and not os.path.isdir(
        os.path.join(args.save_location, "originals")
    ):
        print("Original images are also saved.")
        os.makedirs(os.path.join(args.save_location, "originals"))

    resolution = config.autoencoder.resolution
    val_dataset = tdatasets.ImageFolder(
        root=args.image_dir,
        transform=transforms.Compose(
            [transforms.Resize((resolution, resolution)), transforms.ToTensor()]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = D2C(args, config)
    state_dict = torch.load(args.d2c_path)
    model.load_state_dict(state_dict)
    model.eval()

    r_model = BoundaryInterpolationLayer(1, *model.latent_size).cuda()
    r_model.load_state_dict(torch.load(args.boundary_path))

    count = 0
    with torch.no_grad():
        for x, _ in tqdm.tqdm(val_loader):
            x = x.cuda()
            z = model.image_to_latent(x)
            z_ = model.manipulate_latent(z, r_model, args.step)
            z_ = model.postprocess_latent(
                z_, range(0, args.postprocess_steps, args.postprocess_skip)
            )
            x_ = model.latent_to_image(z_)
            for j in range(0, x_.size(0)):
                torchvision.utils.save_image(
                    x_[j : j + 1],
                    os.path.join(args.save_location, f"edited/{count}.png"),
                    padding=0,
                )
                if args.save_originals:
                    torchvision.utils.save_image(
                        x[j : j + 1],
                        os.path.join(args.save_location, f"originals/{count}.png"),
                        padding=0,
                    )
                count = count + 1


def sample_uncond(args, config):
    print(f"Saving to {args.save_location} ...")

    if not os.path.isdir(args.save_location):
        os.makedirs(args.save_location)

    model = D2C(args, config)
    state_dict = torch.load(args.d2c_path)
    model.load_state_dict(state_dict)
    model.eval()

    count = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, args.num_batches)):
            z = model.sample_latent(args.batch_size, args.skip)
            x = model.latent_to_image(z)
            for j in range(0, args.batch_size):
                torchvision.utils.save_image(
                    x[j : j + 1],
                    os.path.join(args.save_location, f"{count}.png"),
                    padding=0,
                )
                count = count + 1


if __name__ == "__main__":
    args, config = parse_args_and_config()
    dist.init_process_group(
        args.backend, init_method=args.init_method, world_size=1, rank=0
    )

    if args.action == "manipulation":
        manipulation(args, config)

    if args.action == "sample_uncond":
        sample_uncond(args, config)
