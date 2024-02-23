import argparse
import torch
from anime_ddpm.ddpm import Diffusion
from anime_ddpm.utils import save_images_normal
from models import UNet


def sample(weight_path, n=5, out='out.png'):
    model = UNet().to('cuda')
    weights = torch.load(weight_path)
    diffusion = Diffusion()

    model.load_state_dict(weights)

    sample = diffusion.sample(model, n)

    save_images_normal(sample, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to sample images using diffusion models.")
    parser.add_argument("weight_path", type=str, help="Path to the weights file.")
    parser.add_argument("-n", "--num_samples", type=int, default=5, help="Number of samples to generate (default: 5).")
    parser.add_argument("-o", "--output", type=str, default="out.png", help="Output file path for generated samples (default: out.png).")
    args = parser.parse_args()

    sample(args.weight_path, args.num_samples, args.output)
