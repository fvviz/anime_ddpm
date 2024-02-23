import wandb
import argparse
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim

from anime_ddpm.utils import get_loader, setup_logging, save_images
from anime_ddpm.models import UNet
from anime_ddpm.ddpm import Diffusion


def train_fn(run_name, dataloader, epochs, LOAD_MODEL=False, noise_sched='linear', weight_path=None, completed_epochs=0, wandb_logs=False)
    l = len(dataloader)
    LOAD_MODEL= True

    device='cuda'
    
    
    if wandb_logs:
        wandb.init(name=run_name, 
                   project='anime_ddpm_2',
                   notes='anime_ddpm_proj', 
                   entity='user',
                   settings=wandb.Settings(start_method="fork"))

    setup_logging(run_name)
    device = 'cuda'

    model = UNet().to(device)

    if LOAD_MODEL:
      print("loading weights..")
      weights = torch.load(weight_path)
      model.load_state_dict(weights)
      print("loaded:",  weight_path)

    optimizer = optim.AdamW(model.parameters())
    scaler  = torch.cuda.amp.GradScaler()
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_sched=noise_sched)



    for epoch in range(epochs):
      logging.info(f'Starting epoch {epoch}')
      print("Resuming from:", epoch+completed_epochs)
      bar = tqdm(dataloader)
      for i, (images, _) in enumerate(bar):
        images = images.to('cuda')
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        with torch.cuda.amp.autocast():
          predicted_noise = model(x_t, t)
          loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bar.set_postfix(loss=loss.item())



      print("Saving weights...")
      torch.save(model.state_dict(), os.path.join('models', run_name, 'ckpt.pt'))
      print("Saved weights as ckpt.pt")


      print("Sampling..")
      sampled= diffusion.sample(model, n=4)
      print("sampled images")
        
      save_images(sampled, os.path.join('results', run_name, f'epoch{epoch+completed_epochs}.png'))
    
      if wandb_logs:
          images = wandb.Image(sampled, caption="Generated:")

          wandb.log({"Train Loss": loss.item(),                 
                   "images:": images})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train anime DDPM")
    parser.add_argument("--run_name", type=str, help="name of the run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run model")
    parser.add_argument("--data", type=str, help="The folder where the anime faces data is stored")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for loading the data")
    args = parser.parse_args()

    dataloader = get_loader(args.data, batch_size=args.batch_size)
    train_fn(args.run_name, dataloader, args.epochs)

