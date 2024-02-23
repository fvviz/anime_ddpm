import math
import torch

class Diffusion:
  def __init__(self, noise_steps = 1000, img_size=64, noise_sched='linear',device='cuda'):
    self.noise_steps = noise_steps
    self.img_size = img_size
    self.device = device

    self.beta = self.prepare_noise_schedule(mode=noise_sched).to(device)
    self.alpha = 1 - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)
  def prepare_noise_schedule(self, mode):
    if mode=='linear':
        scale = 1000 / self.noise_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start, beta_end, self.noise_steps)
    else:
        print("calling cosine sched...")
        return self.betas_for_alpha_bar(lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    
  def betas_for_alpha_bar(self, alpha_bar,max_beta=0.999):
    betas = []
    for i in range(self.noise_steps):
        t1 = i / self.noise_steps
        t2 = (i + 1) / self.noise_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)
    

  def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

  def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

  def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    