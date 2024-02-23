import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
    super().__init__()
    self.residual = residual
    if not mid_channels:
      mid_channels = out_channels

    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(1, mid_channels),
        nn.GELU(),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(1, out_channels)
    )

  def forward(self, x):
      if self.residual:
        return F.gelu(x + self.double_conv(x))
      else:
        return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim=256):
    super().__init__()
    self.maxpool_conv_down = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, in_channels, residual=True),
        DoubleConv(in_channels, out_channels)
    )

    self.emb_layer = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_dim, out_channels)
    )

  def forward(self, x, t):
      x = self.maxpool_conv_down(x)
      emb_i = self.emb_layer(t)
      emb = emb_i[:,:, None, None].repeat(1 , 1, x.shape[-2], x.shape[-1])
      return x + emb


class Up(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim=256):
    super().__init__()

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Sequential(
        DoubleConv(in_channels, in_channels, residual=True),
        DoubleConv(in_channels, out_channels, in_channels//2)
    )

    self.emb_layer = nn.Sequential(
        nn.SiLU(),
        nn.Linear(
            emb_dim,out_channels)
    )

  def forward(self, x, skip_x, t):
      x = self.up(x)
      x = torch.cat([x, skip_x],dim=1)
      x = self.conv(x)
      emb_i = self.emb_layer(t)
      emb = emb_i[:,:, None, None].repeat(1 , 1, x.shape[-2], x.shape[-1])
      return x + emb

class SelfAttention(nn.Module):
  def __init__(self, channels, size):
    super().__init__()
    self.channels= channels
    self.size = size
    self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
    self.ln = nn.LayerNorm([channels])
    self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

  def forward(self, x):
    x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
    x_ln = self.ln(x)

    attn_value, _ = self.mha(x_ln, x_ln, x_ln)
    attn_value += x
    attn_value += self.ff_self(attn_value)
    return attn_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)