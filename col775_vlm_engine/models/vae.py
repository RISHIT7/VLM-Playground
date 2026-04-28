import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""
    pass


class VAEEncoder(nn.Module):
    """Conv encoder: 3→32→64→128 channels with 3 downsample stages, outputs mean + logvar."""
    pass


class VAEDecoder(nn.Module):
    """Conv decoder: 128→64→32→3 channels with 3 upsample stages, Tanh output."""
    pass


class VAE(nn.Module):
    """Full VAE: encodes (128,128,3) → (16,16,4) latent, decodes back."""
    pass
