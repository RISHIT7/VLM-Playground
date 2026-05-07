import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class VAEEncoder(nn.Module):
    """Conv encoder: 3→32→64→128 channels with 3 downsample stages, outputs mean + logvar."""
    def __init__(self, base_ch=32, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down1 = nn.Sequential(
            ResBlock(base_ch, base_ch),
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            ResBlock(base_ch*2, base_ch*2),
            ResBlock(base_ch*2, base_ch*2),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4),
            ResBlock(base_ch*4, base_ch*4),
            nn.Conv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)
        )
        self.mid = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4),
            ResBlock(base_ch*4, base_ch*4)
        )
        self.conv_out = nn.Conv2d(base_ch*4, 2*z_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.conv_out(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar

class VAEDecoder(nn.Module):
    """Conv decoder: 128→64→32→3 channels with 3 upsample stages, Tanh output."""
    def __init__(self, base_ch=32, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(z_channels, base_ch*4, 3, padding=1)
        self.mid = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4),
            ResBlock(base_ch*4, base_ch*4)
        )
        # Upsample 16 -> 32 (128ch -> 64ch)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),   # 3x3 conv
            ResBlock(base_ch*4, base_ch*2),
            ResBlock(base_ch*2, base_ch*2)
        )
        # Upsample 32 -> 64 (64ch -> 32ch)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),   # 3x3 conv
            ResBlock(base_ch*2, base_ch),
            ResBlock(base_ch, base_ch)
        )
        # Upsample 64 -> 128 (32ch -> 32ch)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),       # 3x3 conv
            ResBlock(base_ch, base_ch),
            ResBlock(base_ch, base_ch)
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return self.conv_out(x)

class VAE(nn.Module):
    """Full VAE: encodes (128,128,3) → (16,16,4) latent, decodes back."""
    def __init__(self, z_channels=4):
        super().__init__()
        self.encoder = VAEEncoder(z_channels=z_channels)
        self.decoder = VAEDecoder(z_channels=z_channels)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar
