import torch
import torch.nn as nn

class ResBlock(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, base_ch=32, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(3, base_ch, 3, padding=1)
        self.down1 = nn.Sequential(
            ResBlock(base_ch, base_ch), ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            ResBlock(base_ch*2, base_ch*2), ResBlock(base_ch*2, base_ch*2),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4), ResBlock(base_ch*4, base_ch*4),
            nn.Conv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1)
        )
        self.mid = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4), ResBlock(base_ch*4, base_ch*4)
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

class Decoder(nn.Module):
    def __init__(self, base_ch=32, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(z_channels, base_ch*4, 3, padding=1)
        self.mid = nn.Sequential(
            ResBlock(base_ch*4, base_ch*4), ResBlock(base_ch*4, base_ch*4)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1),
            ResBlock(base_ch*4, base_ch*2),
            ResBlock(base_ch*2, base_ch*2)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            ResBlock(base_ch*2, base_ch),
            ResBlock(base_ch, base_ch)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
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
    def __init__(self, z_channels=4):
        super().__init__()
        self.encoder = Encoder(z_channels=z_channels)
        self.decoder = Decoder(z_channels=z_channels)
    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        recon = self.decoder(z)
        return recon, mean, logvar

