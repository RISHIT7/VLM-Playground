import torch
import torch.nn as nn
import math
from einops import rearrange

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        if context is None:
            context = x
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b l (h d) -> b h l d', h=h)
        v = rearrange(v, 'b l (h d) -> b h l d', h=h)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads)  # self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads)  # cross-attention
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.transformer = BasicTransformerBlock(channels, context_dim, heads)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in

class ResBlockWithTime(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, context_dim=None, use_attention=False, heads=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None
        self.attn = SpatialTransformer(out_ch, context_dim, heads) if use_attention else None

    def forward(self, x, time_emb=None, context=None):
        h = self.silu(self.norm1(x))
        h = self.conv1(h)
        if self.time_mlp is not None and time_emb is not None:
            time_emb_out = self.time_mlp(self.silu(time_emb))
            h = h + time_emb_out[:, :, None, None]
        h = self.silu(self.norm2(h))
        h = self.conv2(h)
        out = h + self.shortcut(x)
        if self.attn is not None:
            out = self.attn(out, context=context)
        return out

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_ch=128, time_dim=256, context_dim=512, heads=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Down stages
        self.down1_blocks = nn.Sequential(
            ResBlockWithTime(base_ch, base_ch, time_dim, context_dim, use_attention=False),
            ResBlockWithTime(base_ch, base_ch, time_dim, context_dim, use_attention=False)
        )
        self.down1_down = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)

        self.down2_blocks = nn.Sequential(
            ResBlockWithTime(base_ch * 2, base_ch * 2, time_dim, context_dim, use_attention=False),
            ResBlockWithTime(base_ch * 2, base_ch * 2, time_dim, context_dim, use_attention=False)
        )
        self.down2_down = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)

        self.down3_blocks = nn.Sequential(
            ResBlockWithTime(base_ch * 4, base_ch * 4, time_dim, context_dim, use_attention=True, heads=heads),
            ResBlockWithTime(base_ch * 4, base_ch * 4, time_dim, context_dim, use_attention=True, heads=heads)
        )
        self.down3_down = nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)

        # Mid
        self.mid_blocks = nn.Sequential(
            ResBlockWithTime(base_ch * 8, base_ch * 8, time_dim, context_dim, use_attention=True, heads=heads),
            ResBlockWithTime(base_ch * 8, base_ch * 8, time_dim, context_dim, use_attention=True, heads=heads)
        )

        # Up stages
        self.up3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1)
        )
        self.up3_blocks = nn.Sequential(
            ResBlockWithTime(base_ch * 4, base_ch * 4, time_dim, context_dim, use_attention=True, heads=heads),
            ResBlockWithTime(base_ch * 4, base_ch * 4, time_dim, context_dim, use_attention=True, heads=heads)
        )

        self.up2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        )
        self.up2_blocks = nn.Sequential(
            ResBlockWithTime(base_ch * 2, base_ch * 2, time_dim, context_dim, use_attention=False),
            ResBlockWithTime(base_ch * 2, base_ch * 2, time_dim, context_dim, use_attention=False)
        )

        self.up1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        )
        self.up1_blocks = nn.Sequential(
            ResBlockWithTime(base_ch, base_ch, time_dim, context_dim, use_attention=False),
            ResBlockWithTime(base_ch, base_ch, time_dim, context_dim, use_attention=False)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_channels, 3, padding=1)
        )

    def forward(self, z, t, context):
        t_emb = self.time_mlp(t)
        x = self.conv_in(z)

        # Down
        x = self.down1_blocks[0](x, t_emb, context)
        x = self.down1_blocks[1](x, t_emb, context)
        x = self.down1_down(x)

        x = self.down2_blocks[0](x, t_emb, context)
        x = self.down2_blocks[1](x, t_emb, context)
        x = self.down2_down(x)

        x = self.down3_blocks[0](x, t_emb, context)
        x = self.down3_blocks[1](x, t_emb, context)
        x = self.down3_down(x)

        # Mid
        x = self.mid_blocks[0](x, t_emb, context)
        x = self.mid_blocks[1](x, t_emb, context)

        # Up
        x = self.up3_up(x)
        x = self.up3_blocks[0](x, t_emb, context)
        x = self.up3_blocks[1](x, t_emb, context)

        x = self.up2_up(x)
        x = self.up2_blocks[0](x, t_emb, context)
        x = self.up2_blocks[1](x, t_emb, context)

        x = self.up1_up(x)
        x = self.up1_blocks[0](x, t_emb, context)
        x = self.up1_blocks[1](x, t_emb, context)

        return self.conv_out(x)
