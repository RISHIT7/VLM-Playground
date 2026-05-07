import torch
import math

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionSchedule:
    def __init__(self, timesteps=500, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_t * noise

class DDPMSampler:
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    @torch.no_grad()
    def generate(self, model, text_emb, null_embedding, guidance_scale=4.0, latent_mean=None, latent_std=None, vae_decoder=None):
        device = text_emb.device
        b = text_emb.shape[0]
        
        # Null context
        null_ctx = null_embedding.expand(b, -1, -1)
        
        # Start from pure noise
        z = torch.randn(b, 4, 16, 16, device=device)
        
        # DDPM sampling with CFG
        for t in reversed(range(self.schedule.timesteps)):
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Predict noise with text context
            pred_text = model(z, t_tensor, text_emb)
            # Predict noise with null context
            pred_null = model(z, t_tensor, null_ctx)
            # Apply CFG
            pred = (1 + guidance_scale) * pred_text - guidance_scale * pred_null
            
            alpha_t = self.schedule.alphas[t]
            alpha_cumprod_t = self.schedule.alphas_cumprod[t]
            beta_t = self.schedule.betas[t]
            
            noise = torch.randn_like(z) if t > 0 else 0
            z = 1 / torch.sqrt(alpha_t) * (z - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * pred) + torch.sqrt(beta_t) * noise
        
        if vae_decoder is not None and latent_mean is not None and latent_std is not None:
            # Denormalize and decode
            z = z * latent_std.to(device) + latent_mean.to(device)
            img = vae_decoder(z)
            img = (img + 1) / 2
            return torch.clamp(img, 0, 1)
        
        return z

