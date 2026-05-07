import torch
import math

class DiffusionSchedule:
    """Noise schedule (cosine or linear) + forward diffusion q(z_t | z_0)."""
    def __init__(self, timesteps=1000, s=0.008, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class DDPMSampler:
    """DDPM reverse sampling with optional Classifier-Free Guidance."""
    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    @torch.no_grad()
    def generate(self, model, text_embeds, null_context, w=3.0, shape=(4, 16, 16)):
        device = text_embeds.device
        b = text_embeds.shape[0]
        img_shape = (b, *shape)
        img = torch.randn(img_shape, device=device)
        
        context = torch.cat([text_embeds, null_context.repeat(b, 1, 1)])
        
        for i in reversed(range(0, self.schedule.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_double = torch.cat([t, t])
            
            latent_double = torch.cat([img, img])
            pred_noise_double = model(latent_double, t_double, context)
            
            pred_noise_cond, pred_noise_uncond = pred_noise_double.chunk(2)
            pred_noise = pred_noise_uncond + w * (pred_noise_cond - pred_noise_uncond)
            
            alpha = self.schedule.alphas[t][:, None, None, None]
            alpha_hat = self.schedule.alphas_cumprod[t][:, None, None, None]
            beta = self.schedule.betas[t][:, None, None, None]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0.0
                
            img = 1 / torch.sqrt(alpha) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
            
        return img
