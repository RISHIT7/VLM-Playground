import logging
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import ConditionalUNet
from models.diffusion import DiffusionSchedule
from data.clevr_dataset import CLEVRDataset

logger = logging.getLogger(__name__)

def train_ldm(cfg) -> None:
    """Train conditional U-Net on frozen VAE latents with CLIP text conditioning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # The dataset should yield 'latent' and 'text_embed' from precomputed arrays
    train_dataset = CLEVRDataset(cfg, mode="ldm", split="train")
    train_loader = DataLoader(train_dataset, batch_size=getattr(cfg, 'batch_size', 32), shuffle=True, num_workers=4)
    
    # Initialize LDM UNet
    model = ConditionalUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(cfg, 'ldm_lr', 1e-4))
    
    # Setup diffusion schedule
    timesteps = getattr(cfg, 'diffusion_timesteps', 1000)
    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    
    start_epoch = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "ldm_latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed LDM from epoch {start_epoch}")

    epochs = getattr(cfg, 'ldm_epochs', 100)
    
    # Null context for CFG (using a placeholder token for 'empty' caption)
    # The actual empty token logic depends on how the tokenizer handles ""
    # In the notebook, it might just use precomputed null_context
    # Here we assume it's part of the precomputation or passed via dataset/cfg
    
    logger.info("Starting LDM training loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Latents are already normalized from VAE (or need scaling)
            z_start = batch["latent"].to(device)
            context = batch["text_embed"].to(device)
            b = z_start.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (b,), device=device).long()
            
            # CFG: Drop context 10% of the time (replace with null_context)
            if torch.rand(1).item() < 0.1:
                # Assuming batch contains 'null_embed' or we can zero it
                context = batch["null_embed"].to(device) if "null_embed" in batch else torch.zeros_like(context)

            # Add noise
            noise = torch.randn_like(z_start)
            z_noisy = schedule.q_sample(z_start, t, noise)
            
            optimizer.zero_grad()
            
            # Predict noise
            pred_noise = model(z_noisy, t, context)
            
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished with avg loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
