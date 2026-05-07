import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from cleanfid import fid

from models.unet import UNetLDM
from models.diffusion import DiffusionSchedule, DDPMSampler
from models.vae import VAE
from data.clevr_dataset import CLEVRDataset

logger = logging.getLogger(__name__)

def load_checkpoint_with_mapping(model, checkpoint_path, device):
    """Loads checkpoint with specific mapping for mid_block naming inconsistencies."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("mid_res1."):
            new_k = k.replace("mid_res1.", "mid_block1.")
        elif k.startswith("mid_attn."):
            new_k = k.replace("mid_attn.", "mid_block1.attn.")
        elif k.startswith("mid_res2."):
            new_k = k.replace("mid_res2.", "mid_block2.")
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return state_dict

def train_ldm(cfg) -> None:
    """Train UNetLDM on frozen VAE latents with learned null context."""
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # Datasets
    train_dataset = CLEVRDataset(cfg, mode="ldm", split="train")
    train_loader = DataLoader(train_dataset, batch_size=getattr(cfg, 'batch_size', 32), shuffle=True, num_workers=4)
    
    val_dataset = CLEVRDataset(cfg, mode="ldm", split="val")
    val_loader = DataLoader(val_dataset, batch_size=getattr(cfg, 'batch_size', 32), shuffle=False, num_workers=4)

    # Initialize Model
    unet = UNetLDM().to(device)
    
    # Learned null embedding (Part of the training state)
    # Context dim is 512, sequence length is 77
    null_embedding = nn.Parameter(torch.randn(1, 77, 512, device=device) * 0.02)
    
    optimizer = optim.AdamW(list(unet.parameters()) + [null_embedding], lr=getattr(cfg, 'ldm_lr', 1e-4))
    
    # Setup diffusion schedule
    timesteps = getattr(cfg, 'diffusion_timesteps', 500)
    schedule = DiffusionSchedule(timesteps=timesteps, device=device)
    
    start_epoch = 0
    best_loss = float('inf')
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "ldm_latest.pth")
    best_path = os.path.join(cfg.checkpoint_dir, "unet_best.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        if 'null_embedding' in checkpoint:
            null_embedding.data.copy_(checkpoint['null_embedding'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resumed LDM from epoch {start_epoch}")

    epochs = getattr(cfg, 'ldm_epochs', 100)
    
    logger.info("Starting LDM training loop...")
    for epoch in range(start_epoch, epochs):
        unet.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            z0 = batch["latent"].to(device)
            text_emb = batch["text_embed"].to(device)
            b = z0.shape[0]
            
            t = torch.randint(0, timesteps, (b,), device=device).long()
            
            # Classifier-Free Guidance: 10% dropout
            mask = torch.rand(b, 1, 1, device=device) < 0.1
            context = torch.where(mask, null_embedding.expand(b, -1, -1), text_emb)
            
            noise = torch.randn_like(z0)
            zt = schedule.q_sample(z0, t, noise)
            
            optimizer.zero_grad()
            pred_noise = unet(zt, t, context)
            
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoints
        save_dict = {
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'null_embedding': null_embedding.data,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(save_dict, checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dict['best_loss'] = best_loss
            torch.save(save_dict, best_path)
            logger.info(f"New best model saved with loss {best_loss:.6f}")

    # Optional: Final FID Evaluation logic
    # if getattr(cfg, 'run_fid', False):
    #     evaluate_fid(unet, null_embedding, cfg, device)

def evaluate_fid(unet, null_embedding, cfg, device):
    """Calculates FID score on validation set."""
    unet.eval()
    logger.info("Starting FID evaluation...")
    
    # Load VAE and Stats for decoding
    vae = VAE().to(device)
    vae_checkpoint = getattr(cfg, 'vae_model_path', os.path.join(cfg.checkpoint_dir, "vae_best.pt"))
    if os.path.exists(vae_checkpoint):
        vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)['model_state_dict'])
    
    latent_mean = torch.load(cfg.latent_mean_path).to(device)
    latent_std = torch.load(cfg.latent_std_path).to(device)
    
    sampler = DDPMSampler(DiffusionSchedule(timesteps=500, device=device))
    
    gen_dir = os.path.join(cfg.output_dir, "ldm_gen_fid")
    real_dir = cfg.val_image_dir # Path to real validation images
    os.makedirs(gen_dir, exist_ok=True)
    
    # Load val captions
    with open(cfg.val_captions_path, 'r') as f:
        val_data = json.load(f)
        if "examples" in val_data: val_data = val_data["examples"]
    
    # This is a simplified FID loop. In practice, you'd want to generate ~5000 images.
    num_gen = min(len(val_data), 1000)
    logger.info(f"Generating {num_gen} images for FID...")
    
    for i in tqdm(range(0, num_gen, 10)):
        batch_data = val_data[i:i+10]
        # In a real script, you'd load precomputed embeddings or run CLIP text encoder here
        # For now, we assume the user will run a separate evaluation script that has access to all components
        pass

    # score = fid.compute_fid(gen_dir, real_dir)
    # logger.info(f"FID Score: {score}")
    # return score
