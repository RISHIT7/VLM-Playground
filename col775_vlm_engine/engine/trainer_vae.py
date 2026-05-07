import logging
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vae import VAE
from data.clevr_dataset import CLEVRDataset

logger = logging.getLogger(__name__)

def train_vae(cfg) -> None:
    """Train convolutional VAE to compress (128,128,3) → (16,16,4) latent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset & DataLoader setup
    # Note: Ensure the transforms inside CLEVRDataset scale images to [-1, 1]
    train_dataset = CLEVRDataset(cfg, mode="vae", split="train")
    train_loader = DataLoader(train_dataset, batch_size=getattr(cfg, 'batch_size', 32), shuffle=True, num_workers=4)
    
    # Initialize VAE
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    start_epoch = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "vae_latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed VAE from epoch {start_epoch}")

    epochs = getattr(cfg, 'vae_epochs', 100)
    kl_weight = getattr(cfg, 'vae_kl_weight', 0.00025)
    
    logger.info("Starting VAE training loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch["image"].to(device)
            
            optimizer.zero_grad()
            recon, mean, logvar = model(x)
            
            recon_loss = F.mse_loss(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.shape[0]  # mean over batch
            
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({'loss': loss.item(), 'recon': recon_loss.item(), 'kl': kl_loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished with avg loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
