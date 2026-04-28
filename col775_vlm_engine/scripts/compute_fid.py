"""
Compute FID scores for VAE reconstructions and LDM-generated images (Part C evaluation).

Usage:
    python scripts/compute_fid.py \
        --vae-ckpt checkpoints/vae/checkpoint_best.pt \
        --ldm-ckpt checkpoints/ldm/checkpoint_best.pt \
        --output-dir outputs/fid
"""
