"""
Generate images from text prompts using trained VAE + LDM (Part C).

Usage:
    python scripts/generate_samples.py \
        --vae-ckpt checkpoints/vae/checkpoint_best.pt \
        --ldm-ckpt checkpoints/ldm/checkpoint_best.pt \
        --prompts "A scene with 3 red cubes" "Two blue spheres" \
        --output-dir outputs/generated
"""
