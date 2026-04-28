import logging

logger = logging.getLogger(__name__)


def train_vae(cfg) -> None:
    """Train convolutional VAE to compress (128,128,3) → (16,16,4) latent."""
    raise NotImplementedError
