import logging

logger = logging.getLogger(__name__)


def train_ldm(cfg) -> None:
    """Train conditional U-Net on frozen VAE latents with CLIP text conditioning."""
    raise NotImplementedError
