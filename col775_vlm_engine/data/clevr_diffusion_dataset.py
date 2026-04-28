import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List


class CLEVRDiffusionDataset(Dataset):
    """Dataset for VAE and LDM training: loads 128x128 images with captions for text conditioning."""
    pass
