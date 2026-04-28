import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List


class CLEVRCaptionDataset(Dataset):
    """Stage-1 dataset: image + caption pairs from Clevr_official for alignment training."""
    pass


class CLEVRQADataset(Dataset):
    """Stage-2 dataset: image + question + chain-of-thought explanation + answer for fine-tuning."""
    pass
