from torchvision import transforms
from typing import Dict, List
import torch

class CLIPTransforms:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, image):
        return self.transform(image)

class DINOMultiCropTransforms:
    def __init__(self, local_crops_number: int = 8):
        self.local_crops_number = local_crops_number
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image) -> Dict[str, List[torch.Tensor]]:
        global_crops = [self.global_transform(image) for _ in range(2)]
        local_crops = [self.local_transform(image) for _ in range(self.local_crops_number)]
        return {"global_crops": global_crops, "local_crops": local_crops}

class LinearProbeTransforms:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, image):
        return self.transform(image)