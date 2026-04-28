import torch
import torch.nn as nn


class VLMProjector(nn.Module):
    """2-layer MLP with reverse-bottleneck that maps vision patch tokens to LLM hidden dim."""
    pass


class VLMModel(nn.Module):
    """Full VLM wrapping frozen vision encoder + projector + LLM (Qwen3-4B)."""
    pass
