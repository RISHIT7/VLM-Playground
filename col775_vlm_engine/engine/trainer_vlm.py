import logging

logger = logging.getLogger(__name__)


def train_vlm_stage1(cfg) -> None:
    """Stage-1: Freeze vision encoder + LLM, train only the MLP projector on captioning."""
    raise NotImplementedError


def train_vlm_stage2(cfg) -> None:
    """Stage-2: Fine-tune projector + LLM (LoRA) on chain-of-thought QA."""
    raise NotImplementedError
