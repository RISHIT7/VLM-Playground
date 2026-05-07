import os
import logging
from typing import Dict, Any, List, Optional
import torch
import wandb
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

class WandbLogger:
    def __init__(
        self,
        project_name: str,
        config: Dict[str, Any],
        run_name: str,
        run_id: Optional[str] = None,
        rank: int = 0,
        offline: bool = False
    ) -> None:
        """
        Sets up the W&B run. We only initialize on Rank 0 to avoid DDP headaches.
        If the internet is being flaky (common on HPC nodes), we catch the error 
        so the training doesn't just die.

        Args:
            project_name (str): The name of the W&B project.
            config (dict): A dictionary containing training configuration and hyperparameters.
            run_name (str): The display name for this run in the W&B UI.
            run_id (Optional[str]): The unique W&B run ID to resume a crashed run. Defaults to None.
            rank (int): The DDP rank of the current process. Defaults to 0.
            offline (bool): If True, forces W&B to operate in offline mode. Defaults to False.
        """
        self.rank = rank
        self.is_active = False
        
        if self.rank != 0:
            return

        if offline:
            os.environ["WANDB_MODE"] = "offline"
            logger.info("WandbLogger initialized in OFFLINE mode as requested.")

        try:
            self.run: Optional[Run] = wandb.init(
                project=project_name,
                name=run_name,
                id=run_id,
                config=config,
                resume="allow"
            )
            self.is_active = True
            logger.info(f"Successfully initialized Weights & Biases: {run_name}")
        except Exception as e:
            self.is_active = False
            logger.warning(
                f"Failed to initialize Weights & Biases. Running without it to prevent crash. "
                f"Error: {str(e)}"
            )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Sends your scalar stats (like loss, LR, or accuracy) over to W&B.

        Args:
            metrics (Dict[str, float]): The metrics to log.
            step (int): The current training step/iteration.
        """
        if self.rank != 0 or not self.is_active:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics at step {step}: {e}")

    def log_clip_alignment(
        self,
        images: torch.Tensor,
        raw_captions: List[str],
        step: int,
        max_samples: int = 4
    ) -> None:
        """
        Visual check for CLIP: dumps images and their labels to W&B after 
        un-normalizing them back to a human-viewable range.

        Args:
            images (torch.Tensor): A batch of input images of shape (B, 3, 224, 224).
                Expected to be normalized with ImageNet mean and std.
            raw_captions (List[str]): The raw text captions corresponding to the images.
            step (int): The current training step/iteration.
            max_samples (int): The maximum number of image-caption pairs to log. Defaults to 4.
        """
        if self.rank != 0 or not self.is_active:
            return

        try:
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(3, 1, 1)

            n_samples = min(max_samples, images.size(0))
            
            wandb_images = []
            for i in range(n_samples):
                img = images[i] * std + mean
                img = torch.clamp(img, 0, 1)
                
                wandb_img = wandb.Image(img, caption=raw_captions[i])
                wandb_images.append(wandb_img)
            
            wandb.log({"CLIP Alignment": wandb_images}, step=step)
            
        except Exception as e:
            logger.error(f"Failed to log CLIP alignment images at step {step}: {e}")

    def log_dino_crops(
        self,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        step: int,
        max_samples: int = 2
    ) -> None:
        """
        Displays global views side-by-side with their corresponding local views to verify
        the self-distillation data pipeline.

        Args:
            global_crops (torch.Tensor): Batch of global crops, shape (B, 2, 3, 224, 224).
            local_crops (torch.Tensor): Batch of local crops, shape (B, V, 3, 96, 96), 
                where V is the number of local views.
            step (int): The current training step/iteration.
            max_samples (int): Max number of samples in the batch to log. Defaults to 2.
        """
        if self.rank != 0 or not self.is_active:
            return

        try:
            mean = torch.tensor([0.485, 0.456, 0.406], device=global_crops.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=global_crops.device).view(3, 1, 1)

            n_samples = min(max_samples, global_crops.size(0))
            n_locals = local_crops.size(1)

            wandb_images = []
            
            for i in range(n_samples):
                g_view1 = torch.clamp(global_crops[i, 0] * std + mean, 0, 1)
                g_view2 = torch.clamp(global_crops[i, 1] * std + mean, 0, 1)
                
                wandb_images.append(wandb.Image(g_view1, caption=f"Sample {i}: Global View 1 (224x224)"))
                wandb_images.append(wandb.Image(g_view2, caption=f"Sample {i}: Global View 2 (224x224)"))
                
                for v in range(n_locals):
                    l_view = torch.clamp(local_crops[i, v] * std + mean, 0, 1)
                    wandb_images.append(wandb.Image(l_view, caption=f"Sample {i}: Local View {v+1} (96x96)"))

            wandb.log({"DINO Multi-Crop Grid": wandb_images}, step=step)

        except Exception as e:
            logger.error(f"Failed to log DINO crops at step {step}: {e}")

    def log_model_artifact(self, model_path: str, artifact_name: str, type: str = "model") -> None:
        """
        Logs saved PyTorch model weights (.pt or .pth) to W&B Artifacts for
        version control and reproducibility.

        Args:
            model_path (str): The local filesystem path to the saved model weights.
            artifact_name (str): The logical name of the artifact (e.g., 'clip-vit-base', 'dino-resnet50').
            type (str): The classification type of the artifact. Defaults to "model".
        """
        if self.rank != 0 or not self.is_active:
            return

        try:
            if not os.path.exists(model_path):
                logger.error(f"Artifact path does not exist: {model_path}")
                return

            artifact = wandb.Artifact(name=artifact_name, type=type)
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            logger.info(f"Successfully logged model artifact: {artifact_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model artifact '{artifact_name}': {e}")

    def finish(self) -> None:
        """
        Wraps up the session and ensures all pending data is synced to the cloud.
        """
        if self.rank != 0 or not self.is_active:
            return

        try:
            wandb.finish()
            self.is_active = False
            logger.info("Weights & Biases logging session finished gracefully.")
        except Exception as e:
            logger.error(f"Failed to cleanly finish Weights & Biases run: {e}")