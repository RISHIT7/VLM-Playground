class DINOEngine(nn.Module):
    def __init__(self, vit_backbone: nn.Module, out_dim: int = 4096, center_momentum: float = 0.9):
        """
        API Description:
        Initializes the Student network (ViT + DINO Projection Head) and the Teacher network 
        (deepcopy of Student). Freezes Teacher gradients. Registers buffer for the `center` tensor.
        """
        pass

    def forward_student(self, global_crops: torch.Tensor, local_crops: torch.Tensor) -> torch.Tensor:
        """
        API Description:
        Processes all crops through the Student network.
        
        Args:
            global_crops: Shape (B, 2, 3, 224, 224)
            local_crops: Shape (B, V, 3, 96, 96)
            
        Returns:
            Logits of shape ((2+V)*B, out_dim)
        """
        pass

    @torch.no_grad()
    def forward_teacher(self, global_crops: torch.Tensor) -> torch.Tensor:
        """
        API Description:
        Processes only the global crops through the Teacher network.
        
        Returns:
            Logits of shape (2*B, out_dim)
        """
        pass

    def compute_loss(self, student_out: torch.Tensor, teacher_out: torch.Tensor, student_temp: float, teacher_temp: float) -> torch.Tensor:
        """
        API Description:
        Applies temperature scaling. Centers the teacher output by subtracting `self.center`. 
        Computes cross-entropy between Student and Teacher, ensuring the Teacher does not 
        predict against the exact same crop view it generated.
        """
        pass

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        API Description:
        Exponential Moving Average (EMA) update of Teacher weights using Student weights.
        """
        pass

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor):
        """
        API Description:
        Exponential Moving Average (EMA) update of `self.center` using the current batch of Teacher outputs.
        """
        pass
        
    def get_optim_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        """
        API Description:
        Filters Student parameters for the optimizer, isolating LayerNorm and Bias terms for 0.0 weight decay.
        """
        pass