import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DINOEngine(nn.Module):
    def __init__(self, vit_backbone: nn.Module, student_head: nn.Module, teacher_head: nn.Module, out_dim: int = 4096, center_momentum: float = 0.9):
        """
        API Description:
        Initializes the Student network (ViT + DINO Projection Head) and the Teacher network 
        (deepcopy of Student). Freezes Teacher gradients. Registers buffer for the `center` tensor.

        Args:
            vit_backbone: The ViT backbone to use.
            student_head: The DINO projection head for the student.
            teacher_head: The DINO projection head for the teacher.
            out_dim: The output dimension of the projection heads.
            center_momentum: The momentum for the teacher network.
        """
        super().__init__()

        self.student_network = nn.ModuleList([vit_backbone, student_head])
        self.teacher_network = nn.ModuleList([copy.deepcopy(vit_backbone), teacher_head])
        self.teacher_network.requires_grad_(False)
        
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum
        self.out_dim = out_dim

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
        global_crops = global_crops.transpose(0,1).flatten(0, 1) # (B, 2, 3, 224, 224) -> (2*B, 3, 224, 224)
        local_crops = local_crops.transpose(0,1).flatten(0, 1) # (B, V, 3, 96, 96) -> (V*B, 3, 96, 96)
        global_latent_embeddings = self.student_network[0](global_crops) # (2*B, 384)
        local_latent_embeddings = self.student_network[0](local_crops) # (V*B, 384)

        concat_latent_embeddings = torch.cat([global_latent_embeddings, local_latent_embeddings], dim=0) # ((2+V)*B, 384)

        logits = self.student_network[1](concat_latent_embeddings) # ((2+V)*B, out_dim)

        return logits

    @torch.no_grad()
    def forward_teacher(self, global_crops: torch.Tensor) -> torch.Tensor:
        """
        API Description:
        Processes only the global crops through the Teacher network.
        
        Returns:
            Logits of shape (2*B, out_dim)
        """
        global_crops = global_crops.transpose(0,1).flatten(0, 1) # (B, 2, 3, 224, 224) -> (2*B, 3, 224, 224)
        latent_embeddings = self.teacher_network[0](global_crops) # (2*B, 384)
        logits = self.teacher_network[1](latent_embeddings) # (2*B, out_dim)
        return logits

    def compute_loss(self, student_out: torch.Tensor, teacher_out: torch.Tensor, student_temp: float, teacher_temp: float) -> torch.Tensor:
        """
        API Description:
        Applies temperature scaling. Centers the teacher output by subtracting `self.center`. 
        Computes cross-entropy between Student and Teacher, ensuring the Teacher does not 
        predict against the exact same crop view it generated.
        """
        teacher_out = (teacher_out - self.center) / teacher_temp
        teacher_out = F.softmax(teacher_out, dim = -1)

        student_out = student_out / student_temp
        student_out = F.log_softmax(student_out, dim = -1)

        teacher_chunks = teacher_out.chunk(2)

        n_student_chunks = student_out.shape[0] // (teacher_out.shape[0]//2)
        student_chunks = student_out.chunk(n_student_chunks)
        

        total_loss = 0.0
        n_loss_term = 0

        for t_idx, t_view in enumerate(teacher_chunks):
            for s_idx, s_view in enumerate(student_chunks):
                if t_idx == s_idx:
                    continue
                
                loss = torch.sum(-t_view * s_view, dim = -1).mean()
                
                total_loss += loss
                n_loss_term += 1
        
        return total_loss / n_loss_term

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        API Description:
        Exponential Moving Average (EMA) update of Teacher weights using Student weights.
        """
        # using special torch .mul_() and .add_() for in-place operations
        for param_t, param_s in zip(self.teacher_network.parameters(), self.student_network.parameters()):
            param_t.data.mul_(momentum)
            param_t.data.add_(param_s.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor):
        """
        API Description:
        Exponential Moving Average (EMA) update of `self.center` using the current batch of Teacher outputs.
        """
        batch_mean = teacher_out.mean(dim=0, keepdim=True)
        self.center.data.mul_(self.center_momentum).add_(batch_mean, alpha=1 - self.center_momentum)
        
        
    def get_optim_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        """
        API Description:
        Filters Student parameters for the optimizer, isolating LayerNorm and Bias terms for 0.0 weight decay.
        """
        decay = []
        no_decay = []

        for name, param in self.student_network.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]