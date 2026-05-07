import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# CLIP Retrieval Evaluator
@torch.no_grad()
def evaluate_clip_retrieval(
    model,  # CLIPEngine
    dataloader: DataLoader,
    device: torch.device,
    k_values: Tuple[int, ...] = (1, 3),
) -> Dict[str, float]:
    """
    Computes cross-modal retrieval metrics on the provided validation DataLoader.
    The function embeds the entire validation set, then builds a similarity matrix and reports Recall@K for both directions:

    - Image-to-Text  (I→T): for each image, rank all captions and check if the correct one appears in the top-K.
    - Text-to-Image  (T→I): for each caption, rank all images.

    'Correct' is defined by matching raw_caption strings.

    Args:
        model:      CLIPEngine in eval mode.
        dataloader: val DataLoader for mode="clip".
        device:     Torch device.
        k_values:   Tuple of K values for Recall@K, default (1, 3).

    Returns:
        Dict with keys like "i2t_R@1", "i2t_R@3", "t2i_R@1", "t2i_R@3",
        "avg_R@1" (average of i2t and t2i R@1).
    """

    model.eval()
    all_img_feats = []
    all_txt_feats = []
    all_captions = []

    pbar = tqdm(dataloader, desc="CLIP Retrieval Evaluation", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        tokens = batch["tokens"].to(device)
        masks = batch["padding_mask"].to(device)
        captions = batch["raw_captions"]

        img_feat, txt_feat = model(images, tokens, masks)
        all_img_feats.append(img_feat.cpu())
        all_txt_feats.append(txt_feat.cpu())
        all_captions.extend(captions)

    # Stack into (N, D) matrices
    img_mat = torch.cat(all_img_feats, dim=0)  # (N, proj_dim)
    txt_mat = torch.cat(all_txt_feats, dim=0)  # (N, proj_dim)

    # Cosine similarity matrix (already L2-normalised inside the model)
    sim = img_mat @ txt_mat.T  # (N, N)
    N = sim.shape[0]

    # Build ground-truth binary target matrix: T[i,j] = 1 if caption_i == caption_j
    target = torch.tensor(
        [[all_captions[i] == all_captions[j] for j in range(N)] for i in range(N)],
        dtype=torch.bool,
    )  # (N, N)

    results: Dict[str, float] = {}

    def recall_at_k(sim_matrix: torch.Tensor, gt_matrix: torch.Tensor, k: int) -> float:
        """Fraction of queries where at least one positive is in top-K."""
        topk_indices = sim_matrix.topk(k, dim=1).indices  # (N, k)
        hit = torch.zeros(N, dtype=torch.bool)
        for i in range(N):
            hit[i] = gt_matrix[i][topk_indices[i]].any()
        return hit.float().mean().item()

    for k in k_values:
        results[f"i2t_R@{k}"] = recall_at_k(sim, target, k)
        results[f"t2i_R@{k}"] = recall_at_k(sim.T, target.T, k)

    # Averaged Recall@1 is the primary scalar used for best-checkpoint selection
    results["avg_R@1"] = (results["i2t_R@1"] + results["t2i_R@1"]) / 2.0
    results["avg_R@3"] = (results["i2t_R@3"] + results["t2i_R@3"]) / 2.0
    return results


# DINO Validation Loss Evaluator
@torch.no_grad()
def evaluate_dino_val_loss(
    model,  # DINOEngine
    dataloader: DataLoader,
    device: torch.device,
    student_temp: float,
    teacher_temp: float,
) -> Dict[str, float]:
    """
    Computes the DINO distillation cross-entropy on the validation set.

    Uses the *teacher* for global crops and the *student* for all crops,
    following the same logic as the training loop (but without gradient updates
    or EMA/center updates — we want a pure held-out signal).

    Args:
        model:        DINOEngine in eval mode.
        dataloader:   val DataLoader for mode="dino".
        device:       Torch device.
        student_temp: Temperature for student logits.
        teacher_temp: Current teacher temperature.

    Returns:
        Dict with key "val_loss".
    """

    model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="DINO Val Loss Eval", leave=False)
    for batch in pbar:
        global_crops = batch["global_crops"].to(device)
        local_crops = batch["local_crops"].to(device)

        teacher_out = model.forward_teacher(global_crops)
        student_out = model.forward_student(global_crops, local_crops)

        loss = model.compute_loss(student_out, teacher_out, student_temp, teacher_temp)
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, n_batches)
    return {"val_loss": avg_loss}


# Linear Probe Evaluator  (post training)
@torch.no_grad()
def extract_features(
    backbone,  # VisionTransformer (frozen)
    dataloader: DataLoader,
    device: torch.device,
    representation: str = "cls",  # "cls" | "gap"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs the frozen ViT over the linear-probe DataLoader and returns pooled
    features, count labels, and multi-hot color labels.

    Args:
        backbone:       Frozen VisionTransformer.
        dataloader:     DataLoader for mode="linear_probe".
        device:         Torch device.
        representation: Which pooling to use:
                          "cls" → [CLS] token output  (B, embed_dim)
                          "gap" → mean over patch tokens, excluding [CLS]

    Returns:
        Tuple of (features (N, D), count_labels (N,), color_labels (N, C)).
    """

    backbone.eval()
    all_feats = []
    all_counts = []
    all_colors = []

    pbar = tqdm(dataloader, desc=f"Extracting Features ({representation})", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        count_labels = batch["count_labels"]
        color_labels = (
            batch["color_label"] if "color_label" in batch else batch["color_labels"]
        )

        if representation == "cls":
            feats = backbone(images, return_patches=False)  # (B, D)
        elif representation == "gap":
            patches = backbone(images, return_patches=True)  # (B, N, D)
            feats = patches.mean(dim=1)  # (B, D)
        else:
            raise ValueError(
                f"representation must be 'cls' or 'gap', got '{representation}'"
            )

        all_feats.append(feats.cpu())
        all_counts.append(count_labels)
        all_colors.append(
            color_labels.cpu() if torch.is_tensor(color_labels) else color_labels
        )

    features = torch.cat(all_feats, dim=0)
    count_labels = torch.cat(all_counts, dim=0)
    color_labels = torch.cat(all_colors, dim=0)
    return features, count_labels, color_labels


def evaluate_linear_probe(
    probe,  # LinearProbe (count head or color head)
    features: torch.Tensor,
    count_labels: torch.Tensor,
    color_labels: torch.Tensor,
    device: torch.device,
    task: str = "count",  # "count" | "color"
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Runs inference of a LinearProbe on pre-extracted features and returns
    the relevant metric:
      - "count" task : classification accuracy
      - "color" task : macro F1 score

    Args:
        probe:        Trained LinearProbe (frozen during this call).
        features:     (N, D) tensor of pre-extracted embeddings.
        count_labels: (N,) integer labels for counting task.
        color_labels: (N, C) multi-hot float labels for color task.
        device:       Torch device.
        task:         "count" or "color".
        batch_size:   Mini-batch size for the inference pass.

    Returns:
        Dict with key "accuracy" (count) or "f1" (color).
    """

    probe.eval()
    N = features.shape[0]
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(range(0, N, batch_size), desc=f"Evaluating Probe ({task})", leave=False)
        for start in pbar:
            end = min(start + batch_size, N)
            feats_b = features[start:end].to(device)
            logits = probe(feats_b)

            if task == "count":
                preds = logits.argmax(dim=1).cpu()
            else:  # color — multi-label
                preds = (torch.sigmoid(logits) > 0.5).cpu().long()

            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0)

    if task == "count":
        accuracy = (all_preds == count_labels).float().mean().item()
        return {"accuracy": accuracy}
    else:
        y_true = color_labels.long().numpy()
        y_pred = all_preds.numpy()
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"f1": f1}


# VLM Evaluators
@torch.no_grad()
def evaluate_vlm_stage1(
    model, 
    dataloader: DataLoader, 
    device: torch.device, 
    tokenizer
) -> Dict[str, float]:
    """
    Evaluates the VLM in Stage 1 (Captioning) using BLEU score.
    """
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu not installed. BLEU score will not be computed.")
        return {"bleu": 0.0}

    model.eval()
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="VLM Stage 1 Eval (BLEU)", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        target_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device):
            outputs = model.module.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            ) if hasattr(model, "module") else model.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
        generated_ids = outputs
        pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        all_preds.extend([p.strip() for p in pred_texts])
        all_targets.extend([[t.strip()] for t in target_texts]) 

    if len(all_preds) == 0:
        return {"bleu": 0.0}

    bleu = sacrebleu.corpus_bleu(all_preds, all_targets)
    
    # Also compute exact match for captioning as requested by prompt "exact-match accuracy for stage-2 and stage-1"
    exact_matches = sum(1 for p, t in zip(all_preds, all_targets) if p == t[0])
    acc = exact_matches / len(all_preds)
    
    return {"bleu": bleu.score, "exact_match": acc}

@torch.no_grad()
def evaluate_vlm_stage2(
    model, 
    dataloader: DataLoader, 
    device: torch.device, 
    tokenizer
) -> Dict[str, float]:
    """
    Evaluates the VLM in Stage 2 (QA) using Exact-Match Accuracy.
    """
    model.eval()
    exact_matches = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="VLM Stage 2 Eval (Exact Match)", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        target_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device):
            outputs = model.module.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            ) if hasattr(model, "module") else model.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
        generated_ids = outputs
        pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, target in zip(pred_texts, target_texts):
            pred = pred.strip().lower()
            target = target.strip().lower()
            
            if pred == target:
                exact_matches += 1
            else:
                try:
                    target_ans = target.split("therefore, the answer is ")[1].replace(".", "").strip()
                    pred_ans = pred.split("therefore, the answer is ")[1].replace(".", "").strip()
                    if pred_ans == target_ans:
                        exact_matches += 1
                except IndexError:
                    pass
            total += 1

    acc = exact_matches / max(1, total)
    return {"exact_match": acc}
