import torch
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# VLM Evaluators
@torch.no_grad()
def evaluate_vlm_stage1(
    model, dataloader: DataLoader, device: torch.device, tokenizer
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
            outputs = (
                model.module.generate(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                if hasattr(model, "module")
                else model.generate(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
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
    model, dataloader: DataLoader, device: torch.device, tokenizer
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
            outputs = (
                model.module.generate(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                if hasattr(model, "module")
                else model.generate(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
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
                    target_ans = (
                        target.split("therefore, the answer is ")[1]
                        .replace(".", "")
                        .strip()
                    )
                    pred_ans = (
                        pred.split("therefore, the answer is ")[1]
                        .replace(".", "")
                        .strip()
                    )
                    if pred_ans == target_ans:
                        exact_matches += 1
                except IndexError:
                    pass
            total += 1

    acc = exact_matches / max(1, total)
    return {"exact_match": acc}
