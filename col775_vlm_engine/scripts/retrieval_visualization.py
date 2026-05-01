import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.clip_config import CLIPTrainConfig
from configs.data_config import get_config
from data.clevr_dataset import CLEVRCollateFn, CLEVRDataset
from data.text_tokenizer import CLEVRTokenizer
from data.transforms import CLIPTransforms
from models.clip_heads import CLIPEngine
from models.text_encoder import TextEncoder
from models.vit_backbone import VisionTransformer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_clip_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("cfg", {})
    tok_state = ckpt.get("tokenizer_state", {})

    cfg = CLIPTrainConfig()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    tokenizer = CLEVRTokenizer(max_seq_len=cfg.max_seq_len)
    if tok_state:
        tokenizer.word2idx = tok_state["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in tok_state["idx2word"].items()}
        tokenizer.vocab_size = tok_state["vocab_size"]
    else:
        env_cfg = get_config(cfg.env)
        train_json = os.path.join(env_cfg.base_dir_part_a, "train", "clevr_train_captions.json")
        tokenizer.build_vocab(train_json)

    vit = VisionTransformer(
        img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth, num_heads=cfg.vit_num_heads, mlp_dim=cfg.vit_mlp_dim,
    )
    text_enc = TextEncoder(
        vocab_size=tokenizer.vocab_size, embed_dim=cfg.embed_dim,
        depth=cfg.text_depth, num_heads=cfg.text_num_heads,
        mlp_dim=cfg.text_mlp_dim, max_len=cfg.max_seq_len,
    )
    model = CLIPEngine(
        vit_backbone=vit, text_encoder=text_enc,
        embed_dim=cfg.embed_dim, proj_dim=cfg.proj_dim,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, cfg


@torch.no_grad()
def embed_validation_set(model, dataloader, device):
    all_img_feats, all_txt_feats, all_captions = [], [], []
    
    pbar = tqdm(dataloader, desc="Embedding Val Set", leave=False)
    for batch in pbar:
        images = batch["images"].to(device)
        tokens = batch["tokens"].to(device)
        masks = batch["padding_mask"].to(device)

        img_feat, txt_feat = model(images, tokens, masks)
        all_img_feats.append(img_feat.cpu())
        all_txt_feats.append(txt_feat.cpu())
        all_captions.extend(batch["raw_captions"])

    img_mat = torch.cat(all_img_feats, dim=0)
    txt_mat = torch.cat(all_txt_feats, dim=0)
    return img_mat, txt_mat, all_captions


def compute_recall(sim, all_captions, k_values=(1, 3)):
    N = sim.shape[0]
    target = torch.tensor(
        [[all_captions[i] == all_captions[j] for j in range(N)] for i in range(N)],
        dtype=torch.bool,
    )
    results = {}
    for k in k_values:
        topk_i2t = sim.topk(k, dim=1).indices
        topk_t2i = sim.T.topk(k, dim=1).indices
        hit_i2t = torch.tensor([target[i][topk_i2t[i]].any() for i in range(N)])
        hit_t2i = torch.tensor([target.T[i][topk_t2i[i]].any() for i in range(N)])
        results[f"i2t_R@{k}"] = hit_i2t.float().mean().item()
        results[f"t2i_R@{k}"] = hit_t2i.float().mean().item()
    return results


def show_i2t_examples(sim, all_captions, image_paths, indices, top_k, output_dir):
    os.makedirs(os.path.join(output_dir, "i2t"), exist_ok=True)

    for idx in indices:
        scores, topk_idx = sim[idx].topk(top_k)
        gt_caption = all_captions[idx]
        retrieved = [(all_captions[j], scores[rank].item()) for rank, j in enumerate(topk_idx)]

        fig, axes = plt.subplots(1, 1 + top_k, figsize=(4 * (1 + top_k), 4))
        img = Image.open(image_paths[idx]).convert("RGB")
        axes[0].imshow(img)
        axes[0].set_title(f"Query Image #{idx}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        for rank, (cap, score) in enumerate(retrieved):
            is_hit = (cap == gt_caption)
            color = "#2ecc71" if is_hit else "#e74c3c"
            axes[rank + 1].text(
                0.5, 0.5, f"#{rank+1}\n\n{cap}\n\nsim={score:.4f}",
                ha="center", va="center", fontsize=9, wrap=True,
                transform=axes[rank + 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15),
            )
            axes[rank + 1].set_title("✓ Match" if is_hit else "✗ No match", fontsize=9, color=color)
            axes[rank + 1].axis("off")

        fig.suptitle(f"I→T | GT: \"{gt_caption}\"", fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "i2t", f"i2t_query_{idx}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def show_t2i_examples(sim, all_captions, image_paths, indices, top_k, output_dir):
    os.makedirs(os.path.join(output_dir, "t2i"), exist_ok=True)
    sim_t2i = sim.T

    for idx in indices:
        scores, topk_idx = sim_t2i[idx].topk(top_k)
        gt_caption = all_captions[idx]

        fig, axes = plt.subplots(1, 1 + top_k, figsize=(4 * (1 + top_k), 4))

        axes[0].text(
            0.5, 0.5, f"Query Caption #{idx}:\n\n\"{gt_caption}\"",
            ha="center", va="center", fontsize=10, wrap=True,
            transform=axes[0].transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#3498db", alpha=0.15),
        )
        axes[0].set_title("Query Text", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        for rank, j in enumerate(topk_idx):
            is_hit = (all_captions[j.item()] == gt_caption)
            img = Image.open(image_paths[j.item()]).convert("RGB")
            axes[rank + 1].imshow(img)
            color = "#2ecc71" if is_hit else "#e74c3c"
            axes[rank + 1].set_title(
                f"#{rank+1} sim={scores[rank]:.4f} {'✓' if is_hit else '✗'}",
                fontsize=9, color=color,
            )
            axes[rank + 1].axis("off")

        fig.suptitle(f"T→I Retrieval", fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "t2i", f"t2i_query_{idx}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def select_example_indices(sim, all_captions, n_success=3, n_failure=3):
    N = sim.shape[0]
    top1_idx = sim.topk(1, dim=1).indices.squeeze(1)

    success, failure = [], []
    for i in range(N):
        if all_captions[i] == all_captions[top1_idx[i].item()]:
            if len(success) < n_success:
                success.append(i)
        else:
            if len(failure) < n_failure:
                failure.append(i)
        if len(success) >= n_success and len(failure) >= n_failure:
            break
    return success + failure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-ckpt", type=str, required=True)
    p.add_argument("--env", type=str, default="local_omen")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/retrieval")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-examples", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading CLIP model ...")
    model, tokenizer, cfg = load_clip_model(args.clip_ckpt, device)

    env_cfg = get_config(args.env)
    transform = CLIPTransforms(image_size=cfg.img_size)
    collate = CLEVRCollateFn(mode="clip")

    val_ds = CLEVRDataset(env_cfg, mode="clip", split="val", transform=transform, tokenizer=tokenizer)

    dl_kwargs = {}
    if env_cfg.num_workers > 0 and env_cfg.prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = env_cfg.prefetch_factor

    val_dl = DataLoader(
        val_ds, batch_size=env_cfg.batch_size, shuffle=False,
        num_workers=env_cfg.num_workers, pin_memory=env_cfg.pin_memory,
        drop_last=False, collate_fn=collate, **dl_kwargs,
    )

    logger.info("Embedding validation set ...")
    img_mat, txt_mat, all_captions = embed_validation_set(model, val_dl, device)
    sim = img_mat @ txt_mat.T

    logger.info("Computing recall metrics ...")
    metrics = compute_recall(sim, all_captions)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics → {metrics_path}")

    image_dir = os.path.join(env_cfg.base_dir_part_a, "val", "images")
    image_paths = []
    for ann in val_ds.annotations:
        fname = ann.get("image_filename", ann.get("image", ann.get("filename", "")))
        image_paths.append(os.path.join(image_dir, fname))

    logger.info("Selecting example queries ...")
    i2t_indices = select_example_indices(sim, all_captions, args.n_examples, args.n_examples)
    t2i_indices = select_example_indices(sim.T, all_captions, args.n_examples, args.n_examples)

    logger.info(f"Generating I→T examples ({len(i2t_indices)} queries) ...")
    show_i2t_examples(sim, all_captions, image_paths, i2t_indices, args.top_k, args.output_dir)

    logger.info(f"Generating T→I examples ({len(t2i_indices)} queries) ...")
    show_t2i_examples(sim, all_captions, image_paths, t2i_indices, args.top_k, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
