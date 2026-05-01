import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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


def load_clip_model(ckpt_path: str, device: torch.device):
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
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        mlp_dim=cfg.vit_mlp_dim,
    )
    text_enc = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.text_depth,
        num_heads=cfg.text_num_heads,
        mlp_dim=cfg.text_mlp_dim,
        max_len=cfg.max_seq_len,
    )
    model = CLIPEngine(
        vit_backbone=vit,
        text_encoder=text_enc,
        embed_dim=cfg.embed_dim,
        proj_dim=cfg.proj_dim,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, cfg


def build_val_dataloader(env_cfg, tokenizer, cfg):
    transform = CLIPTransforms(image_size=cfg.img_size)
    collate = CLEVRCollateFn(mode="clip")
    val_ds = CLEVRDataset(env_cfg, mode="clip", split="val", transform=transform, tokenizer=tokenizer)

    dl_kwargs: Dict[str, Any] = {}
    if env_cfg.num_workers > 0 and env_cfg.prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = env_cfg.prefetch_factor

    val_dl = DataLoader(
        val_ds,
        batch_size=env_cfg.batch_size,
        shuffle=False,
        num_workers=env_cfg.num_workers,
        pin_memory=env_cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
        **dl_kwargs,
    )
    return val_ds, val_dl


def collect_image_paths(dataset: CLEVRDataset) -> List[str]:
    image_paths: List[str] = []
    for ann in dataset.annotations:
        fname = ann.get("image_filename", ann.get("image", ann.get("filename", "")))
        image_paths.append(os.path.join(dataset.image_dir, fname))
    return image_paths


@torch.no_grad()
def embed_validation_set(model, dataloader, device: torch.device):
    all_img_feats, all_txt_feats, all_captions = [], [], []

    pbar = tqdm(dataloader, desc="Embedding validation set", leave=False)
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


def retrieve_topk(
    query_mat: torch.Tensor,
    bank_mat: torch.Tensor,
    top_k: int,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k scores and indices for every query without materialising the full similarity matrix."""

    top_k = min(top_k, bank_mat.shape[0])
    all_scores: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []

    for start in range(0, query_mat.shape[0], chunk_size):
        end = min(start + chunk_size, query_mat.shape[0])
        scores = query_mat[start:end] @ bank_mat.T
        top_scores, top_indices = scores.topk(top_k, dim=1)
        all_scores.append(top_scores.cpu())
        all_indices.append(top_indices.cpu())

    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)


def compute_recall_from_topk(
    query_captions: Sequence[str],
    retrieved_indices: torch.Tensor,
    all_captions: Sequence[str],
    k_values: Tuple[int, ...] = (1, 3),
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for k in k_values:
        hits = []
        topk = retrieved_indices[:, : min(k, retrieved_indices.shape[1])]
        for i in range(topk.shape[0]):
            query_caption = query_captions[i]
            retrieved_captions = [all_captions[j] for j in topk[i].tolist()]
            hits.append(any(cap == query_caption for cap in retrieved_captions))
        results[f"R@{k}"] = float(torch.tensor(hits, dtype=torch.float32).mean().item())
    return results


def make_examples(
    retrieved_indices: torch.Tensor,
    query_captions: Sequence[str],
    all_captions: Sequence[str],
    n_success: int,
    n_failure: int,
) -> Tuple[List[int], List[int]]:
    top1 = retrieved_indices[:, 0]
    success, failure = [], []

    for i in range(len(query_captions)):
        top1_idx = int(top1[i].item())
        is_hit = all_captions[top1_idx] == query_captions[i]
        if is_hit and len(success) < n_success:
            success.append(i)
        elif not is_hit and len(failure) < n_failure:
            failure.append(i)
        if len(success) >= n_success and len(failure) >= n_failure:
            break

    return success, failure


def _save_figure(fig: Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def show_i2t_examples(
    retrieved_scores: torch.Tensor,
    retrieved_indices: torch.Tensor,
    all_captions: Sequence[str],
    image_paths: Sequence[str],
    indices: Sequence[int],
    top_k: int,
    output_dir: str,
) -> List[Dict[str, Any]]:
    out_dir = os.path.join(output_dir, "i2t")
    os.makedirs(out_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for idx in indices:
        k = min(top_k, retrieved_indices.shape[1])
        top_indices = retrieved_indices[idx, :k]
        top_scores = retrieved_scores[idx, :k]
        query_caption = all_captions[idx]

        retrieved = []
        fig, axes = plt.subplots(1, 1 + k, figsize=(4.2 * (1 + k), 4.0))

        img = Image.open(image_paths[idx]).convert("RGB")
        axes[0].imshow(img)
        axes[0].set_title(f"Query Image #{idx}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        for rank, (j, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            j = int(j)
            cap = all_captions[j]
            hit = cap == query_caption
            retrieved.append({"rank": rank + 1, "index": j, "caption": cap, "score": float(score), "hit": hit})

            color = "#2ecc71" if hit else "#e74c3c"
            axes[rank + 1].text(
                0.5,
                0.5,
                f"#{rank + 1}\n\n{cap}\n\nsim={score:.4f}",
                ha="center",
                va="center",
                fontsize=9,
                wrap=True,
                transform=axes[rank + 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15),
            )
            axes[rank + 1].set_title("✓ Match" if hit else "✗ No match", fontsize=9, color=color)
            axes[rank + 1].axis("off")

        fig.suptitle(f'I→T | Query caption: "{query_caption}"', fontsize=11, fontweight="bold", y=0.99)
        fig_path = os.path.join(out_dir, f"i2t_query_{idx}.png")
        _save_figure(fig, fig_path)

        records.append(
            {
                "query_index": idx,
                "query_caption": query_caption,
                "query_image": image_paths[idx],
                "figure": os.path.relpath(fig_path, output_dir),
                "retrieved": retrieved,
            }
        )

    return records


def show_t2i_examples(
    retrieved_scores: torch.Tensor,
    retrieved_indices: torch.Tensor,
    all_captions: Sequence[str],
    image_paths: Sequence[str],
    indices: Sequence[int],
    top_k: int,
    output_dir: str,
) -> List[Dict[str, Any]]:
    out_dir = os.path.join(output_dir, "t2i")
    os.makedirs(out_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for idx in indices:
        k = min(top_k, retrieved_indices.shape[1])
        top_indices = retrieved_indices[idx, :k]
        top_scores = retrieved_scores[idx, :k]
        query_caption = all_captions[idx]

        retrieved = []
        fig, axes = plt.subplots(1, 1 + k, figsize=(4.2 * (1 + k), 4.0))

        axes[0].text(
            0.5,
            0.5,
            f'Query Caption #{idx}:\n\n"{query_caption}"',
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            transform=axes[0].transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#3498db", alpha=0.15),
        )
        axes[0].set_title("Query Text", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        for rank, (j, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            j = int(j)
            cap = all_captions[j]
            hit = cap == query_caption
            retrieved.append({"rank": rank + 1, "index": j, "caption": cap, "score": float(score), "hit": hit})

            img = Image.open(image_paths[j]).convert("RGB")
            axes[rank + 1].imshow(img)
            color = "#2ecc71" if hit else "#e74c3c"
            axes[rank + 1].set_title(
                f"#{rank + 1} sim={score:.4f} {'✓' if hit else '✗'}",
                fontsize=9,
                color=color,
            )
            axes[rank + 1].axis("off")

        fig.suptitle("T→I Retrieval", fontsize=11, fontweight="bold", y=0.99)
        fig_path = os.path.join(out_dir, f"t2i_query_{idx}.png")
        _save_figure(fig, fig_path)

        records.append(
            {
                "query_index": idx,
                "query_caption": query_caption,
                "query_image": image_paths[idx],
                "figure": os.path.relpath(fig_path, output_dir),
                "retrieved": retrieved,
            }
        )

    return records


def write_report(
    output_dir: str,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    i2t_examples: List[Dict[str, Any]],
    t2i_examples: List[Dict[str, Any]],
) -> str:
    report_path = os.path.join(output_dir, "retrieval_report.md")

    lines: List[str] = []
    lines.append("# CLIP Cross-Modal Retrieval Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Checkpoint: {args.clip_ckpt}")
    lines.append(f"- Environment: {args.env}")
    lines.append(f"- Device: {args.device}")
    lines.append(f"- Top-K shown: {args.top_k}")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Direction | R@1 | R@3 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Image→Text | {metrics['i2t_R@1']:.4f} | {metrics['i2t_R@3']:.4f} |")
    lines.append(f"| Text→Image | {metrics['t2i_R@1']:.4f} | {metrics['t2i_R@3']:.4f} |")
    lines.append("")
    lines.append("### Notes")
    lines.append("- Success cases are queries where the exact validation caption appears among the top retrieved neighbours.")
    lines.append("- Failure cases usually indicate visually similar scenes with a different count, color, or relation attribute.")
    lines.append("")

    def add_examples(title: str, examples: List[Dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for ex in examples:
            lines.append(f"### Query #{ex['query_index']}")
            lines.append(f"- Query caption: {ex['query_caption']}")
            lines.append(f"- Query image: {ex['query_image']}")
            lines.append(f"- Figure: {ex['figure']}")
            lines.append("")
            for r in ex["retrieved"]:
                status = "hit" if r["hit"] else "miss"
                lines.append(
                    f"  {r['rank']}. [{status}] score={r['score']:.4f} | idx={r['index']} | {r['caption']}"
                )
            lines.append("")

    add_examples("Image→Text examples", i2t_examples)
    add_examples("Text→Image examples", t2i_examples)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal retrieval evaluation for CLIP embeddings.")
    parser.add_argument("--clip-ckpt", type=str, default="checkpoints/clip/checkpoint_best.pt", help="Path to a trained CLIP checkpoint.")
    parser.add_argument("--env", type=str, default="local_omen", help="Environment config name.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--output-dir", type=str, default="outputs/cross_modal_retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbours to display.")
    parser.add_argument("--n-examples", type=int, default=3, help="Number of success/failure examples per direction.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Query chunk size for retrieval.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading CLIP model ...")
    model, tokenizer, cfg = load_clip_model(args.clip_ckpt, device)

    env_cfg = get_config(args.env)
    val_ds, val_dl = build_val_dataloader(env_cfg, tokenizer, cfg)
    image_paths = collect_image_paths(val_ds)

    logger.info("Embedding validation split ...")
    img_mat, txt_mat, all_captions = embed_validation_set(model, val_dl, device)

    retrieval_k = max(args.top_k, 3)
    logger.info("Retrieving nearest neighbours ...")
    i2t_scores, i2t_indices = retrieve_topk(img_mat, txt_mat, retrieval_k, chunk_size=args.chunk_size)
    t2i_scores, t2i_indices = retrieve_topk(txt_mat, img_mat, retrieval_k, chunk_size=args.chunk_size)

    metrics = {
        "i2t_R@1": 0.0,
        "i2t_R@3": 0.0,
        "t2i_R@1": 0.0,
        "t2i_R@3": 0.0,
    }
    i2t_recalls = compute_recall_from_topk(all_captions, i2t_indices, all_captions, k_values=(1, 3))
    t2i_recalls = compute_recall_from_topk(all_captions, t2i_indices, all_captions, k_values=(1, 3))
    metrics["i2t_R@1"] = i2t_recalls["R@1"]
    metrics["i2t_R@3"] = i2t_recalls["R@3"]
    metrics["t2i_R@1"] = t2i_recalls["R@1"]
    metrics["t2i_R@3"] = t2i_recalls["R@3"]
    metrics["avg_R@1"] = (metrics["i2t_R@1"] + metrics["t2i_R@1"]) / 2.0
    metrics["avg_R@3"] = (metrics["i2t_R@3"] + metrics["t2i_R@3"]) / 2.0

    logger.info("Metrics:")
    logger.info("  Image→Text  R@1=%.4f  R@3=%.4f", metrics["i2t_R@1"], metrics["i2t_R@3"])
    logger.info("  Text→Image  R@1=%.4f  R@3=%.4f", metrics["t2i_R@1"], metrics["t2i_R@3"])

    metrics_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_path)

    i2t_success, i2t_failure = make_examples(i2t_indices, all_captions, all_captions, args.n_examples, args.n_examples)
    t2i_success, t2i_failure = make_examples(t2i_indices, all_captions, all_captions, args.n_examples, args.n_examples)

    i2t_example_ids = i2t_success + i2t_failure
    t2i_example_ids = t2i_success + t2i_failure

    logger.info("Saving I→T examples ...")
    i2t_examples = show_i2t_examples(
        i2t_scores,
        i2t_indices,
        all_captions,
        image_paths,
        i2t_example_ids,
        args.top_k,
        args.output_dir,
    )

    logger.info("Saving T→I examples ...")
    t2i_examples = show_t2i_examples(
        t2i_scores,
        t2i_indices,
        all_captions,
        image_paths,
        t2i_example_ids,
        args.top_k,
        args.output_dir,
    )

    examples_path = os.path.join(args.output_dir, "retrieval_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump({"i2t": i2t_examples, "t2i": t2i_examples}, f, indent=2)
    logger.info("Saved example metadata -> %s", examples_path)

    report_path = write_report(args.output_dir, args, metrics, i2t_examples, t2i_examples)
    logger.info("Saved markdown report -> %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()