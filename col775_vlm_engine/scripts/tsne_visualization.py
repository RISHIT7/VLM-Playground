import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.data_config import get_config
from data.clevr_dataset import CLEVRCollateFn, CLEVRDataset
from data.transforms import LinearProbeTransforms
from engine.evaluator import extract_features
from engine.trainer_linear_probe import _build_backbone
from configs.linear_probe_config import LinearProbeConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_backbone(backbone_name, clip_ckpt, dino_ckpt, env, device):
    cfg = LinearProbeConfig(
        backbone=backbone_name,
        clip_checkpoint=clip_ckpt,
        dino_checkpoint=dino_ckpt,
        env=env,
    )
    return _build_backbone(cfg, device)


def get_features(backbone, env, device, representation="cls"):
    env_cfg = get_config(env)
    transform = LinearProbeTransforms(image_size=224)
    collate = CLEVRCollateFn(mode="linear_probe")

    train_ds = CLEVRDataset(env_cfg, mode="linear_probe", split="train", transform=transform)
    dl_kwargs = {}
    if env_cfg.num_workers > 0 and env_cfg.prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = env_cfg.prefetch_factor

    train_dl = DataLoader(
        train_ds, batch_size=256, shuffle=False,
        num_workers=env_cfg.num_workers, pin_memory=env_cfg.pin_memory,
        drop_last=False, collate_fn=collate, **dl_kwargs,
    )

    feats, counts, _ = extract_features(backbone, train_dl, device, representation=representation)
    return feats.numpy(), counts.numpy()


def run_tsne(features, perplexity=30, random_state=42):
    logger.info(f"Running t-SNE on {features.shape[0]} samples (dim={features.shape[1]}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                n_iter=1000, init="pca", learning_rate="auto")
    embeddings_2d = tsne.fit_transform(features)
    logger.info(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
    return embeddings_2d


def plot_tsne(embeddings_2d, count_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_counts = np.sort(np.unique(count_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_counts))

    for i, c in enumerate(unique_counts):
        mask = count_labels == c
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[cmap(i)], label=f"{c} objects", s=3, alpha=0.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=4, loc="best", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-ckpt", type=str, required=True)
    p.add_argument("--dino-ckpt", type=str, required=True)
    p.add_argument("--env", type=str, default="local_omen")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/tsne")
    p.add_argument("--representation", type=str, default="cls", choices=["cls", "gap"])
    p.add_argument("--perplexity", type=float, default=30)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    backbone_names = ["clip", "dino", "dino_teacher"]
    display_names = ["CLIP", "DINO Student", "DINO Teacher"]

    all_embeddings = {}
    count_labels = None

    for bname, dname in zip(backbone_names, display_names):
        logger.info(f"\n{'='*50}\nProcessing {dname}\n{'='*50}")
        backbone = load_backbone(bname, args.clip_ckpt, args.dino_ckpt, args.env, device)
        feats, counts = get_features(backbone, args.env, device, representation=args.representation)
        count_labels = counts

        emb_2d = run_tsne(feats, perplexity=args.perplexity)
        all_embeddings[dname] = emb_2d

        save_path = os.path.join(args.output_dir, f"tsne_{bname}_{args.representation}.png")
        plot_tsne(emb_2d, counts, f"t-SNE — {dname} ({args.representation.upper()})", save_path)

        del backbone
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    unique_counts = np.sort(np.unique(count_labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_counts))

    for ax, (dname, emb_2d) in zip(axes, all_embeddings.items()):
        for i, c in enumerate(unique_counts):
            mask = count_labels == c
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=[cmap(i)], label=f"{c}", s=2, alpha=0.4,
            )
        ax.set_title(dname, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{l} objects" for l in labels],
               loc="lower center", ncol=len(unique_counts), markerscale=4, fontsize=9)
    fig.suptitle(f"t-SNE Visualization — {args.representation.upper()} Embeddings (color = object count)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(args.output_dir, f"tsne_combined_{args.representation}.png")
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined figure → {combined_path}")


if __name__ == "__main__":
    main()
