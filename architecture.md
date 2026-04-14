# Model Architecture Specifications

This document catalogs the architectural choices established across the vision and text pathways for implementing CLIP and DINO models, reflecting the specific constraints laid out in the COL775 Assignment 2.

## 1. Transformer Architecture Fundamentals
Instead of using opaque high-level PyTorch abstractions like `nn.Transformer` or `TransformerLayer`, the core backbone relies exclusively on low-level neural primitives:
- **Prerequisites:** Architected exclusively using `nn.MultiheadAttention`, `nn.Linear`, `nn.LayerNorm`, and `nn.GELU`.
- **Pre-Norm Flow:** Follows modern Transformer design where input signals are explicitly normalized *before* crossing Attention and MLP functional blocks: e.g., `x = x + Module(LayerNorm(x))`.

## 2. Vision Transformer (ViT) Image Encoder
Leveraged interchangeably to power the shared visual sequence tracking for both CLIP contrastive alignment and DINO self-distillation architectures.
- **Topology:** Standard ViT setup with deep computational layout mapping a `depth=12` transformer layers block size.
- **Attention Protocol:** Features `num_heads=6` parallel multi-head attention spaces scaling globally over patches.
- **Dimensions:** Utilizes a standard representation footprint carrying `embed_dim=384` context properties, feeding an extrapolated hidden capacity mapping `mlp_dim=1536` across linear feed-forward layers.
- **Patch Embedding Conversion:** Projects `224×224` physical pixel inputs via structurally efficient discrete 2D strided convolutional kernels (`patch_size=16x16`). This slices parameters mathematically mapping cleanly over `196` non-overlapping uniform grids.
- **Positional Encoding (PE):** Establishes integrated 1D **learnable positional embeddings** initiated against truncated standard normal parameters (`std=0.02`). Furthermore, implements a localized, 2D bicubic interpolation strategy dynamically recalculating grid spacing distances locally in response to arbitrary local/global multi-crop variations triggered computationally within DINO.
- **Representation Tokens:** Assigns individual learnable `[CLS]` zero tensors prepended identically to respective image stacks, forming the globally aggregate anchor dimension defining categorical summaries independent of strict geometry embeddings.

## 3. Lightweight Text Encoder
Maintains dedicated transformer tracking responsible for CLIP alignment via synthetic, attribute-packed CLEVR captions mapping count/color parameters.
- **Topology:** Streamlines textual alignment matching constrained scale pipelines enforcing `depth=6` transformer sequences.
- **Attention & Dimension Profiles:** Perfectly mirrors matching cross-modality scaling protocols targeting analogous parameters configurations: `num_heads=6`, `embed_dim=384`, and an MLP feature scaling capacity reaching `mlp_dim=1536`.
- **Positional Encoding (PE):** Foregoes dynamic variance backprop updates in favor of mathematically constant trigonometric matrices. Fuses absolute alternating sine-cosine embeddings mapped fundamentally reflecting properties modeled natively by Vaswani et al. Fixed buffer registry guarantees these constraints execute immutably per parameter mapping.
- **Representation Outputs:** Extracts contextual sequences structurally aggregating the terminal sentence element identifying the `[EOS]` non-padding endpoint across masked iterations tracking textual data sets implicitly context-length separated.

## 4. Projection Head Paradigms (Unimplemented / Deferred Backbones)
Decoupled logic abstracts embedding dimension sizes inherently. Both Transformer encodings functionally abstain assigning precise terminal projections inside their module layers directly:
- Shared cross-modal outputs matching text bindings are processed downstream projecting CLIP alignments safely mapping exact target constraint geometries reaching exactly `512` dimensional feature space parameters.
- Alternatively, instances initialized routing configurations into DINO model variants construct multi-crop dense feature parameters directly over large `4096` self-distillation representations globally tracking teacher-student bounds.
