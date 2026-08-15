# COL775 - Assignment 2

This repository contains the codebase, reports, and visualizations for Assignment 2 of **COL775**. The project focuses on implementing and analyzing Deep Learning architectures, specifically centered around Vision-Language Models (VLMs), Vision Transformers (ViT), Text Encoders, and Variational Autoencoders (VAEs).

## 👥 Contributors
*   **Rishit Jakharia** ([@RISHIT7](https://github.com/RISHIT7))
*   **Karan Deo Burnwal** ([@logxdx](https://github.com/logxdx))
*   **Keerthana Jatoth** ()

---

## ✨ Key Features
Based on the project's commit history and architecture, the engine supports:
*   **Vision-Language Model (VLM) Engine:** Custom implementation of modular `TransformerBlock` and `TextEncoder` components.
*   **Vision Transformers (ViT):** Relocated and integrated ViT architecture for feature extraction.
*   **Variational Autoencoders (VAE):** VAE classes with an updated, robust training pipeline.
*   **Feature Visualizations:** Output generation including t-SNE visualizations to compare representations from CLIP and DINO (Student & Teacher models).
*   **Experiment Tracking:** Integrated `WandbLogger` utility for tracking training runs using Weights & Biases.

---

## 📂 Repository Structure

The workspace is organized into modular components and analysis scripts:

*   **`col775_vlm_engine/`**: The core source code containing the VLM architecture, VAE classes, and the primary training pipeline.
*   **`TeamName_A2/`**: Contains specific assignment scripts separated by parts (e.g., Part A, Part C).
*   **`data/`**: Data loading utilities and dataset processing scripts.
*   **`outputs/tsne/`**: Directory for generated outputs, primarily containing t-SNE visualizations for CLIP and DINO.
*   **`tests/`**: Testing scripts and logging utilities (e.g., `WandbLogger`).
*   **`*.pdf`**: Project documentation, including the main assignment instructions (`a2.pdf`), the final report (`COL775_A2_report (1).pdf`), and Notion page exports.
*   **`architecture.md`**: Detailed documentation regarding the modular Transformer and TextEncoder architectural choices.

---

## 🛠️ Environment & Setup

This project uses Python (configured via `.python-version`) and relies on `pyproject.toml` and `uv.lock` for strict dependency management.

### Prerequisites
Make sure you have [uv](https://github.com/astral-sh/uv) installed to manage the dependencies efficiently.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/RISHIT7/Assignment-2_COL775.git](https://github.com/RISHIT7/Assignment-2_COL775.git)
   cd Assignment-2_COL775
