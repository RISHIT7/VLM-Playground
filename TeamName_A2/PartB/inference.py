import argparse
import json
import os
import re
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models.vit_backbone import VisionTransformer
from models.vlm import VLMModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

IMG_PLACEHOLDER_ID = -200
NUM_IMAGE_TOKENS = 196


def build_inference_inputs(questions_batch, tokenizer, device):

    img_pad = torch.full((NUM_IMAGE_TOKENS,), IMG_PLACEHOLDER_ID, dtype=torch.long)
    newline_token = tokenizer(
        "\n", return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze(0)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    input_ids_list = []
    for q_text in questions_batch:
        prompt_str = (
            "Analyse the input image and output a json like this:\n"
            "```json\n"
            "{\n"
            '    "reasoning": "The image contains a large metallic object. I compare its shape with the other visible objects and do not find another object of the same shape.",\n'
            '    "answer": "no"\n'
            "}\n"
            "```\n"
            f"Question: {q_text}\nAnswer:"
        )
        prompt_tokens = tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0)
        context_ids = torch.cat([img_pad, newline_token, prompt_tokens])
        input_ids_list.append(context_ids)

    input_ids_batched = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(ids) for ids in input_ids_list],
        batch_first=True,
        padding_value=0,
    )

    return input_ids_batched.to(device), attention_mask.to(device)


# Answer parsing
def parse_generated_text(text: str):

    text = text.strip()

    print(f"\n\nAnswer: {text}\n\n")

    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                json_str = text

        parsed = json.loads(json_str)
        if "reasoning" in parsed and "answer" in parsed:
            return parsed["reasoning"], parsed["answer"]
    except Exception:
        pass

    pattern = r"[Tt]herefore,?\s*the\s+answer\s+is\s+"
    parts = re.split(pattern, text, maxsplit=1)

    if len(parts) == 2:
        reasoning = parts[0].strip()
        answer = parts[1].rstrip(".").strip()
        return reasoning, answer

    return text, text


# Model loading
BASE_LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"


def _locate_artifacts(model_dir: str):

    proj_path = f"{model_dir}/vlm_stage2.pt"
    vit_path = f"{model_dir}/clip.pt"

    if vit_path is None:
        raise FileNotFoundError(f"Could not find ViT checkpoint (.pt) in {model_dir}")
    if proj_path is None:
        raise FileNotFoundError(
            f"Could not find Projector checkpoint (.pt) in {model_dir}"
        )

    return str(vit_path), str(proj_path), str(model_dir)


def _patch_adapter_config(lora_dir: str):

    cfg_path = Path(lora_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    changed = False

    if cfg.get("base_model_name_or_path", "").startswith("unsloth/"):
        cfg["base_model_name_or_path"] = BASE_LLM_ID
        changed = True

    if "auto_mapping" in cfg:
        am = cfg["auto_mapping"]
        if isinstance(am, dict) and am.get("unsloth_fixed"):
            am.pop("unsloth_fixed", None)
            changed = True

    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info("  Patched adapter_config.json for PEFT compatibility")


def load_model(model_dir: str, device: torch.device, dtype: torch.dtype):
    """
    Loads the full VLM pipeline from the model directory.
    """
    model_dir = str(Path(model_dir).resolve())
    logger.info(f"Loading model components from: {model_dir}")

    # 1. Locate files
    vit_ckpt_path, proj_ckpt_path, lora_dir = _locate_artifacts(model_dir)
    logger.info(f"  ViT checkpoint : {vit_ckpt_path}")
    logger.info(f"  Projector      : {proj_ckpt_path}")
    logger.info(f"  LoRA adapter   : {lora_dir}")

    # 2. Patch adapter config for PEFT compatibility
    _patch_adapter_config(lora_dir)

    # 3. Load tokenizer from Qwen (cached)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_ID)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. Load LLM + LoRA
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"Loading base LLM ({BASE_LLM_ID}) ...")
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_LLM_ID,
        torch_dtype=dtype,
    )
    logger.info("Applying LoRA adapter ...")
    llm = PeftModel.from_pretrained(llm, lora_dir)
    llm = llm.merge_and_unload()  # Merge LoRA into base weights for faster inference
    llm = llm.to(device)
    llm.eval()

    if hasattr(llm, "generation_config"):
        llm.generation_config.use_cache = True
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    # 5. Load Vision Encoder
    logger.info("Loading Vision Encoder ...")
    vision_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
    )
    ckpt = torch.load(vit_ckpt_path, weights_only=False)

    full_state = ckpt.get("model_state", ckpt)
    state_dict = {
        k.replace("vit_backbone.", ""): v
        for k, v in full_state.items()
        if k.startswith("vit_backbone.")
    }

    vision_encoder.load_state_dict(state_dict, strict=False)

    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    # 6. Assemble VLM
    model = VLMModel(
        vision_encoder=vision_encoder,
        llm=llm,
        img_placeholder_id=IMG_PLACEHOLDER_ID,
        expansion_factor=2,
    ).to(device)

    # 7. Load Projector
    logger.info("Loading MLP Projector ...")
    proj_state = torch.load(proj_ckpt_path, map_location=device, weights_only=False)
    if "projector_state_dict" in proj_state:
        proj_state = proj_state["projector_state_dict"]
    model.projector.load_state_dict(proj_state)

    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


# Main inference loop
@torch.no_grad()
def run_inference(model, tokenizer, questions, device, dtype, batch_size=4):
    """
    Runs batched inference over all questions.
    Returns a dict: { str(question_index): {"reasoning": ..., "answer": ...} }
    """
    predictions = {}
    n = len(questions)

    for start_idx in tqdm(range(0, n, batch_size), desc="Inference"):
        batch = questions[start_idx : start_idx + batch_size]

        # Load and preprocess images
        images = []
        for q in batch:
            img = Image.open(q["image_path"]).convert("RGB")
            images.append(EVAL_TRANSFORM(img))
        images_tensor = torch.stack(images).to(device)

        # Build tokenized prompts
        q_texts = [q["question"] for q in batch]
        input_ids, attention_mask = build_inference_inputs(q_texts, tokenizer, device)

        # Generate
        autocast_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
        with torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")
        ):
            output_ids = model.generate(
                images=images_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for q, gen_text in zip(batch, generated_texts):
            q_idx = str(q["question_index"])
            reasoning, answer = parse_generated_text(gen_text)
            predictions[q_idx] = {
                "reasoning": reasoning,
                "answer": answer,
            }

    return predictions


# Entry point
def main():
    parser = argparse.ArgumentParser(description="Part B VLM Inference for CLEVR QA")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory containing ViT, projector, LoRA adapter, etc.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to CLEVR questions JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save QA predictions JSON",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Inference batch size"
    )
    args = parser.parse_args()

    # Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    logger.info(f"Device: {device} | dtype: {dtype}")

    # Load data
    logger.info(f"Loading questions from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = (
        data["questions"] if isinstance(data, dict) and "questions" in data else data
    )
    logger.info(f"  Total questions: {len(questions)}")

    # Load model
    model, tokenizer = load_model(args.model_dir, device, dtype)

    # Run inference
    predictions = run_inference(
        model, tokenizer, questions, device, dtype, batch_size=args.batch_size
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions saved to {args.output_file} ({len(predictions)} entries)")


if __name__ == "__main__":
    main()
