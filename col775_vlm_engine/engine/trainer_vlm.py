import os
import torch
import logging
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
from unsloth import FastLanguageModel

from models.vlm import VLMModel
from models.vit_backbone import VisionTransformer
from data.clevr_vlm_dataset import CLEVRCaptionDataset, CLEVRQADataset, VLMCollateFn
from utils.wandb_logger import WandbLogger
from engine.evaluator import evaluate_vlm_stage1, evaluate_vlm_stage2

logger = logging.getLogger(__name__)

def setup_ddp(rank: int, world_size: int):
    """Initializes the Distributed Data Parallel process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Destroys the DDP process group."""
    dist.destroy_process_group()

def stage2_worker(rank: int, cfg, vit_checkpoint_path: str, stage1_ckpt_path: str):
    """
    The actual training loop running on each individual GPU.
    """
    is_multi_gpu = cfg.num_gpus > 1
    if is_multi_gpu:
        setup_ddp(rank, cfg.num_gpus)
        
    device = torch.device(f"cuda:{rank}")
    
    is_main_process = (rank == 0)
    if is_main_process:
        logger.info(f"[Stage 2] Initializing on {cfg.num_gpus} GPU(s)...")

    # Detect if we should use the model's native dtype (e.g., for FP8 or pre-quantized models)
    if "FP8" in cfg.llm_model_id.upper() or "INT8" in cfg.llm_model_id.upper():
        dtype = "auto"
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    max_seq_length = 2048 # Define a safe max sequence length
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.llm_model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False, # Set to True if you ever want extreme VRAM savings
    )
    if dtype != "auto":
        llm = llm.to(device)

    vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    
    ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
    full_state = ckpt.get('model_state', ckpt)
    state_dict = {k.replace("vit_backbone.", ""): v for k, v in full_state.items() if k.startswith("vit_backbone.")}
    
    msg = vision_encoder.load_state_dict(state_dict, strict=False)
    if is_main_process:
        logger.info(f"[Stage 2] Loaded CLIP Vision Backbone with message: {msg}")

    vision_encoder = vision_encoder.to(device)
    vision_encoder.requires_grad_(False)

    # Unsloth's optimized LoRA injection
    llm = FastLanguageModel.get_peft_model(
        llm,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth", # <-- This is the magic Unsloth VRAM saver
        random_state=42,
    )
    
    if is_main_process: llm.print_trainable_parameters()

    model = VLMModel(
        vision_encoder=vision_encoder, llm=llm, 
        img_placeholder_id=(tokenizer.unk_token_id or 0),
        expansion_factor=cfg.expansion_factor
    ).to(device)

    stage1_ckpt = torch.load(stage1_ckpt_path, map_location=device)
    model.projector.load_state_dict(stage1_ckpt['projector_state_dict'])
    for param in model.projector.parameters():
        param.requires_grad = True

    if is_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    train_ds = CLEVRQADataset(cfg, split="train", tokenizer=tokenizer)
    val_ds = CLEVRQADataset(cfg, split="val", tokenizer=tokenizer)
    
    collate_fn = VLMCollateFn(tokenizer, mode="stage2")
    val_collate_fn = VLMCollateFn(tokenizer, mode="eval_stage2")
    
    sampler = DistributedSampler(train_ds, num_replicas=cfg.num_gpus, rank=rank, shuffle=True) if is_multi_gpu else None
    
    train_dl = DataLoader(
        train_ds, batch_size=cfg.stage2_per_device_bs, 
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
    )
    
    val_dl = DataLoader(
        val_ds, batch_size=cfg.stage2_per_device_bs, 
        shuffle=False, num_workers=cfg.num_workers, 
        pin_memory=True, collate_fn=val_collate_fn, drop_last=False
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_dl) * cfg.stage2_epochs // cfg.stage2_grad_accum
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    if is_main_process:
        wandb_logger = WandbLogger(project_name=cfg.wandb_project, config=cfg.__dict__, run_name=f"vlm-stage2-{cfg.num_gpus}gpu")
        logger.info(f"Effective Batch Size: {cfg.stage2_per_device_bs * cfg.stage2_grad_accum * cfg.num_gpus}")

    global_step = 0
    for epoch in range(cfg.stage2_epochs):
        if is_multi_gpu: sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Stage 2 | Ep {epoch+1}", disable=not is_main_process)
        
        for step, batch in enumerate(pbar):
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Fallback dtype for autocast if model is in 'auto' (FP8) mode
            autocast_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / cfg.stage2_grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % cfg.stage2_grad_accum == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                
                if is_main_process and global_step % cfg.log_every_n_steps == 0:
                    wandb_logger.log_metrics({
                        "stage2/train_loss": outputs.loss.item(),
                        "stage2/lr": scheduler.get_last_lr()[0]
                    }, step=global_step)

            epoch_loss += outputs.loss.item()
            if is_main_process: pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        if is_main_process:
            # Validation
            logger.info(f"[Stage 2] Starting validation for Epoch {epoch+1}...")
            val_metrics = evaluate_vlm_stage2(model, val_dl, device, tokenizer)
            logger.info(f"[Stage 2] Epoch {epoch+1} Val Exact-Match: {val_metrics['exact_match']:.4f}")
            wandb_logger.log_metrics({
                "stage2/val_exact_match": val_metrics["exact_match"],
            }, step=global_step)
            
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            
            unwrapped_model = model.module if hasattr(model, "module") else model
            
            # Save Projector
            torch.save(unwrapped_model.projector.state_dict(), os.path.join(cfg.checkpoint_dir, f"vlm_stage2_proj_ep{epoch+1}.pt"))
            
            # Save Unsloth LoRA Adapters
            lora_path = os.path.join(cfg.checkpoint_dir, f"vlm_stage2_lora_ep{epoch+1}")
            unwrapped_model.llm.save_pretrained(lora_path) # Unsloth safely patches this

    if is_multi_gpu: cleanup_ddp()
    if is_main_process: wandb_logger.finish()

def train_vlm_stage2_launcher(cfg, vit_checkpoint_path: str, stage1_ckpt_path: str):
    """
    API Description:
    The main entry point. Automatically detects 1 vs N GPUs and routes the execution.
    """
    if cfg.num_gpus > 1:
        mp.spawn(stage2_worker, nprocs=cfg.num_gpus, args=(cfg, vit_checkpoint_path, stage1_ckpt_path))
    else:
        stage2_worker(0, cfg, vit_checkpoint_path, stage1_ckpt_path)

def stage1_worker(rank: int, cfg, vit_checkpoint_path: str):
    """
    The training loop for Stage 1 Core Alignment (Image Captioning).
    Only the MLP Projector is trained. LLM and ViT are frozen[cite: 8].
    """
    is_multi_gpu = cfg.num_gpus > 1
    if is_multi_gpu:
        setup_ddp(rank, cfg.num_gpus)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(cfg.device)
        
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info(f"[Stage 1] Initializing Core Alignment on {cfg.num_gpus} GPU(s)...")

    # Determine dtype based on device
    if "FP8" in cfg.llm_model_id.upper() or "INT8" in cfg.llm_model_id.upper():
        dtype = "auto"
    elif device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16 # MPS prefers float16
    else:
        dtype = torch.float32 # CPU fallback
        
    max_seq_length = 2048
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.llm_model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False, 
    )
    if dtype != "auto":
        llm = llm.to(device)
    
    llm.requires_grad_(False)
    llm.gradient_checkpointing_enable() # Standard HF checkpointing for frozen models

    vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    
    ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
    full_state = ckpt.get('model_state', ckpt)
    state_dict = {k.replace("vit_backbone.", ""): v for k, v in full_state.items() if k.startswith("vit_backbone.")}
    
    msg = vision_encoder.load_state_dict(state_dict, strict=False)
    if is_main_process:
        logger.info(f"[Stage 1] Loaded CLIP Vision Backbone with message: {msg}")
        
    vision_encoder = vision_encoder.to(device)
    vision_encoder.requires_grad_(False)

    model = VLMModel(
        vision_encoder=vision_encoder, 
        llm=llm, 
        img_placeholder_id=-200,
        expansion_factor=cfg.expansion_factor
    ).to(device)

    for name, param in model.named_parameters():
        if "projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if is_main_process:
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[Stage 1] Trainable Parameters (Projector Only): {trainable_params_count:,}")

    if is_multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    train_ds = CLEVRCaptionDataset(cfg, split="train", tokenizer=tokenizer)
    val_ds = CLEVRCaptionDataset(cfg, split="val", tokenizer=tokenizer)
    
    collate_fn = VLMCollateFn(tokenizer, mode="stage1")
    val_collate_fn = VLMCollateFn(tokenizer, mode="eval_stage1")
    
    sampler = DistributedSampler(train_ds, num_replicas=cfg.num_gpus, rank=rank, shuffle=True) if is_multi_gpu else None
    
    train_dl = DataLoader(
        train_ds, batch_size=cfg.stage1_per_device_bs, 
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
    )
    
    val_dl = DataLoader(
        val_ds, batch_size=cfg.stage1_per_device_bs, 
        shuffle=False, num_workers=cfg.num_workers, 
        pin_memory=True, collate_fn=val_collate_fn, drop_last=False
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_dl) * cfg.stage1_epochs // cfg.stage1_grad_accum
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # GradScaler only for CUDA
    use_scaler = (dtype == torch.float16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if is_main_process:
        wandb_logger = WandbLogger(project_name=cfg.wandb_project, config=cfg.__dict__, run_name=f"vlm-stage1-{cfg.num_gpus}gpu")
        logger.info(f"Effective Batch Size: {cfg.stage1_per_device_bs * cfg.stage1_grad_accum * cfg.num_gpus}")

    global_step = 0
    for epoch in range(cfg.stage1_epochs):
        if is_multi_gpu: sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Stage 1 | Ep {epoch+1}", disable=not is_main_process)
        
        for step, batch in enumerate(pbar):
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Use torch.amp.autocast for device-agnostic mixed precision
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            autocast_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
            with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=(dtype != torch.float32)):
                outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / cfg.stage1_grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % cfg.stage1_grad_accum == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                
                if is_main_process and global_step % cfg.log_every_n_steps == 0:
                    wandb_logger.log_metrics({
                        "stage1/train_loss": outputs.loss.item(),
                        "stage1/lr": scheduler.get_last_lr()[0]
                    }, step=global_step)

            epoch_loss += outputs.loss.item()
            if is_main_process: pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        if is_main_process:
            # Validation
            logger.info(f"[Stage 1] Starting validation for Epoch {epoch+1}...")
            val_metrics = evaluate_vlm_stage1(model, val_dl, device, tokenizer)
            logger.info(f"[Stage 1] Epoch {epoch+1} Val BLEU: {val_metrics['bleu']:.4f} | Exact-Match: {val_metrics.get('exact_match', 0.0):.4f}")
            wandb_logger.log_metrics({
                "stage1/val_bleu": val_metrics["bleu"],
                "stage1/val_exact_match": val_metrics.get("exact_match", 0.0),
            }, step=global_step)
            
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            unwrapped_model = model.module if hasattr(model, "module") else model
            
            proj_path = os.path.join(cfg.checkpoint_dir, f"vlm_stage1_proj_ep{epoch+1}.pt")
            torch.save({
                'projector_state_dict': unwrapped_model.projector.state_dict()
            }, proj_path)
            
            logger.info(f"Stage 1 checkpoint saved to {proj_path}")

    if is_multi_gpu: cleanup_ddp()
    if is_main_process: wandb_logger.finish()

def train_vlm_stage1_launcher(cfg, vit_checkpoint_path: str):
    """
    API Description:
    The main entry point. Automatically detects 1 vs N GPUs and routes the execution.
    """
    if cfg.num_gpus > 1:
        mp.spawn(stage1_worker, nprocs=cfg.num_gpus, args=(cfg, vit_checkpoint_path))
    else:
        stage1_worker(0, cfg, vit_checkpoint_path)

def eval_vlm_launcher(cfg, stage: int, ckpt_path: str, vit_checkpoint_path: str = None):
    device = torch.device(cfg.device if hasattr(cfg, 'device') else "cuda")
    logger.info(f"[VLM Eval] Starting evaluation for Stage {stage} on {device}")
    
    if "FP8" in cfg.llm_model_id.upper() or "INT8" in cfg.llm_model_id.upper():
        dtype = "auto"
    elif device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
        
    max_seq_length = 2048
    
    if stage == 1:
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.llm_model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )
        if dtype != "auto": llm = llm.to(device)
        llm.requires_grad_(False)
        
        vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        if vit_checkpoint_path and os.path.exists(vit_checkpoint_path):
            ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
            full_state = ckpt.get('model_state', ckpt)
            state_dict = {k.replace("vit_backbone.", ""): v for k, v in full_state.items() if k.startswith("vit_backbone.")}
            vision_encoder.load_state_dict(state_dict, strict=False)
        vision_encoder = vision_encoder.to(device)
        vision_encoder.requires_grad_(False)
        
        model = VLMModel(vision_encoder, llm, img_placeholder_id=-200, expansion_factor=cfg.expansion_factor).to(device)
        
        logger.info(f"[VLM Eval] Loading Stage 1 Projector from {ckpt_path}")
        proj_ckpt = torch.load(ckpt_path, map_location=device)
        if 'projector_state_dict' in proj_ckpt:
            model.projector.load_state_dict(proj_ckpt['projector_state_dict'])
        else:
            model.projector.load_state_dict(proj_ckpt)
            
        val_ds = CLEVRCaptionDataset(cfg, split="val", tokenizer=tokenizer)
        val_collate_fn = VLMCollateFn(tokenizer, mode="eval_stage1")
        val_dl = DataLoader(val_ds, batch_size=cfg.stage1_per_device_bs, shuffle=False, num_workers=cfg.num_workers, collate_fn=val_collate_fn)
        
        val_metrics = evaluate_vlm_stage1(model, val_dl, device, tokenizer)
        logger.info(f"[VLM Eval] Stage 1 Val BLEU: {val_metrics['bleu']:.4f} | Exact-Match: {val_metrics.get('exact_match', 0.0):.4f}")

    elif stage == 2:
        logger.info(f"[VLM Eval] Loading LLM with LoRA from {ckpt_path}")
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ckpt_path, 
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )
        if dtype != "auto": llm = llm.to(device)
        FastLanguageModel.for_inference(llm) 
        
        vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        if vit_checkpoint_path and os.path.exists(vit_checkpoint_path):
            ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
            full_state = ckpt.get('model_state', ckpt)
            state_dict = {k.replace("vit_backbone.", ""): v for k, v in full_state.items() if k.startswith("vit_backbone.")}
            vision_encoder.load_state_dict(state_dict, strict=False)
        vision_encoder = vision_encoder.to(device)
        vision_encoder.requires_grad_(False)
        
        model = VLMModel(vision_encoder, llm, img_placeholder_id=(tokenizer.unk_token_id or 0), expansion_factor=cfg.expansion_factor).to(device)
        
        proj_path = ckpt_path.replace("_lora_", "_proj_") + ".pt"
        if os.path.exists(proj_path):
            logger.info(f"[VLM Eval] Loading Projector from {proj_path}")
            model.projector.load_state_dict(torch.load(proj_path, map_location=device))
        else:
            logger.warning(f"[VLM Eval] Projector path not found at {proj_path}")
            
        val_ds = CLEVRQADataset(cfg, split="val", tokenizer=tokenizer)
        val_collate_fn = VLMCollateFn(tokenizer, mode="eval_stage2")
        val_dl = DataLoader(val_ds, batch_size=cfg.stage2_per_device_bs, shuffle=False, num_workers=cfg.num_workers, collate_fn=val_collate_fn)
        
        val_metrics = evaluate_vlm_stage2(model, val_dl, device, tokenizer)
        logger.info(f"[VLM Eval] Stage 2 Val Exact-Match: {val_metrics['exact_match']:.4f}")