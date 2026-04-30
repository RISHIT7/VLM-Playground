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

from models.vlm import VLMModel
from models.vit_backbone import VisionTransformer
from data.clevr_vlm_dataset import CLEVRCaptionDataset, CLEVRQADataset, VLMCollateFn
from utils.wandb_logger import WandbLogger

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model_id, 
        torch_dtype=dtype,
    ).to(device)
    
    llm.gradient_checkpointing_enable()

    vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    vision_encoder.load_state_dict(torch.load(vit_checkpoint_path, map_location="cpu")['student_network_state'], strict=False)
    vision_encoder = vision_encoder.to(device)
    vision_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none", task_type="CAUSAL_LM"
    )
    llm = get_peft_model(llm, lora_config)
    
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
    collate_fn = VLMCollateFn(tokenizer, mode="stage2")
    
    sampler = DistributedSampler(train_ds, num_replicas=cfg.num_gpus, rank=rank, shuffle=True) if is_multi_gpu else None
    
    train_dl = DataLoader(
        train_ds, batch_size=cfg.stage2_per_device_bs, 
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
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

            with torch.cuda.amp.autocast(dtype=dtype):
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
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            
            unwrapped_model = model.module if hasattr(model, "module") else model
            
            torch.save(unwrapped_model.projector.state_dict(), os.path.join(cfg.checkpoint_dir, f"vlm_stage2_proj_ep{epoch+1}.pt"))
            unwrapped_model.llm.save_pretrained(os.path.join(cfg.checkpoint_dir, f"vlm_stage2_lora_ep{epoch+1}"))

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
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info(f"[Stage 1] Initializing Core Alignment on {cfg.num_gpus} GPU(s)...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model_id, 
        torch_dtype=dtype,
    ).to(device)
    
    llm.requires_grad_(False)
    llm.gradient_checkpointing_enable()

    vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    vision_encoder.load_state_dict(torch.load(vit_checkpoint_path, map_location="cpu")['student_network_state'], strict=False)
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
    collate_fn = VLMCollateFn(tokenizer, mode="stage1")
    
    sampler = DistributedSampler(train_ds, num_replicas=cfg.num_gpus, rank=rank, shuffle=True) if is_multi_gpu else None
    
    train_dl = DataLoader(
        train_ds, batch_size=cfg.stage1_per_device_bs, 
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.stage1_lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_dl) * cfg.stage1_epochs // cfg.stage1_grad_accum
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

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

            with torch.cuda.amp.autocast(dtype=dtype):
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