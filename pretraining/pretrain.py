import os
import sys

# Stop FFmpeg and Numpy from spawning hundreds of conflicting threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FFMPEG_THREADS"] = "1"

import logging
import math
import torch
import torch.nn as nn
from tqdm import tqdm   
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast

# --- IMPORT YOUR CUSTOM MODULES ---
# Add the root repository folder to Python path so 'pretraining' and 'mv_mae' can be imported easily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pretraining.data.dataset import UAVHumanDataset
from pretraining.models.model_wrapper import DiscreteMVMAE
from pretraining.models.tokenizer import MVCodebookTokenizer  # IMPORTING YOUR REAL TOKENIZER

# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================
class Config:
    # --- Paths ---
    data_root = '../datasets/UAVHuman_480p_mp4/Action_Videos'
    train_split = '../datasets/UAVHuman_480p_mp4/train_split.txt'
    checkpoint_dir = 'checkpoints_pretrain'
    codebook_path = 'codebook_ckpt/mv_codebook_1024.pt'
    
    # --- Dataset & Architecture ---
    num_segments = 8
    gop_size = 16
    codebook_size = 1024
    mask_ratio = 0.9      # 90% Masking for 2x efficiency
    
    # --- Pre-training Hyperparameters ---
    epochs = 400
    lr = 3e-4             # Higher base LR for training from scratch
    min_lr = 1e-6
    warmup_epochs = 20    # Mandatory for blank Vision Transformers
    weight_decay = 0.05
    resume = True
    
    # --- Hardware & Cloud Optimization ---
    batch_size = 8
    accumulation_steps = 4  # Effective batch size = 32
    num_workers = 8
    prefetch_factor = 3

args = Config()

# ==============================================================================
# HARDWARE OPTIMIZATIONS
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# ==============================================================================
# CUSTOM LR SCHEDULER (Cosine with Linear Warmup)
# ==============================================================================
class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    """ Slowly warms up the learning rate to prevent exploding gradients, then decays. """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * float(step) / float(max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        # Cosine decay
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def setup_logger(save_dir, log_filename="pretraining.log"):
    logger = logging.getLogger("MVMAE_Pretrain")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(os.path.join(save_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def main():
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(args.checkpoint_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

    logger.info("Loading Unlabeled Pre-training Dataset...")
    train_dataset = UAVHumanDataset(
        data_root=args.data_root, 
        split_file=args.train_split, 
        num_segments=args.num_segments, 
        gop_size=args.gop_size,
        is_train=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        drop_last=True,
        timeout=10  # Stop workers from deadlocking on bad videos
    )

    logger.info("Initializing Discrete MVMAE...")
    model = DiscreteMVMAE(codebook_size=args.codebook_size).to(device)
    
    logger.info("Initializing Real Codebook Tokenizer...")
    # Initialize your tokenizer here. Make sure it loads the .pt file!
    tokenizer = MVCodebookTokenizer(codebook_path=args.codebook_path, device=device)
    # Ensure tokenizer is in eval mode so we don't accidentally update the codebook
    tokenizer.eval() 
    
    # Decoupled weight decay using AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    # Calculate exact step counts for the per-batch scheduler
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=args.min_lr)

    start_epoch = 0
    # Add your load_checkpoint logic here if args.resume is True

    logger.info("=========================================")
    logger.info("       STARTING MVMAE PRE-TRAINING       ")
    logger.info(f"       Masking Ratio: {args.mask_ratio*100}% ")
    logger.info("=========================================")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True) 
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        # Unpack Data: We IGNORE iframes and labels for self-supervised pre-training
        for batch_idx, (_, mvs, _) in enumerate(train_bar):
            mvs = mvs.to(device, non_blocking=True) # [B, 8, 2, 16, 14, 14]

            # 1. GENERATE GROUND TRUTH TARGETS
            # Use the frozen codebook tokenizer to get the target IDs
            with torch.no_grad():
                # Depending on how your tokenizer.py is written, this might be `tokenizer(mvs)`
                target_ids = tokenizer.tokenize(mvs) # Expected shape: [B, 1568]

            # 2. FORWARD PASS (Masking happens automatically inside)
            with autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model(mvs, target_ids, mask_ratio=args.mask_ratio)
                scaled_loss = loss / args.accumulation_steps

            # 3. BACKWARD PASS
            scaled_loss.backward()

            # 4. OPTIMIZER STEP & SCHEDULER (with Gradient Accumulation)
            if ((batch_idx + 1) % args.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                # Gradient clipping is vital when training ViTs from scratch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step() # Note: The scheduler steps PER BATCH, not per epoch
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item()
            train_loss += current_loss
            
            # Live progress bar updates
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

        # --- EPOCH SUMMARY ---
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"--- Epoch {epoch+1} Summary --- | Avg Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save Checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            save_path = os.path.join(args.checkpoint_dir, f"mvmae_epoch_{epoch+1}.pth")
            torch.save(state, save_path)
            logger.info(f"Checkpoint Saved: {save_path}")

if __name__ == "__main__":
    main()