# sft process 
# train.py — 2-Phase SFT Training
import os

# Stop FFmpeg and Numpy from spawning hundreds of conflicting threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FFMPEG_THREADS"] = "1"

import logging
import torch
import torch.nn as nn
from tqdm import tqdm   
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score

from mv_mae.data.dataset import UAVHumanDataset
from mv_mae.models import MVMAE
from mv_mae.utils import setup_logger, get_latest_checkpoint, save_checkpoint, load_checkpoint

# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================
class Config:
    # --- Paths ---
    data_root = 'datasets/UAVHuman_480p_mp4/Action_Videos'
    train_split = 'datasets/UAVHuman_480p_mp4/train_split.txt'
    val_split = 'datasets/UAVHuman_480p_mp4/val_split.txt'
    checkpoint_dir = './checkpoints'
    pretrain_ckpt_path = 'pretraining/checkpoints_pretrain/dmvmae_272_epoch_pretrain/latest_checkpoint.pth'
    
    # --- Dataset & Architecture ---
    num_classes = 155
    num_segments = 8
    gop_size = 16
    
    # --- Training Hyperparameters ---
    epochs = 50
    lr = 5e-5              # LR for Temporal Transformer + Head
    lr_encoder = 1e-5      # Lower LR for unfrozen encoder layers (Phase 2)
    weight_decay = 0.05
    resume = True
    
    # --- 2-Phase SFT ---
    warmup_epochs = 5             # Phase 1 duration (encoders frozen)
    encoder_unfreeze_layers = 2   # Top N layers to unfreeze in Phase 2
    
    # --- Hardware & Cloud Optimization ---
    batch_size = 1
    accumulation_steps = 32  # Simulates a batch size of 32
    num_workers = 8
    prefetch_factor = 2

# Initialize config so it is globally accessible throughout the script
args = Config()

# ==============================================================================
# HARDWARE OPTIMIZATIONS (Ada Lovelace / L4 GPU)
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def build_optimizer_and_scheduler(model, phase, remaining_epochs):
    """Build optimizer and scheduler for the given phase."""
    if phase == 1:
        # Phase 1: Only head/temporal params, single LR
        param_groups = model.get_param_groups(
            lr_head=args.lr, lr_encoder=args.lr_encoder, weight_decay=args.weight_decay
        )
    else:
        # Phase 2: Head + unfrozen encoder params, dual LR
        param_groups = model.get_param_groups(
            lr_head=args.lr, lr_encoder=args.lr_encoder, weight_decay=args.weight_decay
        )
    
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    return optimizer, scheduler

def main():
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(args.checkpoint_dir, log_filename="training.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_dataset = UAVHumanDataset(
        data_root=args.data_root, 
        split_file=args.train_split, 
        num_segments=args.num_segments, 
        gop_size=args.gop_size,
        is_train=True
    )
    
    val_dataset = UAVHumanDataset(
        data_root=args.data_root, 
        split_file=args.val_split, 
        num_segments=args.num_segments, 
        gop_size=args.gop_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )

    logger.info("Initializing the MV_MAE model...")
    model = MVMAE(num_classes=args.num_classes, pretrain_ckpt_path=args.pretrain_ckpt_path).to(device)
    
    criterion = nn.CrossEntropyLoss()

    # --- Determine starting state ---
    start_epoch = 0
    best_acc = 0.0
    current_phase = 1

    latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt and args.resume:
        logger.info(f"Found checkpoint: {latest_ckpt}")
        start_epoch, best_acc, current_phase = load_checkpoint(latest_ckpt, model, device=device)
    
    # --- Apply freeze state based on current phase ---
    if current_phase == 1:
        logger.info("=" * 50)
        logger.info("  PHASE 1: WARMUP (Encoders Frozen)")
        logger.info("=" * 50)
        model.freeze_encoders()
        remaining = args.warmup_epochs - start_epoch
    else:
        logger.info("=" * 50)
        logger.info("  PHASE 2: END-TO-END FINE-TUNING")
        logger.info("=" * 50)
        model.freeze_encoders()  # Freeze all first
        model.unfreeze_encoder_top_layers(
            ctx_top_n=args.encoder_unfreeze_layers,
            mot_top_n=args.encoder_unfreeze_layers
        )
        remaining = args.epochs - start_epoch

    # Build optimizer and scheduler for current phase
    optimizer, scheduler = build_optimizer_and_scheduler(model, current_phase, max(remaining, 1))

    # Load optimizer state if resuming
    if latest_ckpt and args.resume:
        ckpt = torch.load(latest_ckpt, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                logger.info("Resumed optimizer state.")
            except ValueError:
                logger.warning("Optimizer state mismatch (phase transition?). Starting fresh optimizer.")
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                logger.info("Resumed scheduler state.")
            except Exception:
                logger.warning("Scheduler state mismatch. Starting fresh scheduler.")

    logger.info("=========================================")
    logger.info("          STARTING TRAINING              ")
    logger.info(f"  Phase: {current_phase} | Epoch: {start_epoch} -> {args.epochs}")
    logger.info(f"  Batch: {args.batch_size} x {args.accumulation_steps} accum = {args.batch_size * args.accumulation_steps} effective")
    logger.info("=========================================")

    accumulation_steps = args.accumulation_steps 

    for epoch in range(start_epoch, args.epochs):
        
        # === PHASE TRANSITION: Phase 1 -> Phase 2 ===
        if current_phase == 1 and epoch >= args.warmup_epochs:
            current_phase = 2
            logger.info("=" * 50)
            logger.info("  PHASE 2: END-TO-END FINE-TUNING (Transition)")
            logger.info("=" * 50)
            
            # Unfreeze top encoder layers
            model.unfreeze_encoder_top_layers(
                ctx_top_n=args.encoder_unfreeze_layers,
                mot_top_n=args.encoder_unfreeze_layers
            )
            
            # Rebuild optimizer and scheduler for Phase 2
            remaining_epochs = args.epochs - epoch
            optimizer, scheduler = build_optimizer_and_scheduler(model, current_phase, remaining_epochs)
            logger.info(f"Rebuilt optimizer & scheduler for Phase 2 ({remaining_epochs} remaining epochs)")

        # --- TRAINING PHASE ---
        model.train()
        # Keep frozen encoders in eval mode (important for BatchNorm/Dropout)
        if current_phase == 1:
            model.ctx_encoder.eval()
            model.mot_encoder.eval()
        elif current_phase == 2:
            model.ctx_encoder.eval()  # Keep eval for frozen bottom layers
            model.mot_encoder.eval()
        
        train_loss = 0.0
        train_top1 = 0.0
        
        optimizer.zero_grad(set_to_none=True) 

        train_bar = tqdm(train_loader, desc=f"[P{current_phase}] Epoch [{epoch+1}/{args.epochs}] Train")

        for batch_idx, (iframes, mvs, labels) in enumerate(train_bar):
            iframes = iframes.to(device, non_blocking=True)
            mvs = mvs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(iframes, mvs)
                loss = criterion(logits, labels) / accumulation_steps

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1,5))
            
            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss
            train_top1 += acc1

            train_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{acc1:.2f}%")

        scheduler.step()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        total_samples = 0  # Added to track exact sample count
        all_preds = []
        all_labels = []

        val_bar = tqdm(val_loader, desc=f"[P{current_phase}] Epoch [{epoch+1}/{args.epochs}] Val  ")

        with torch.no_grad():
            for iframes, mvs, labels in val_bar:
                iframes = iframes.to(device, non_blocking=True)
                mvs = mvs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Get the actual size of the current batch
                batch_size = iframes.size(0)

                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(iframes, mvs)
                    loss = criterion(logits, labels)

                acc1, _ = calculate_accuracy(logits, labels, topk=(1,5))
                
                # Ensure acc1 is a float, not a tensor
                if isinstance(acc1, torch.Tensor):
                    acc1 = acc1.item()

                # Multiply by batch_size to get the true sum for accurate epoch averaging
                val_loss += loss.item() * batch_size
                val_top1 += acc1 * batch_size
                total_samples += batch_size

                # Collect predictions
                preds = logits.argmax(dim=1).cpu()
                
                # Secure labels for scikit-learn (handles 2D smoothed labels or 1D hard labels)
                if labels.ndim > 1:
                    labels_1d = labels.argmax(dim=1).cpu()
                else:
                    labels_1d = labels.cpu()
                    
                all_preds.append(preds)
                all_labels.append(labels_1d)
                
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc1:.2f}%")

        # Compute Macro F1, Precision, Recall over entire validation set
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                
        # --- EPOCH SUMMARY ---
        # Divide by total_samples, NOT len(val_loader)
        avg_val_loss = val_loss / total_samples
        avg_val_acc = val_top1 / total_samples
        
        # Note: Ensure train_loss and train_top1 were also averaged using total_train_samples
        avg_train_loss = train_loss / len(train_loader) 
        avg_train_acc = train_top1 / len(train_loader)

        logger.info(f"--- Epoch {epoch+1} Summary (Phase {current_phase}) ---")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Train Acc:  {avg_train_acc:.2f}% | Val Acc:  {avg_val_acc:.2f}%")
        logger.info(f"Macro F1: {macro_f1:.4f} | Precision: {macro_precision:.4f} | Recall: {macro_recall:.4f}")
        
        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc
            logger.info("🌟 New best validation accuracy! Saving model...")
            
        state = {
            'epoch': epoch + 1,
            'phase': current_phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        
        save_checkpoint(state, is_best, ckpt_dir=args.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()