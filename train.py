import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from mv_mae.data.dataset import UAVHumanDataset
from mv_mae.models import MVMAE
from mv_mae.utils import setup_logger, get_latest_checkpoint, save_checkpoint, load_checkpoint

# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================
class Config:
    # --- Paths ---
    data_root = 'datasets/UAVHuman_240p_mp4/Action_Videos'
    train_split = 'datasets/UAVHuman_240p_mp4/train_split.txt'
    val_split = 'datasets/UAVHuman_240p_mp4/val_split.txt'
    checkpoint_dir = './checkpoints'
    
    # --- Dataset & Architecture ---
    num_classes = 155
    num_segments = 8
    gop_size = 16
    
    # --- Training Hyperparameters ---
    epochs = 50
    lr = 5e-5
    weight_decay = 0.05
    resume = True
    
    # --- Hardware & Cloud Optimization ---
    batch_size = 8
    accumulation_steps = 4  # batch=8 is too low, so we accumulate, eventually making it look like 32 batches
    num_workers = 8
    log_interval = 20

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
        gop_size=args.gop_size
    )
    val_dataset = UAVHumanDataset(
        data_root=args.data_root, 
        split_file=args.val_split, 
        num_segments=args.num_segments, 
        gop_size=args.gop_size
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        prefetch_factor=3,
        persistent_workers=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )

    logger.info("Initializing the MV_MAE model...")
    model = MVMAE(num_classes=args.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_acc = 0.0
    latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt and args.resume:
        logger.info(f"Found checkpoint : {latest_ckpt}")
        start_epoch, best_acc = load_checkpoint(latest_ckpt, model, optimizer, device=device)

    logger.info("=========================================")
    logger.info("          STARTING TRAINING              ")
    logger.info("=========================================")

    accumulation_steps = args.accumulation_steps 

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_top1 = 0.0
        
        optimizer.zero_grad(set_to_none=True) 

        for batch_idx, (iframes, mvs, labels) in enumerate(train_loader):
            iframes = iframes.to(device, non_blocking=True)
            mvs = mvs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(iframes, mvs)
                loss = criterion(logits, labels) / accumulation_steps

            # Native backward pass (no scaler needed for bfloat16)
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1,5))
            
            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss
            train_top1 += acc1

            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                            f"Loss: {current_loss:.4f} | Top-1 Acc: {acc1:.2f}%")

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_top1 = 0.0

        with torch.no_grad():
            for iframes, mvs, labels in val_loader:
                iframes = iframes.to(device, non_blocking=True)
                mvs = mvs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(iframes, mvs)
                    loss = criterion(logits, labels)

                acc1, _ = calculate_accuracy(logits, labels, topk=(1,5))
                val_loss += loss.item()
                val_top1 += acc1
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_top1 / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_top1 / len(train_loader)

        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Train Acc:  {avg_train_acc:.2f}% | Val Acc:  {avg_val_acc:.2f}%")
        
        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc
            logger.info("🌟 New best validation accuracy! Saving model...")
            
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        
        save_checkpoint(state, is_best, ckpt_dir=args.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()