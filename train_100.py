import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm   
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast

from mv_mae.data.dataset import UAVHumanDataset
from mv_mae.models import MVMAE
from mv_mae.utils import setup_logger, get_latest_checkpoint, save_checkpoint, load_checkpoint

# ==============================================================================
# MASTER CONFIGURATION (DEBUG MODE)
# ==============================================================================
class Config:
    data_root = 'datasets/UAVHuman_240p_mp4/Action_Videos'
    train_split = 'datasets/UAVHuman_240p_mp4/train_split.txt'
    val_split = 'datasets/UAVHuman_240p_mp4/val_split.txt'
    checkpoint_dir = './checkpoints_debug' # Save to a different folder so we don't overwrite your real weights
    
    num_classes = 155
    num_segments = 8
    gop_size = 16
    
    epochs = 2          # Just run 2 epochs to test the loop-around
    lr = 5e-5
    weight_decay = 0.05
    resume = False      # Start fresh for the debug test
    
    batch_size = 2
    accumulation_steps = 16  
    num_workers = 8
    prefetch_factor = 3

args = Config()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

def calculate_accuracy(output, target, topk=(1,)):
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
    logger = setup_logger(args.checkpoint_dir, log_filename="training_debug.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_dataset = UAVHumanDataset(
        data_root=args.data_root, split_file=args.train_split, 
        num_segments=args.num_segments, gop_size=args.gop_size, is_train=True
    )
    val_dataset = UAVHumanDataset(
        data_root=args.data_root, split_file=args.val_split, 
        num_segments=args.num_segments, gop_size=args.gop_size, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, 
        prefetch_factor=args.prefetch_factor, persistent_workers=True,
        drop_last=True # <--- THE BATCHNORM FIX
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=args.prefetch_factor, persistent_workers=True
    )

    logger.info("Initializing the MV_MAE model...")
    model = MVMAE(num_classes=args.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0

    logger.info("=========================================")
    logger.info("     STARTING 100-VIDEO SMOKE TEST       ")
    logger.info("=========================================")

    accumulation_steps = args.accumulation_steps 

    for epoch in range(start_epoch, args.epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_top1 = 0.0
        optimizer.zero_grad(set_to_none=True) 
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")

        for batch_idx, (iframes, mvs, labels) in enumerate(train_bar):
            # ---------------------------------------------------------
            # FAST TEST BREAK: Stop training after 50 batches (100 videos)
            # ---------------------------------------------------------
            if batch_idx >= 50:
                logger.info("DEBUG: Processed 50 batches. Jumping to validation.")
                break

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
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  ")

        with torch.no_grad():
            for batch_idx, (iframes, mvs, labels) in enumerate(val_bar):
                # ---------------------------------------------------------
                # FAST TEST BREAK: Stop validation after 20 batches (40 videos)
                # ---------------------------------------------------------
                if batch_idx >= 20:
                    break

                iframes = iframes.to(device, non_blocking=True)
                mvs = mvs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(iframes, mvs)
                    loss = criterion(logits, labels)

                acc1, _ = calculate_accuracy(logits, labels, topk=(1,5))
                val_loss += loss.item()
                val_top1 += acc1
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc1:.2f}%")
                
        # --- EPOCH SUMMARY ---
        # Note: We divide by 20 and 50 here just for accurate debug printouts
        avg_val_loss = val_loss / 20 
        avg_val_acc = val_top1 / 20
        avg_train_loss = train_loss / 50
        avg_train_acc = train_top1 / 50

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
        
        save_checkpoint(state, is_best, ckpt_dir=args.checkpoint_dir, filename=f"checkpoint_debug_{epoch+1}.pth")

if __name__ == "__main__":
    main()